import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from core.config import (  # noqa: E402
    embedding_model_name,
    image_embedding_model_name,
    nli_model_name,
    vectordb_path,
    eval_dataset_path,
)
from core.embedders import (  # noqa: E402
    EmbeddingManager,
    EmbeddingManager_Image,
    EmbeddingManager_Text_CLIP,
)
from core.retrievers import (  # noqa: E402
    RetrieverMultiModal_experimental,
    RetrieverMultiModal_ImagePageText,
    RetrieverMultiModal_ImageVLMCaptions,
    RetrieverMultiModal_ImageCLIPRepresentations,
)
from core.vectordb import VectorDBManager  # noqa: E402
from score_eval import (  # noqa: E402
    extract_gold_image_ids,
    compute_image_retrieval_metrics,
    build_nli_pipeline,
    entailment_score,
)
from utils_text import image_id_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval-only performance.")
    parser.add_argument("--dataset", default=str(eval_dataset_path))
    parser.add_argument("--run_dir", default="eval/runs_retrieval")
    parser.add_argument("--topk_text", type=int, default=3)
    parser.add_argument("--topk_image", type=int, default=3)
    parser.add_argument("--clip_top_k", type=int, default=10)
    parser.add_argument("--match_threshold_text", type=float, default=-0.5)
    parser.add_argument("--match_threshold_image", type=float, default=-0.6)
    parser.add_argument(
        "--filtered_image_retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, restrict image retrieval to documents that produced the retrieved text chunks.",
    )
    parser.add_argument("--k_list", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--nli_model", default=nli_model_name)
    parser.add_argument("--nli_max_text_docs", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--merge_types",
        nargs="+",
        default=["normalized_mean", "similarity_weighted", "query_interpolation", "pca"],
        help="Merge types to evaluate for clip_repr retriever.",
    )
    return parser.parse_args()


def load_dataset(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def extract_retrieved_image_ids(image_docs: list[dict]) -> list[set[str]]:
    ids = []
    for doc in image_docs:
        metadata = doc.get("metadata", {})
        path = metadata.get("image_path") or metadata.get("page_image_path") or metadata.get("image_rel_path")
        ids.append(image_id_candidates(path))
    return ids


def safe_retrieve(retriever, **kwargs):
    while True:
        try:
            return retriever.retrieve(**kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "merge_type" in msg:
                kwargs.pop("merge_type", None)
                continue
            if "clip_top_k" in msg:
                kwargs.pop("clip_top_k", None)
                continue
            if "filtered_image_retrieval" in msg:
                kwargs.pop("filtered_image_retrieval", None)
                continue
            if "query_image" in msg:
                kwargs.pop("query_image", None)
                continue
            raise


def build_retrievers():
    embedding_manager_txt = EmbeddingManager(model_name=embedding_model_name)
    embedding_manager_images = EmbeddingManager_Image(model_name=image_embedding_model_name)
    embedding_manager_text_clip = EmbeddingManager_Text_CLIP(model_name=image_embedding_model_name)

    vector_db_manager_pdf = VectorDBManager(
        collection_name="pdf_documents_db",
        directory=str(Path(vectordb_path) / "pdf_db/"),
        source_type="pdf",
    )
    vector_db_manager_pdf_images = VectorDBManager(
        collection_name="pdf_image_documents_db",
        directory=str(Path(vectordb_path) / "pdf_image_db/"),
        source_type="pdf_image",
    )
    vector_db_manager_pdf_images_page_text = VectorDBManager(
        collection_name="pdf_image_page_texts_db",
        directory=str(Path(vectordb_path) / "pdf_image_page_texts_db/"),
        source_type="pdf_image_page_texts",
    )
    vector_db_manager_pdf_images_vlm = VectorDBManager(
        collection_name="pdf_image_vlm_captions",
        directory=str(Path(vectordb_path) / "pdf_image_vlm_captions_db/"),
        source_type="pdf_image_vlm_captions",
    )
    vector_db_manager_pdf_text_clip = VectorDBManager(
        collection_name="pdf_text_clip_db",
        directory=str(Path(vectordb_path) / "pdf_text_clip_db/"),
        source_type="pdf_text_clip",
    )

    retrievers = {
        "clip": RetrieverMultiModal_experimental(
            vector_db_text=vector_db_manager_pdf,
            vector_db_image=vector_db_manager_pdf_images,
            embedding_manager_text=embedding_manager_txt,
            embedding_manager_image=embedding_manager_images,
        ),
        "page_text": RetrieverMultiModal_ImagePageText(
            vector_db_text=vector_db_manager_pdf,
            vector_db_image_text=vector_db_manager_pdf_images_page_text,
            embedding_manager_text=embedding_manager_txt,
        ),
        "vlm_caption": RetrieverMultiModal_ImageVLMCaptions(
            vector_db_text=vector_db_manager_pdf,
            vector_db_image_captions=vector_db_manager_pdf_images_vlm,
            embedding_manager_text=embedding_manager_txt,
        ),
        "clip_repr": RetrieverMultiModal_ImageCLIPRepresentations(
            vector_db_text=vector_db_manager_pdf,
            vector_db_text_clip=vector_db_manager_pdf_text_clip,
            vector_db_image=vector_db_manager_pdf_images,
            embedding_manager_text=embedding_manager_txt,
            embedding_manager_text_clip=embedding_manager_text_clip,
        ),
    }
    return retrievers


def summarize(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def run_retriever(
    run_dir: Path,
    name: str,
    retriever,
    dataset: list[dict],
    args: argparse.Namespace,
    merge_type: str | None = None,
    nli_pipe=None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = run_dir / name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    per_item_path = run_path / "per_item.jsonl"
    metrics_path = run_path / "metrics.json"
    metrics_txt_path = run_path / "metrics.txt"
    config_path = run_path / "run_config.json"

    config_out = {
        "retriever": name,
        "merge_type": merge_type,
        "dataset": str(args.dataset),
        "topk_text": args.topk_text,
        "topk_image": args.topk_image,
        "clip_top_k": args.clip_top_k,
        "match_threshold_text": args.match_threshold_text,
        "match_threshold_image": args.match_threshold_image,
        "filtered_image_retrieval": args.filtered_image_retrieval,
        "k_list": args.k_list,
        "nli_model": args.nli_model,
        "nli_max_text_docs": args.nli_max_text_docs,
        "limit": args.limit,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    summary = defaultdict(list)
    totals = defaultdict(int)
    missing_preds = 0
    per_item_records = []

    with per_item_path.open("w", encoding="utf-8") as out_f:
        for item in dataset:
            question = item.get("question", "")
            question_type = (item.get("question_type") or item.get("type") or "unknown").lower()
            gold_ids = extract_gold_image_ids(item)

            retrieve_kwargs = {
                "query": question,
                "query_image": question,
                "top_k_text": args.topk_text,
                "top_k_image": args.topk_image,
                "clip_top_k": args.clip_top_k,
                "match_threshold_text": args.match_threshold_text,
                "match_threshold_image": args.match_threshold_image,
                "filtered_image_retrieval": args.filtered_image_retrieval,
            }
            if merge_type is not None:
                retrieve_kwargs["merge_type"] = merge_type

            text_docs, image_docs = safe_retrieve(retriever, **retrieve_kwargs)

            retrieved_ids = extract_retrieved_image_ids(image_docs)
            metrics = {
                "id": item.get("id"),
                "question_type": question_type,
                "pred_answer": "",
                "gold_answer": item.get("answer") or item.get("gold_answer") or "",
            }

            if gold_ids:
                retrieval_scores = compute_image_retrieval_metrics(
                    retrieved_ids, gold_ids, args.k_list
                )
                metrics.update(retrieval_scores)
                for k in args.k_list:
                    summary[f"image_hit_at_{k}"].append(retrieval_scores[f"image_hit_at_{k}"])
                summary["image_mrr"].append(retrieval_scores["image_mrr"])
                totals["image_retrieval_count"] += 1

            if nli_pipe is not None:
                doc_texts = []
                for doc in text_docs[: args.nli_max_text_docs]:
                    doc_text = doc.get("document") if isinstance(doc, dict) else None
                    if doc_text:
                        doc_texts.append(doc_text)
                if doc_texts:
                    query_scores = []
                    for doc_text in doc_texts:
                        score_query = entailment_score(nli_pipe, doc_text, question)
                        if score_query is not None:
                            query_scores.append(score_query)
                    if query_scores:
                        metrics["nli_query_max"] = max(query_scores)
                        metrics["nli_query_mean"] = sum(query_scores) / len(query_scores)
                        metrics["nli_query_scores"] = query_scores
                        summary["nli_query_max"].append(metrics["nli_query_max"])
                        summary["nli_query_mean"].append(metrics["nli_query_mean"])
                    totals["nli_count"] += 1

            metrics["gold_image_ids"] = sorted(gold_ids)
            metrics["retrieved_image_ids"] = [sorted(list(ids)) for ids in retrieved_ids]
            metrics["retriever"] = name
            metrics["merge_type"] = merge_type
            metrics["n_text_docs"] = len(text_docs)
            metrics["n_image_docs"] = len(image_docs)

            out_f.write(json.dumps(metrics) + "\n")
            per_item_records.append(metrics)

    def mean(values: list) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    summary_out = {
        "dataset": str(args.dataset),
        "predictions": str(per_item_path),
        "missing_predictions": missing_preds,
        "counts": dict(totals),
        "metrics": {k: mean(v) for k, v in summary.items()},
        "retriever": name,
        "merge_type": merge_type,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    with metrics_txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {summary_out['dataset']}\n")
        f.write(f"Predictions: {summary_out['predictions']}\n")
        f.write(f"Missing predictions: {summary_out['missing_predictions']}\n")
        f.write("\nMetric definitions:\n")
        f.write("- nli_query_*: entailment score with premise = retrieved text chunk, hypothesis = original query (all questions)\n")
        f.write("\nCounts:\n")
        for key, value in sorted(summary_out["counts"].items()):
            f.write(f"- {key}: {value}\n")
        f.write("\nMetrics:\n")
        for key, value in sorted(summary_out["metrics"].items()):
            f.write(f"- {key}: {value}\n")

    return summary_out


def main() -> None:
    args = parse_args()
    dataset = load_dataset(Path(args.dataset), args.limit)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    retrievers = build_retrievers()
    nli_pipe = build_nli_pipeline(args.nli_model) if args.nli_model else None

    summaries = []

    summaries.append(
        run_retriever(
            run_dir=run_dir,
            name="clip",
            retriever=retrievers["clip"],
            dataset=dataset,
            args=args,
            merge_type=None,
            nli_pipe=nli_pipe,
        )
    )
    summaries.append(
        run_retriever(
            run_dir=run_dir,
            name="page_text",
            retriever=retrievers["page_text"],
            dataset=dataset,
            args=args,
            merge_type=None,
            nli_pipe=nli_pipe,
        )
    )
    summaries.append(
        run_retriever(
            run_dir=run_dir,
            name="vlm_caption",
            retriever=retrievers["vlm_caption"],
            dataset=dataset,
            args=args,
            merge_type=None,
            nli_pipe=nli_pipe,
        )
    )

    for merge_type in args.merge_types:
        name = f"clip_repr_{merge_type}"
        summaries.append(
            run_retriever(
                run_dir=run_dir,
                name=name,
                retriever=retrievers["clip_repr"],
                dataset=dataset,
                args=args,
                merge_type=merge_type,
                nli_pipe=nli_pipe,
            )
        )

    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    if summaries:
        header = list(summaries[0].keys())
        with summary_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in summaries:
                f.write(",".join("" if row[h] is None else str(row[h]) for h in header) + "\n")


if __name__ == "__main__":
    main()
