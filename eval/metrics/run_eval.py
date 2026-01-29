"""
CUDA_VISIBLE_DEVICES=0 python eval/metrics/run_eval.py \
  --dataset eval/datasets/dataset_mcq_fib.json \
  --run_dir eval/runs/clip_filtered \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --image_retrieval_mode vlm_caption \
  --filtered_image_retrieval \
  --merge_type normalized_mean \
  --topk_text 3 --topk_image 3 --max_images 3 \
  --match_threshold_text -0.5 --match_threshold_image -0.6 \
  --max_new_tokens 64

"""



import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from core.models_llm import (  # noqa: E402
    groq_api_key,
    get_groq_llm,
    load_llava_model,
    load_qwen_model,
)
from core.rag_pipelines import AdvancedMultimodalRAG  # noqa: E402
from core.retrievers import (  # noqa: E402
    RetrieverMultiModal_experimental,
    RetrieverMultiModal_ImagePageText,
    RetrieverMultiModal_ImageVLMCaptions,
    RetrieverMultiModal_ImageCLIPRepresentations,
)
from core.vectordb import VectorDBManager  # noqa: E402
from core.embedders import EmbeddingManager, EmbeddingManager_Image, EmbeddingManager_Text_CLIP  # noqa: E402
from core.config import (  # noqa: E402
    embedding_model_name,
    image_embedding_model_name,
    vectordb_path,
    eval_dataset_path,
    eval_runs_path,
)
from utils_text import parse_choice  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal RAG eval and log predictions.")
    parser.add_argument("--dataset", default=str(eval_dataset_path))
    parser.add_argument("--run_dir", default=str(eval_runs_path))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--preprocess_type", default=None)
    parser.add_argument("--summarize", action="store_true", default=False)
    parser.add_argument("--image_query_captioning", action="store_false", default=True)
    parser.add_argument(
        "--image_retrieval_mode",
        choices=["clip", "page_text", "vlm_caption", "clip_repr"],
        default="clip",
    )
    parser.add_argument(
        "--filtered_image_retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, restrict image retrieval to documents that produced the retrieved text chunks.",
    )
    parser.add_argument(
        "--merge_type",
        default="normalized_mean",
        choices=["normalized_mean", "similarity_weighted", "query_interpolation", "pca"],
        help="Merge strategy for CLIP-representation retrieval (clip_repr mode).",
    )
    parser.add_argument("--topk_text", type=int, default=3)
    parser.add_argument("--topk_image", type=int, default=3)
    parser.add_argument("--max_images", type=int, default=3)
    parser.add_argument("--match_threshold_text", type=float, default=-0.5)
    parser.add_argument("--match_threshold_image", type=float, default=-0.6)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser.parse_args()


def make_json_safe(obj, _seen=None):
    if _seen is None:
        _seen = set()
    if isinstance(obj, dict):
        obj_id = id(obj)
        if obj_id in _seen:
            return "<recursion>"
        _seen.add(obj_id)
        out = {str(k): make_json_safe(v, _seen) for k, v in obj.items()}
        _seen.remove(obj_id)
        return out
    if isinstance(obj, list):
        obj_id = id(obj)
        if obj_id in _seen:
            return "<recursion>"
        _seen.add(obj_id)
        out = [make_json_safe(v, _seen) for v in obj]
        _seen.remove(obj_id)
        return out
    if isinstance(obj, tuple):
        obj_id = id(obj)
        if obj_id in _seen:
            return "<recursion>"
        _seen.add(obj_id)
        out = [make_json_safe(v, _seen) for v in obj]
        _seen.remove(obj_id)
        return out
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def init_rag_pipeline(
    model_name: str,
    preprocess_type: str | None,
    summarize: bool,
    image_query_captioning: bool,
    image_retrieval_mode: str,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_qwen, processor_qwen = None, None
    model_llava, processor_llava = None, None

    if model_name == "Qwen/Qwen3-VL-8B-Instruct":
        model_qwen, processor_qwen = load_qwen_model(device=device)
    elif model_name == "llava-hf/llava-v1.6-mistral-7b-hf":
        model_llava, processor_llava = load_llava_model(device=device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    use_text_llm = bool(preprocess_type or summarize or image_query_captioning)
    text_llm = get_groq_llm(api_key=groq_api_key, model_name="llama-3.1-8b-instant") if use_text_llm else None

    embedding_manager_txt = EmbeddingManager(model_name=embedding_model_name)
    embedding_manager_images = (
        EmbeddingManager_Image(model_name=image_embedding_model_name)
        if image_retrieval_mode == "clip"
        else None
    )
    embedding_manager_text_clip = (
        EmbeddingManager_Text_CLIP(model_name=image_embedding_model_name)
        if image_retrieval_mode == "clip_repr"
        else None
    )

    vector_db_manager_pdf = VectorDBManager(
        collection_name="pdf_documents_db",
        directory=os.path.join(vectordb_path, "pdf_db/"),
        source_type="pdf",
    )
    if image_retrieval_mode == "page_text":
        vector_db_manager_pdf_images = VectorDBManager(
            collection_name="pdf_image_page_texts_db",
            directory=os.path.join(vectordb_path, "pdf_image_page_texts_db/"),
            source_type="pdf_image_page_texts",
        )
        retriever_multimodal_image = RetrieverMultiModal_ImagePageText(
            vector_db_text=vector_db_manager_pdf,
            vector_db_image_text=vector_db_manager_pdf_images,
            embedding_manager_text=embedding_manager_txt,
        )
    elif image_retrieval_mode == "vlm_caption":
        vector_db_manager_pdf_images = VectorDBManager(
            collection_name="pdf_image_vlm_captions",
            directory=os.path.join(vectordb_path, "pdf_image_vlm_captions_db/"),
            source_type="pdf_image_vlm_captions",
        )
        retriever_multimodal_image = RetrieverMultiModal_ImageVLMCaptions(
            vector_db_text=vector_db_manager_pdf,
            vector_db_image_captions=vector_db_manager_pdf_images,
            embedding_manager_text=embedding_manager_txt,
        )
    elif image_retrieval_mode == "clip_repr":
        vector_db_manager_pdf_text_clip = VectorDBManager(
            collection_name="pdf_text_clip_db",
            directory=os.path.join(vectordb_path, "pdf_text_clip_db/"),
            source_type="pdf_text_clip",
        )
        vector_db_manager_pdf_images = VectorDBManager(
            collection_name="pdf_image_documents_db",
            directory=os.path.join(vectordb_path, "pdf_image_db/"),
            source_type="pdf_image",
        )
        retriever_multimodal_image = RetrieverMultiModal_ImageCLIPRepresentations(
            vector_db_text=vector_db_manager_pdf,
            vector_db_text_clip=vector_db_manager_pdf_text_clip,
            vector_db_image=vector_db_manager_pdf_images,
            embedding_manager_text=embedding_manager_txt,
            embedding_manager_text_clip=embedding_manager_text_clip,
        )
    else:
        vector_db_manager_pdf_images = VectorDBManager(
            collection_name="pdf_image_documents_db",
            directory=os.path.join(vectordb_path, "pdf_image_db/"),
            source_type="pdf_image",
        )
        retriever_multimodal_image = RetrieverMultiModal_experimental(
            vector_db_text=vector_db_manager_pdf,
            vector_db_image=vector_db_manager_pdf_images,
            embedding_manager_text=embedding_manager_txt,
            embedding_manager_image=embedding_manager_images,
        )

    mm_rag = AdvancedMultimodalRAG(
        retriever_multimodal=retriever_multimodal_image,
        model_qwen=model_qwen,
        model_llava=model_llava,
        processor_llava=processor_llava,
        processor_qwen=processor_qwen,
        text_llm=text_llm,
    )

    backend = "qwen_vl" if model_name == "Qwen/Qwen3-VL-8B-Instruct" else "llava"
    return mm_rag, backend, device


def main() -> None:
    args = parse_args()
    dataset_path = ROOT / args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit is not None:
        dataset = dataset[: args.limit]

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")

    if args.num_shards > 1:
        dataset = [item for idx, item in enumerate(dataset) if idx % args.num_shards == args.shard_id]

    run_dir = Path(args.run_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.num_shards > 1:
        run_dir = run_dir / f"shard_{args.shard_id:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    mm_rag, backend, device = init_rag_pipeline(
        model_name=args.model,
        preprocess_type=args.preprocess_type,
        summarize=args.summarize,
        image_query_captioning=args.image_query_captioning,
        image_retrieval_mode=args.image_retrieval_mode,
    )

    config_out = {
        "dataset": str(dataset_path),
        "run_dir": str(run_dir),
        "model": args.model,
        "backend": backend,
        "device": device,
        "preprocess_type": args.preprocess_type,
        "summarize": args.summarize,
        "image_query_captioning": args.image_query_captioning,
        "image_retrieval_mode": args.image_retrieval_mode,
        "filtered_image_retrieval": args.filtered_image_retrieval,
        "merge_type": args.merge_type,
        "topk_text": args.topk_text,
        "topk_image": args.topk_image,
        "max_images": args.max_images,
        "match_threshold_text": args.match_threshold_text,
        "match_threshold_image": args.match_threshold_image,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    predictions_path = run_dir / "predictions.jsonl"
    failures_path = run_dir / "failed.jsonl"

    with predictions_path.open("w", encoding="utf-8") as pred_f, failures_path.open("w", encoding="utf-8") as fail_f:
        for idx, item in enumerate(dataset, start=1):
            qid = item.get("id", f"item_{idx:04d}")
            question = item.get("question", "")
            question_type = item.get("question_type") or item.get("type") or "unknown"
            gold_answer = item.get("answer") or item.get("gold_answer")
            try:
                result = mm_rag.generate_response(
                    query=question,
                    backend=backend,
                    top_k_text=args.topk_text,
                    top_k_image=args.topk_image,
                    match_threshold_text=args.match_threshold_text,
                    match_threshold_image=args.match_threshold_image,
                    max_images=args.max_images,
                    max_new_tokens=args.max_new_tokens,
                    preprocess_type=args.preprocess_type,
                    summarize=args.summarize,
                    image_query_captioning=args.image_query_captioning,
                    filtered_image_retrieval=args.filtered_image_retrieval,
                    merge_type=args.merge_type,
                    device=device,
                )
                pred_answer = result.get("answer", "")
                pred_choice = parse_choice(pred_answer)
                rag_result = dict(result)
                rag_result.pop("message_history", None)
                record = {
                    "id": qid,
                    "question_type": question_type,
                    "question": question,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "pred_choice": pred_choice,
                    "rag_result": make_json_safe(rag_result),
                    "dataset_item": make_json_safe(item),
                }
                pred_f.write(json.dumps(record) + "\n")
            except Exception as exc:
                fail_f.write(json.dumps({"id": qid, "error": str(exc)}) + "\n")


if __name__ == "__main__":
    main()
