import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import evaluate
import torch
from transformers import pipeline

from utils_text import (
    image_id_candidates,
    normalize_for_exact,
    parse_choice,
    parse_number_list,
    token_f1,
)

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from core.config import nli_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score predictions with ROUGE and retrieval metrics.")
    parser.add_argument("--dataset", default="eval/datasets/dataset_mcq_fib.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--k_list", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--nli_model", default=nli_model_name)
    parser.add_argument("--nli_max_text_docs", type=int, default=6)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_for_rouge(text: str) -> str:
    return normalize_for_exact(text)


def extract_gold_image_ids(item: dict) -> set[str]:
    gold = set()
    raw = item.get("ground_truth_images") or item.get("gold_image_paths") or item.get("gold_images")
    if raw is None:
        return gold
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str):
                gold.update(image_id_candidates(entry))
            elif isinstance(entry, dict):
                path = entry.get("image_path") or entry.get("image_name") or entry.get("image_filename")
                if path:
                    gold.update(image_id_candidates(path))
    elif isinstance(raw, str):
        gold.update(image_id_candidates(raw))
    return gold


def extract_retrieved_image_ids(pred: dict) -> list[set[str]]:
    ids = []
    result = pred.get("rag_result", {})
    retrieved = result.get("retrieved_image_docs", []) or []
    for doc in retrieved:
        metadata = doc.get("metadata", {})
        path = metadata.get("image_path") or metadata.get("page_image_path") or metadata.get("image_rel_path")
        ids.append(image_id_candidates(path))
    return ids


def compute_image_retrieval_metrics(retrieved_ids: list[set[str]], gold_ids: set[str], k_list: list[int]) -> dict:
    metrics = {}
    hit_rank = None
    for idx, cand_set in enumerate(retrieved_ids, start=1):
        if cand_set & gold_ids:
            hit_rank = idx
            break
    for k in k_list:
        metrics[f"image_hit_at_{k}"] = 1 if hit_rank is not None and hit_rank <= k else 0
    metrics["image_mrr"] = 1 / hit_rank if hit_rank is not None else 0.0
    return metrics


def evaluate_fill(pred_answer: str, gold_answer: str, item: dict) -> dict:
    pred_norm = normalize_for_exact(pred_answer)
    gold_norm = normalize_for_exact(gold_answer)
    exact = 1 if pred_norm == gold_norm else 0
    f1 = token_f1(pred_answer, gold_answer)

    numeric_ok = None
    answer_type = item.get("answer_type")
    tolerance = item.get("tolerance") or {}
    if answer_type == "number":
        pred_nums = parse_number_list(pred_answer)
        gold_nums = parse_number_list(gold_answer)
        if pred_nums and gold_nums:
            pred_val = pred_nums[0]
            gold_val = gold_nums[0]
            abs_tol = tolerance.get("abs")
            rel_tol = tolerance.get("rel")
            if abs_tol is not None:
                numeric_ok = abs(pred_val - gold_val) <= float(abs_tol)
            elif rel_tol is not None:
                numeric_ok = abs(pred_val - gold_val) / max(abs(gold_val), 1e-9) <= float(rel_tol)
    return {
        "fill_exact_match": exact,
        "fill_token_f1": f1,
        "fill_numeric_ok": numeric_ok,
    }


def evaluate_mcq(pred_answer: str, gold_choice: str, choices: list[str] | None) -> dict:
    pred_choice = parse_choice(pred_answer, choices)
    gold_choice = (gold_choice or "").strip().upper()
    is_correct = 1 if pred_choice == gold_choice and pred_choice else 0
    return {
        "mcq_pred_choice": pred_choice,
        "mcq_is_correct": is_correct,
        "mcq_parse_fail": 1 if pred_choice is None else 0,
    }


def evaluate_open_ended(
    rouge_metric,
    pred_answer: str,
    gold_answer: str,
    reference_answers: list[str] | None,
) -> dict:
    references = reference_answers or [gold_answer]
    best = {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}
    pred_norm = normalize_for_rouge(pred_answer)
    for ref in references:
        ref_norm = normalize_for_rouge(ref)
        scores = rouge_metric.compute(
            predictions=[pred_norm],
            references=[ref_norm],
            use_stemmer=True,
        )
        candidate = {
            "rouge1_f1": float(scores.get("rouge1", 0.0)),
            "rouge2_f1": float(scores.get("rouge2", 0.0)),
            "rougeL_f1": float(scores.get("rougeL", 0.0)),
        }
        if candidate["rougeL_f1"] > best["rougeL_f1"]:
            best = candidate
    return best


def build_nli_pipeline(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        return_all_scores=True,
        device=device,
    )


def entailment_score(nli_pipe, premise: str, hypothesis: str) -> float | None:
    if not premise or not hypothesis:
        return None
    outputs = nli_pipe(
        {"text": premise, "text_pair": hypothesis},
        truncation=True,
        max_length=512,
    )
    if not outputs:
        return None
    # Normalize pipeline outputs across transformers versions.
    if isinstance(outputs, dict):
        scores = [outputs] if "label" in outputs else outputs.get("scores") or []
    elif isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
        scores = outputs[0]
    else:
        scores = outputs
    if not scores:
        return None
    label2id = getattr(nli_pipe.model.config, "label2id", {}) or {}
    id2label = {v: k for k, v in label2id.items()}
    entail_ids = []
    for idx, label in id2label.items():
        if "entail" in str(label).lower():
            entail_ids.append(idx)
    if entail_ids:
        for entry in scores:
            label = str(entry.get("label", "")).lower()
            if "entail" in label:
                return float(entry.get("score", 0.0))
    if len(scores) == 3:
        for entry in scores:
            if str(entry.get("label")) == "LABEL_2":
                return float(entry.get("score", 0.0))
    best = max(scores, key=lambda x: x.get("score", 0.0))
    return float(best.get("score", 0.0))


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    predictions_path = Path(args.predictions)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    predictions = load_jsonl(predictions_path)
    pred_map = {p.get("id"): p for p in predictions}

    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    per_item_path = output_dir / "per_item.jsonl"
    metrics_path = output_dir / "metrics.json"
    summary_txt_path = output_dir / "metrics.txt"

    summary = defaultdict(list)
    totals = defaultdict(int)
    missing_preds = 0

    rouge_metric = evaluate.load("rouge")
    nli_pipe = build_nli_pipeline(args.nli_model) if args.nli_model else None

    with per_item_path.open("w", encoding="utf-8") as per_f:
        for item in dataset:
            qid = item.get("id")
            pred = pred_map.get(qid)
            if pred is None:
                missing_preds += 1
                continue

            question_type = (item.get("question_type") or item.get("type") or "unknown").lower()
            pred_answer = pred.get("pred_answer", "") or ""
            gold_answer = item.get("answer") or item.get("gold_answer") or ""

            metrics = {
                "id": qid,
                "question_type": question_type,
                "pred_answer": pred_answer,
                "gold_answer": gold_answer,
            }

            if "multiple_choice" in question_type or question_type == "mcq":
                choices = None
                if isinstance(item.get("structured_metadata"), dict):
                    choices = item["structured_metadata"].get("choices")
                mcq_scores = evaluate_mcq(pred_answer, gold_answer, choices)
                metrics.update(mcq_scores)
                summary["mcq_accuracy"].append(mcq_scores["mcq_is_correct"])
                summary["mcq_parse_fail"].append(mcq_scores["mcq_parse_fail"])
                totals["mcq_count"] += 1

            if "fill" in question_type:
                fill_scores = evaluate_fill(pred_answer, gold_answer, item)
                metrics.update(fill_scores)
                summary["fill_exact_match"].append(fill_scores["fill_exact_match"])
                summary["fill_token_f1"].append(fill_scores["fill_token_f1"])
                if fill_scores["fill_numeric_ok"] is not None:
                    summary["fill_numeric_ok"].append(1 if fill_scores["fill_numeric_ok"] else 0)
                totals["fill_count"] += 1

            if "open" in question_type or "free_form" in question_type:
                reference_answers = item.get("reference_answers")
                rouge_scores = evaluate_open_ended(rouge_metric, pred_answer, gold_answer, reference_answers)
                metrics.update(rouge_scores)
                summary["rouge1_f1"].append(rouge_scores["rouge1_f1"])
                summary["rouge2_f1"].append(rouge_scores["rouge2_f1"])
                summary["rougeL_f1"].append(rouge_scores["rougeL_f1"])
                totals["open_count"] += 1

            gold_image_ids = extract_gold_image_ids(item)
            retrieved_image_ids = extract_retrieved_image_ids(pred)
            if gold_image_ids:
                retrieval_scores = compute_image_retrieval_metrics(retrieved_image_ids, gold_image_ids, args.k_list)
                metrics.update(retrieval_scores)
                for k in args.k_list:
                    summary[f"image_hit_at_{k}"].append(retrieval_scores[f"image_hit_at_{k}"])
                summary["image_mrr"].append(retrieval_scores["image_mrr"])
                totals["image_retrieval_count"] += 1

            if nli_pipe is not None:
                retrieved_text_docs = pred.get("rag_result", {}).get("retrieved_text_docs", []) or []
                doc_texts = []
                for doc in retrieved_text_docs[: args.nli_max_text_docs]:
                    doc_text = doc.get("document") if isinstance(doc, dict) else None
                    if doc_text:
                        doc_texts.append(doc_text)
                if doc_texts:
                    answer_scores = []
                    query_scores = []
                    for doc_text in doc_texts:
                        if "open" in question_type:
                            score_answer = entailment_score(nli_pipe, doc_text, pred_answer)
                            if score_answer is not None:
                                answer_scores.append(score_answer)
                        score_query = entailment_score(nli_pipe, doc_text, item.get("question", ""))
                        if score_query is not None:
                            query_scores.append(score_query)
                    if answer_scores:
                        metrics["nli_answer_max"] = max(answer_scores)
                        metrics["nli_answer_mean"] = sum(answer_scores) / len(answer_scores)
                        metrics["nli_answer_scores"] = answer_scores
                        summary["nli_answer_max"].append(metrics["nli_answer_max"])
                        summary["nli_answer_mean"].append(metrics["nli_answer_mean"])
                    if query_scores:
                        metrics["nli_query_max"] = max(query_scores)
                        metrics["nli_query_mean"] = sum(query_scores) / len(query_scores)
                        metrics["nli_query_scores"] = query_scores
                        summary["nli_query_max"].append(metrics["nli_query_max"])
                        summary["nli_query_mean"].append(metrics["nli_query_mean"])
                    totals["nli_count"] += 1

            metrics["gold_image_ids"] = sorted(gold_image_ids)
            metrics["retrieved_image_ids"] = [sorted(list(ids)) for ids in retrieved_image_ids]

            per_f.write(json.dumps(metrics) + "\n")

    def mean(values: list) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    summary_out = {
        "dataset": str(dataset_path),
        "predictions": str(predictions_path),
        "missing_predictions": missing_preds,
        "counts": dict(totals),
        "metrics": {k: mean(v) for k, v in summary.items()},
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {summary_out['dataset']}\n")
        f.write(f"Predictions: {summary_out['predictions']}\n")
        f.write(f"Missing predictions: {summary_out['missing_predictions']}\n")
        f.write("\nCounts:\n")
        for key, value in sorted(summary_out["counts"].items()):
            f.write(f"- {key}: {value}\n")
        f.write("\nMetrics:\n")
        for key, value in sorted(summary_out["metrics"].items()):
            f.write(f"- {key}: {value}\n")


if __name__ == "__main__":
    main()
