import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "eval" / "runs"


def list_eval_runs() -> dict[str, Path]:
    run_map = {}
    if not RUNS_DIR.exists():
        return run_map
    for pred_path in RUNS_DIR.rglob("predictions.jsonl"):
        run_dir = pred_path.parent
        label = str(run_dir.relative_to(RUNS_DIR))
        run_map[label] = run_dir
    return dict(sorted(run_map.items(), key=lambda x: x[0]))


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_run_data(run_label: str):
    run_map = list_eval_runs()
    run_dir = run_map.get(run_label)
    if run_dir is None:
        return [], {}, {}

    predictions_path = run_dir / "predictions.jsonl"
    per_item_path = run_dir / "per_item.jsonl"
    predictions = load_jsonl(predictions_path) if predictions_path.exists() else []
    pred_map = {p.get("id"): p for p in predictions if p.get("id")}
    metrics_map = {}
    if per_item_path.exists():
        metrics = load_jsonl(per_item_path)
        metrics_map = {m.get("id"): m for m in metrics if m.get("id")}
    return predictions, pred_map, metrics_map


def format_retrieved_texts(rag_result: dict) -> str:
    docs = rag_result.get("retrieved_text_docs", []) or []
    if not docs:
        return "No retrieved text docs."
    parts = []
    for idx, doc in enumerate(docs, start=1):
        text = doc.get("document") if isinstance(doc, dict) else ""
        score = doc.get("score") if isinstance(doc, dict) else None
        meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        source = meta.get("source") or meta.get("file_path") or ""
        header = f"[Text {idx}] score={score} source={source}"
        parts.append(header)
        if text:
            parts.append(text.strip())
        parts.append("-" * 40)
    return "\n".join(parts)


def build_gallery_items(image_docs: list[dict], key: str) -> list[tuple[str, str]]:
    items = []
    for idx, doc in enumerate(image_docs, start=1):
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        path = metadata.get("image_path") or metadata.get("page_image_path") or metadata.get("image_rel_path")
        score = doc.get("score") if isinstance(doc, dict) else None
        if not path:
            continue
        caption = f"{key} {idx} | score={score} | {Path(path).name}"
        items.append((path, caption))
    return items


def build_ground_truth_gallery(dataset_item: dict) -> tuple[list[tuple[str, str]], str]:
    entries = dataset_item.get("ground_truth_images") or []
    gallery = []
    filenames = []
    for idx, entry in enumerate(entries, start=1):
        if isinstance(entry, str):
            filenames.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        path = entry.get("image_path")
        name = entry.get("image_filename") or (Path(path).name if path else "")
        if name:
            filenames.append(name)
        if path:
            gallery.append((path, f"GT {idx} | {name}"))
    return gallery, ", ".join(filenames)


def on_select_run(run_label: str):
    predictions, pred_map, metrics_map = load_run_data(run_label)
    question_ids = sorted([p.get("id") for p in predictions if p.get("id")])
    if not question_ids:
        return gr.Dropdown(choices=[], value=None), "", "", [], [], "", ""
    first_id = question_ids[0]
    return (
        gr.Dropdown(choices=question_ids, value=first_id),
        json.dumps(pred_map.get(first_id, {}).get("question", ""), indent=2),
        json.dumps(pred_map.get(first_id, {}).get("pred_answer", ""), indent=2),
        [],
        [],
        "",
        json.dumps(metrics_map.get(first_id, {}), indent=2),
    )


def on_select_question(run_label: str, question_id: str):
    _, pred_map, metrics_map = load_run_data(run_label)
    pred = pred_map.get(question_id, {})
    rag_result = pred.get("rag_result", {}) or {}
    dataset_item = pred.get("dataset_item", {}) or {}

    question_text = pred.get("question", "")
    answer_text = pred.get("pred_answer", "")
    retrieved_texts = format_retrieved_texts(rag_result)
    retrieved_images = build_gallery_items(rag_result.get("retrieved_image_docs", []) or [], "Retrieved")
    gt_gallery, gt_names = build_ground_truth_gallery(dataset_item)
    metrics = metrics_map.get(question_id, {})

    answer_block = f"Predicted Answer:\n{answer_text}\n\nGold Answer:\n{pred.get('gold_answer', '')}"

    return (
        question_text,
        answer_block,
        retrieved_texts,
        retrieved_images,
        gt_gallery,
        gt_names,
        json.dumps(metrics, indent=2),
    )


def build_interface():
    run_map = list_eval_runs()
    run_choices = list(run_map.keys())

    with gr.Blocks(title="Evaluation Viewer") as demo:
        gr.Markdown("# Evaluation Run Viewer")

        with gr.Row():
            run_dropdown = gr.Dropdown(
                label="Evaluation Run",
                choices=run_choices,
                value=run_choices[0] if run_choices else None,
            )

        with gr.Row():
            question_dropdown = gr.Dropdown(
                label="Question ID",
                choices=[],
                value=None,
            )

        with gr.Row():
            question_box = gr.Textbox(label="Question", lines=6)
            answer_box = gr.Textbox(label="Answer", lines=6)

        with gr.Row():
            retrieved_texts_box = gr.Textbox(label="Retrieved Texts", lines=18)

        with gr.Row():
            retrieved_gallery = gr.Gallery(label="Retrieved Images", columns=3, rows=2, height=240)
            gt_gallery = gr.Gallery(label="Ground-Truth Images", columns=3, rows=2, height=240)

        gt_names_box = gr.Textbox(label="Ground-Truth Image Filenames", lines=2)

        metrics_box = gr.Textbox(label="Per-Question Metrics (JSON)", lines=12)

        run_dropdown.change(
            fn=on_select_run,
            inputs=[run_dropdown],
            outputs=[
                question_dropdown,
                question_box,
                answer_box,
                retrieved_gallery,
                gt_gallery,
                gt_names_box,
                metrics_box,
            ],
        )

        question_dropdown.change(
            fn=on_select_question,
            inputs=[run_dropdown, question_dropdown],
            outputs=[
                question_box,
                answer_box,
                retrieved_texts_box,
                retrieved_gallery,
                gt_gallery,
                gt_names_box,
                metrics_box,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()
