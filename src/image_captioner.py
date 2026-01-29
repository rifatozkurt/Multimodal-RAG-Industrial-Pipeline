"""

CUDA_VISIBLE_DEVICES=2 python src/image_captioner.py --output_mode single

"""
import argparse
import json
import time
from pathlib import Path
import re

import torch
from PIL import Image
from tqdm import tqdm

from core.models_llm import load_qwen_model


IMAGE_EXTS = {".png"}


PROMPT = (
    """You are generating a retrieval caption for an image from an industrial technical manual.
Purpose: the caption + tags will be embedded for semantic similarity search to retrieve this image later.
Be compact and technical. Prefer visible labels/headers/terminology over long prose.
Do NOT invent text that is not visible. If something is unclear, omit it.

Return ONLY a valid JSON object with exactly two fields:
{
  "caption": "string",
  "tags": ["string", "..."]
}

CAPTION requirements:
- 3-4 sentences total.
- Must include the image type (e.g., table, wiring diagram, schematic, UI screenshot, plot, flowchart).
- If a table: mention the table title (if present) and the most important column headers (and optionally key row labels).
- If a diagram: mention the main components/blocks and what they are connected to or represent.
- If a plot: mention axis labels/units and what the curves/legend represent.
- If UI screenshot: mention product/software name, screen title, menu path, key buttons/status/error codes.
- Include product name/model if visible.
- Include key visible headers/labels (exact wording where possible).

TAGS requirements:
- 5-10 tags, single words or short phrases.
- Include: image type, product/model names, main entities/components, OCR-visible terminology (titles, headers, labels, codes), and likely query keywords/synonyms/abbreviations.
- For tables, include column headers as tags.
- For diagrams, include connector/terminal/signal names as tags.
- Keep tags high-signal; do not include full sentences; avoid generic filler words.

Now analyze the image and produce the JSON."""

)


def build_qwen_caption_inputs(processor_qwen, image_path: Path):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    return processor_qwen.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def _extract_caption_tags_from_raw(text: str):
    text = text.strip()
    # Try JSON-like caption field (supports multiline and escaped quotes)
    m = re.search(r'"caption"\s*:\s*"((?:\\\\.|[^"])*)"', text, flags=re.IGNORECASE | re.DOTALL)
    caption = m.group(1).strip() if m else ""
    if not caption:
        m = re.search(r'"caption"\s*:\s*"(.+)$', text, flags=re.IGNORECASE | re.DOTALL)
        caption = m.group(1).strip() if m else ""
    # Try JSON-like tags array
    tags = []
    m = re.search(r'"tags"\s*:\s*(\[[^\]]*\])', text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        try:
            tags = json.loads(m.group(1))
        except Exception:
            tags = []
    return caption, tags


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    caption, tags = _extract_caption_tags_from_raw(text)
    if caption or tags:
        return {"caption": caption, "tags": tags}
    return None


def load_existing_records(path: Path):
    if not path.exists():
        return [], set()
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            return [], set()
    except Exception:
        return [], set()
    processed = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        image_path = item.get("image_path")
        if not image_path:
            continue
        processed.update(normalize_path(image_path))
    return data, processed


def normalize_path(path_value):
    try:
        path = Path(path_value)
    except Exception:
        return {str(path_value)}
    normalized = {str(path)}
    try:
        normalized.add(str(path.resolve()))
    except Exception:
        pass
    return normalized


def is_processed(image_path: Path, processed_set: set):
    for key in normalize_path(image_path):
        if key in processed_set:
            return True
    return False


def save_records(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def iter_image_files(root: Path):
    for doc_dir in sorted(root.iterdir()):
        if not doc_dir.is_dir():
            continue
        images = sorted(p for p in doc_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        yield doc_dir, images


def caption_images(
    input_dir: Path,
    output_mode: str,
    output_name: str,
    max_new_tokens: int,
    limit: int | None,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = load_qwen_model(device=device)
    model.eval()

    global_records = []
    global_processed = set()
    folder_cache = {}
    if output_mode == "single":
        global_path = input_dir / output_name
        global_records, global_processed = load_existing_records(global_path)

    tasks = []
    for doc_dir, images in iter_image_files(input_dir):
        if not images:
            continue

        if output_mode == "per-folder":
            output_path = doc_dir / output_name
            records, processed = load_existing_records(output_path)
            folder_cache[doc_dir] = (output_path, records, processed)
        else:
            output_path = input_dir / output_name
            records, processed = global_records, global_processed

        for image_path in images:
            if is_processed(image_path, processed):
                continue
            tasks.append((doc_dir, image_path))

    if limit is not None:
        tasks = tasks[:limit]

    for doc_dir, image_path in tqdm(tasks, desc="Captioning images"):
        if output_mode == "per-folder":
            output_path, records, processed = folder_cache[doc_dir]
        else:
            output_path, records, processed = input_dir / output_name, global_records, global_processed

        inputs = build_qwen_caption_inputs(processor, image_path)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response_list = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        response = response_list[0].strip() if response_list else ""
        parsed = extract_json(response)

        record = {
            "image_path": str(image_path),
            "document_id": doc_dir.name,
            "filename": image_path.name,
            "model_json": parsed,
            "raw_model_output": response,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        records.append(record)
        processed.update(normalize_path(image_path))
        save_records(output_path, records)

    if output_mode == "single":
        save_records(input_dir / output_name, global_records)


def main():
    parser = argparse.ArgumentParser(
        description="Generate retrieval captions for extracted images using Qwen3-VL."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("documents/pdfs/extracted_images"),
        help="Root directory containing per-document image folders.",
    )
    parser.add_argument(
        "--output_mode",
        choices=["per-folder", "single"],
        default="single",
        help="Store captions in each image folder or in a single JSON file.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="vlm_captions.json",
        help="Output JSON filename (per-folder) or global JSON filename (single).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens for caption generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images to process.",
    )
    args = parser.parse_args()

    caption_images(
        input_dir=args.input_dir,
        output_mode=args.output_mode,
        output_name=args.output_name,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
