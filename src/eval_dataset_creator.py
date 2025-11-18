
"""
Simple PDF QA annotation tool for multimodal RAG datasets.

Features
--------
- Input a directory path containing PDFs.
- Dropdown with all detected PDFs.
- Page slider + preview:
    * Renders PDF page as an image (via PyMuPDF).
    * Shows extracted page text.
- Question authoring:
    * Question text
    * Ground-truth answer
    * Question type:
        - free_form
        - multiple_choice
        - truth_table
        - fill_in_the_blanks
    * Auto-generated JSON templates for structured types.
- Saves annotations to a JSON file as a list of entries.

Dependencies
------------
pip install gradio pymupdf pillow

Run
---
python pdf_qa_annotator.py
"""

import os
import json
import time
from typing import Dict, List, Any, Tuple, Optional

import gradio as gr
import fitz  # PyMuPDF
from PIL import Image


# ----------------------------
# Utility functions
# ----------------------------

def list_pdfs_in_directory(pdf_dir: str) -> Dict[str, str]:
    """Return a mapping {pdf_name: full_path} for all PDFs in directory."""
    pdf_paths: Dict[str, str] = {}

    if not pdf_dir or not os.path.isdir(pdf_dir):
        return pdf_paths

    for fname in sorted(os.listdir(pdf_dir)):
        if fname.lower().endswith(".pdf"):
            full = os.path.join(pdf_dir, fname)
            pdf_paths[fname] = full

    return pdf_paths


def render_pdf_page(pdf_path: str, page_number: int) -> Tuple[Optional[Image.Image], str, int, str]:
    """
    Render a single PDF page to an image and extract text.

    Parameters
    ----------
    pdf_path : str
        Full path to the PDF file.
    page_number : int
        1-based page number.

    Returns
    -------
    (image, text, total_pages, info_message)
    """
    if not os.path.exists(pdf_path):
        return None, "", 0, f"PDF file not found: {pdf_path}"

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return None, "", 0, f"Error opening PDF: {e}"

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        return None, "", 0, "PDF has no pages."

    # Clamp page_number
    if page_number < 1:
        page_number = 1
    if page_number > total_pages:
        page_number = total_pages

    try:
        page = doc[page_number - 1]
        # Render the page as an image (increase zoom for readability)
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        mode = "RGB"
        if pix.n >= 4:
            mode = "RGBA"

        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

        text = page.get_text("text") or ""
        info = f"Loaded page {page_number} / {total_pages} from '{os.path.basename(pdf_path)}'."

        doc.close()
        return image, text, total_pages, info
    except Exception as e:
        doc.close()
        return None, "", 0, f"Error rendering page: {e}"


def load_existing_dataset(json_path: str) -> List[Dict[str, Any]]:
    """Load existing dataset JSON (list of dicts). Return [] if file doesn't exist or is invalid."""
    if not json_path or not os.path.exists(json_path):
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            # Not a list; ignore
            return []
    except Exception:
        return []


def save_dataset(json_path: str, entries: List[Dict[str, Any]]) -> None:
    """Save list of entries to JSON file."""
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def generate_entry_id(pdf_name: str, page: int, existing_len: int) -> str:
    """Generate a simple unique-ish ID based on pdf, page and dataset size."""
    base = os.path.splitext(pdf_name)[0]
    return f"{base}_p{page}_q{existing_len+1:04d}"


# ----------------------------
# Template generation
# ----------------------------

def structured_template_for_type(qtype: str) -> str:
    """
    Return a JSON string template for the given question type.
    This is meant to be edited by the annotator if needed.
    """
    if qtype == "multiple_choice":
        template = {
            "type": "multiple_choice",
            # HELM / Harness-like structure (simplified)
            "choices": ["A", "B", "C", "D"],
            "correct_index": 0
        }
    elif qtype == "truth_table":
        template = {
            "type": "truth_table",
            "columns": ["A", "B", "OUT"],
            "rows": [
                {"A": 0, "B": 0, "OUT": 0},
                {"A": 0, "B": 1, "OUT": 1},
                {"A": 1, "B": 0, "OUT": 1},
                {"A": 1, "B": 1, "OUT": 1}
            ]
        }
    elif qtype == "fill_in_the_blanks":
        template = {
            "type": "fill_in_the_blanks",
            "template": "The ___ is connected to pin ___.",
            "answers": ["resistor", "PA5"]
        }
    else:
        # free_form or unknown
        template = {
            "type": "free_form",
            "notes": "No additional structured metadata required."
        }

    return json.dumps(template, indent=2, ensure_ascii=False)


# ----------------------------
# Gradio callback functions
# ----------------------------

def on_scan_pdfs(pdf_dir: str) -> Tuple[gr.Dropdown, str, str, Dict[str, str]]:
    """
    Scan directory for PDFs and populate dropdown.
    """
    pdf_dir = pdf_dir.strip()
    if not pdf_dir or not os.path.isdir(pdf_dir):
        return (
            gr.Dropdown(choices=[], value=None),
            f"❌ Directory not found: '{pdf_dir}'",
            pdf_dir,
            {}
        )

    pdf_map = list_pdfs_in_directory(pdf_dir)
    if not pdf_map:
        return (
            gr.Dropdown(choices=[], value=None),
            f"⚠️ No PDFs found in: '{pdf_dir}'",
            pdf_dir,
            {}
        )

    choices = list(pdf_map.keys())
    info = f"✅ Found {len(choices)} PDF(s) in '{pdf_dir}'. Select one from the dropdown."
    return gr.Dropdown(choices=choices, value=choices[0]), info, pdf_dir, pdf_map


def on_select_pdf(pdf_dir: str,
                  pdf_map: Dict[str, str],
                  pdf_name: str) -> Tuple[Optional[Image.Image], str, gr.Slider, str]:
    """
    When a PDF is selected (or reselected), load page 1 and update page slider.
    """
    if not pdf_name or not pdf_map or pdf_name not in pdf_map:
        return None, "", gr.Slider(minimum=1, maximum=1, value=1, step=1), "⚠️ No PDF selected."

    pdf_path = pdf_map[pdf_name]
    image, text, total_pages, info = render_pdf_page(pdf_path, page_number=1)

    if total_pages <= 0:
        slider = gr.Slider(minimum=1, maximum=1, value=1, step=1)
    else:
        slider = gr.Slider(minimum=1, maximum=total_pages, value=1, step=1)

    return image, text, slider, info


def on_change_page(pdf_dir: str,
                   pdf_map: Dict[str, str],
                   pdf_name: str,
                   page_number: int) -> Tuple[Optional[Image.Image], str, str]:
    """
    When the page slider changes, re-render the page.
    """
    if not pdf_name or not pdf_map or pdf_name not in pdf_map:
        return None, "", "⚠️ No PDF selected."

    pdf_path = pdf_map[pdf_name]
    image, text, total_pages, info = render_pdf_page(pdf_path, page_number=int(page_number))
    return image, text, info


def on_change_qtype(qtype: str) -> str:
    """
    When question type changes, populate structured metadata template.
    """
    return structured_template_for_type(qtype)


def on_add_question(pdf_dir: str,
                    pdf_map: Dict[str, str],
                    pdf_name: str,
                    page_number: int,
                    question: str,
                    answer: str,
                    qtype: str,
                    structured_meta: str,
                    json_path: str) -> str:
    """
    Add a new question to the JSON dataset.
    """
    pdf_dir = (pdf_dir or "").strip()
    json_path = (json_path or "").strip()

    if not json_path:
        json_path = "dataset.json"

    if not pdf_name or not pdf_map or pdf_name not in pdf_map:
        return "❌ Cannot add question: no valid PDF selected."

    if not question.strip():
        return "❌ Question is empty."

    if not answer.strip():
        return "❌ Answer is empty."

    pdf_path = pdf_map[pdf_name]
    page_number = int(page_number)
    if page_number < 1:
        page_number = 1

    # Load existing entries
    entries = load_existing_dataset(json_path)

    # Parse structured metadata if possible
    structured_obj: Any = None
    structured_meta = (structured_meta or "").strip()
    if structured_meta:
        try:
            structured_obj = json.loads(structured_meta)
        except Exception:
            # If parsing fails, just keep as raw string
            structured_obj = {"raw": structured_meta}

    entry_id = generate_entry_id(pdf_name, page_number, len(entries))

    entry = {
        "id": entry_id,
        "pdf_name": pdf_name,
        "pdf_path": pdf_path,
        "page": page_number,
        "question": question.strip(),
        "answer": answer.strip(),
        "question_type": qtype,
        "structured_metadata": structured_obj,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    entries.append(entry)
    try:
        save_dataset(json_path, entries)
    except Exception as e:
        return f"❌ Error saving to JSON: {e}"

    return (f"✅ Added question #{len(entries)}.\n\n"
            f"- ID: `{entry_id}`\n"
            f"- PDF: `{pdf_name}` page {page_number}\n"
            f"- Saved to: `{json_path}`")


# ----------------------------
# Gradio UI
# ----------------------------

def build_interface():
    with gr.Blocks(title="PDF QA Annotation Tool") as demo:
        gr.Markdown(
            """
            # 📄 PDF QA Annotation Tool

            1. Enter a directory containing PDF files.
            2. Select a PDF and browse its pages.
            3. Write questions and ground-truth answers.
            4. Choose question type (free form / MCQ / truth table / fill-in-the-blanks).
            5. Optionally edit the auto-generated structured JSON template.
            6. Click **Add Question** to append to the dataset JSON.

            The JSON file will be a list of entries, one per question.
            """
        )

        # Shared state
        pdf_dir_state = gr.State("")
        pdf_map_state = gr.State({})

        with gr.Row():
            pdf_dir_input = gr.Textbox(
                value = r"C:\Users\rozku\Desktop\Research_Internship\Project\siemens_test\pdfs",
                label="PDF Directory",
                placeholder="Enter path to directory with PDFs, e.g. ./pdfs",
                lines=1,
            )
            scan_button = gr.Button("🔍 Scan PDFs", variant="primary")

        pdf_info = gr.Markdown("Directory not scanned yet.")
        with gr.Row():
            pdf_dropdown = gr.Dropdown(
                label="Select PDF",
                choices=[],
                interactive=True,
            )

        with gr.Row():
            page_slider = gr.Slider(
                label="Page",
                minimum=1,
                maximum=1,
                step=1,
                value=1,
                interactive=True,
            )

        with gr.Row():
            page_image = gr.Image(
                label="Page Preview",
                type="pil",
                interactive=False,
            )
            page_text = gr.Textbox(
                label="Extracted Page Text",
                lines=25,
                interactive=False,
            )

        page_info = gr.Markdown("No page loaded yet.")

        gr.Markdown("---")
        gr.Markdown("## ✍️ Question Annotation")

        with gr.Row():
            question_box = gr.Textbox(
                label="Question",
                lines=3,
                placeholder="Write your question here...",
            )

        with gr.Row():
            answer_box = gr.Textbox(
                label="Ground-Truth Answer",
                lines=4,
                placeholder="Write the correct answer here...",
            )

        qtype_dropdown = gr.Dropdown(
            label="Question Type",
            choices=["free_form", "multiple_choice", "truth_table", "fill_in_the_blanks"],
            value="free_form",
            interactive=True,
        )

        structured_meta_box = gr.Textbox(
            label="Structured Metadata (JSON, optional; auto-filled based on type)",
            lines=16,
            value=structured_template_for_type("free_form"),
        )

        with gr.Row():
            json_path_box = gr.Textbox(
                label="Output JSON File",
                value="dataset.json",
                lines=1,
                interactive=True,
            )
            add_button = gr.Button("➕ Add Question to Dataset", variant="primary")

        status_box = gr.Markdown("No questions added yet.")

        # ----------------
        # Wire callbacks
        # ----------------

        # Scan PDFs
        scan_button.click(
            fn=on_scan_pdfs,
            inputs=[pdf_dir_input],
            outputs=[pdf_dropdown, pdf_info, pdf_dir_state, pdf_map_state],
        )

        # When PDF is selected, load first page
        pdf_dropdown.change(
            fn=on_select_pdf,
            inputs=[pdf_dir_state, pdf_map_state, pdf_dropdown],
            outputs=[page_image, page_text, page_slider, page_info],
        )

        # When page slider changes, update page preview
        page_slider.change(
            fn=on_change_page,
            inputs=[pdf_dir_state, pdf_map_state, pdf_dropdown, page_slider],
            outputs=[page_image, page_text, page_info],
        )

        # When question type changes, update template
        qtype_dropdown.change(
            fn=on_change_qtype,
            inputs=[qtype_dropdown],
            outputs=[structured_meta_box],
        )

        # Add question
        add_button.click(
            fn=on_add_question,
            inputs=[
                pdf_dir_state,
                pdf_map_state,
                pdf_dropdown,
                page_slider,
                question_box,
                answer_box,
                qtype_dropdown,
                structured_meta_box,
                json_path_box,
            ],
            outputs=[status_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    # You can set share=True if you want a public link
    demo.launch()
