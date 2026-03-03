# Multimodal RAG for Industrial Manuals

This repository implements a multimodal RAG pipeline for technical PDFs (text + images), with:
- PDF/layout extraction and vector indexing
- Multiple image-retrieval strategies (CLIP, page text, VLM captions, CLIP-representation merge)
- Qwen3-VL or LLaVA generation backends
- Evaluation tooling for QA and retrieval metrics

## What Is In This Repo

### Main source files
- `src/add_documents_extractor.py`: Main ingestion/indexing pipeline (text chunks + layout image crops + optional VLM captions + CLIP text representations).
- `src/multimodal_rag.py`: Script entrypoint for one-shot multimodal RAG querying.
- `src/chat_interface.py`: Gradio chat UI with model preloading and optional user-uploaded image prioritization.
- `src/image_captioner.py`: Generates retrieval-oriented captions/tags for extracted images (`vlm_captions.json`).
- `src/eval_dataset_creator.py`: Gradio annotation app to build QA datasets from PDFs.

### Core modules (`src/core`)
- `config.py`: Global paths, embedding models, eval defaults.
- `data_loaders.py`: PDF extraction loaders (`PdfExtractionLoader`, `PdfImagesLoader`) + chunking.
- `embedders.py`: Text/image embedding managers (`SentenceTransformer`, CLIP).
- `vectordb.py`: Chroma persistent collection wrapper.
- `retrievers.py`: Text and multimodal retrievers, including experimental variants.
- `rag_pipelines.py`: `SimpleRAG`, `AdvancedRAG`, and `AdvancedMultimodalRAG` orchestration.
- `models_llm.py`: Prompt templates and model-loading helpers (Groq, Qwen3-VL, LLaVA).

### Evaluation (`eval/metrics`)
- `run_eval.py`: End-to-end QA eval (predict + log per run).
- `score_eval.py`: Scores predictions (MCQ/FIB/open, ROUGE, image retrieval, optional NLI).
- `retrieval_eval.py`: Retrieval-only benchmarking across retriever modes.
- `summarize_runs.py`: Aggregates `metrics.json` files into `summary.json` and `summary.csv`.
- `view_eval.py`: Gradio viewer for run inspection.
- `experiments.py`: Multi-GPU experiment launcher.

### Data/artifacts folders
- `documents/`: Input docs and vector DBs.
  - Expected ingestion input is `documents/pdfs/*.pdf`.
  - Chroma stores under `documents/vectorDB/*`.
- `eval/datasets/`: QA dataset JSONs.
- `eval/runs/` and `eval/runs_retrieval/`: Saved eval outputs.

## Setup

### 1) Python environment
- Recommended: Python 3.11+ (project metadata uses `>=3.11`).
- Install dependencies with one of:

```bash
pip install -r requirements.txt
```

or

```bash
uv sync
```

### 2) Environment variables
Create `.env` with keys you need:
- `GROQ_API_KEY` (query expansion/summarization + text LLM support)
- `OPENAI_API_KEY` / `HUGGINGFACE_API_KEY` only if your setup requires them

Do not commit real API keys.

## End-to-End Workflow

### 1) Prepare documents
- Put PDFs in: `documents/pdfs/`
- If using layout-based extraction, provide layout files under: `documents/outputs/<pdf_stem>/layout_results.json`

### 2) (Optional) Generate image captions for caption-based retrieval
```bash
python src/image_captioner.py --output_mode single
```
This writes `documents/pdfs/extracted_images/vlm_captions.json`.

### 3) Build/update vector databases
```bash
python src/add_documents_extractor.py
```
This script populates:
- `pdf_db` (text chunks)
- `pdf_image_db` (image regions)
- `pdf_image_page_texts_db` (text near image regions)
- `pdf_text_clip_db` (CLIP text embeddings)
- `pdf_image_vlm_captions_db` (caption embeddings)

### 4) Run a query
CLI:
```bash
python src/multimodal_rag.py
```

UI:
```bash
python src/chat_interface.py
```
Then load a backend model in the UI before sending prompts.

## Retrieval Modes

Supported by eval/runtime code:
- `clip`: CLIP text->image retrieval
- `page_text`: retrieve image context by text-only embeddings over page text near images
- `vlm_caption`: retrieve images via generated caption embeddings
- `clip_repr`: retrieve via merged CLIP text-representation strategies:
  - `normalized_mean`
  - `similarity_weighted`
  - `query_interpolation`
  - `pca`

## Evaluation

### Run full QA eval
```bash
python eval/metrics/run_eval.py \
  --dataset eval/datasets/dataset_mcq_fib.json \
  --run_dir eval/runs/vlm_caption \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --image_retrieval_mode vlm_caption \
  --filtered_image_retrieval \
  --topk_text 3 --topk_image 3 --max_images 3
```

### Score an existing predictions file
```bash
python eval/metrics/score_eval.py \
  --dataset eval/datasets/dataset_mcq_fib.json \
  --predictions eval/runs/<run_name>/<timestamp>/predictions.jsonl
```

### Aggregate all runs into summaries
```bash
python eval/metrics/summarize_runs.py --run_root eval/runs
python eval/metrics/summarize_runs.py --run_root eval/runs_retrieval
```

### Retrieval-only benchmark
```bash
python eval/metrics/retrieval_eval.py --dataset eval/datasets/dataset_mcq_fib.json
```

### Inspect runs in a UI
```bash
python eval/metrics/view_eval.py
```

## Notebook/Utility Files
- `src/RAG_pipeline.ipynb` and `src/multimodalRAG.ipynb`: exploratory notebook versions of the pipeline.
- `test.ipynb`: environment/GPU sanity checks.
- `my_env_now.yml`: large conda environment snapshot.

## Notes
- Repository contains persisted Chroma artifacts and historical eval outputs for reproducibility.
- `pyproject.toml` dependency list is minimal; `requirements.txt` is the practical environment definition for this project.
