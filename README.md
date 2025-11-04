# Multimodal RAG Pipeline (industrial prototype)

Lightweight retrieval-augmented-generation (RAG) pipeline for multimodal documents (text + images).  
Intended for industrial use-cases such as retrieving design documents, manuals, and technical drawings with combined vision and text models. Work in progress.

## Highlights
- Extracts text and images from PDFs and other text files.
- Chunks, embeds, and stores vectors in a persistent Chroma DB.
- Retriever + LLM RAG wrapper supporting query expansion / chain-of-thought preprocessing.
- Prototype code lives in a separate scratchpad; the production pipeline is in the main pipeline file.

## Key files
- Pipeline (main): [src/RAG_pipeline.ipynb](src/RAG_pipeline.ipynb)  
  - Important components implemented in the pipeline:
    - [`PdfImagesLoader`](src/RAG_pipeline.ipynb) — embedded-image & page rendering loader
    - [`EmbeddingManager`](src/RAG_pipeline.ipynb) — sentence-transformer embedding wrapper
    - [`VectorDBManager`](src/RAG_pipeline.ipynb) — chroma DB persistent client and collection manager
    - [`Retriever`](src/RAG_pipeline.ipynb) — vector-space retriever using stored embeddings
    - [`SimpleRAG`](src/RAG_pipeline.ipynb) — minimal retrieval → LLM flow
    - [`AdvancedRAG`](src/RAG_pipeline.ipynb) — expanded pipeline with query preprocessing and summarization

- Persistent vector DB (example): [documents/vectorDB/pdf_db/chroma.sqlite3](documents/vectorDB/pdf_db/chroma.sqlite3)
- Documents folder: [documents/](documents/) — source PDFs, extracted images, and text

## Quickstart (local)
1. Use Python 3.12 (see `.python-version`).
2. Create and activate virtualenv:
   - Unix: python -m venv .venv && source .venv/bin/activate
   - Windows: python -m venv .venv && .venv\Scripts\activate
3. Install deps:
   ```sh
   pip install -r requirements.txt
   or
   pip install -r requirements.lock
   or
   uv sync
   ```
4. Set secrets (example `.env`):
   - GROQ_API_KEY for Groq-hosted LLMs (if used).
   - You can get and use a Groq API for free (limits apply)
5. Open and run the notebook cells in order: [src/RAG_pipeline.ipynb](src/RAG_pipeline.ipynb). The notebook runs the end-to-end flow:
   - load documents
   - extract images/pages (via `PdfImagesLoader`)
   - chunk documents
   - compute embeddings (`EmbeddingManager`)
   - persist to Chroma (`VectorDBManager`)
   - query via `Retriever` and generate answers with `SimpleRAG` / `AdvancedRAG`

## Usage Notes
- The notebook is designed to be run cell-by-cell. For production integration, port the classes (`EmbeddingManager`, `VectorDBManager`, `Retriever`, `AdvancedRAG`, etc.) into a standalone Python module.
- `src/scratchpad.ipynb` is for experiments and quick tests only; it contains ad-hoc code and examples — do not rely on it for the canonical pipeline.
- Vector DB path is under `documents/vectorDB/` by default.

## Configuration
- Paths and chunking are set near the top of [src/RAG_pipeline.ipynb](src/RAG_pipeline.ipynb):
  - `documents_path`, `vectordb_path`, `chunking_size`, `chunking_step`, `embedding_model_name`
- Adjust `EmbeddingManager` model name to change embedding size / model.

## Extending the pipeline
- Swap embedding models by changing `EmbeddingManager` model_name.
- Plug different LLMs by replacing the `ChatGroq` instances in the notebook with other LLM clients.
- Move notebook classes into a package for reuse in a service or API.

## Troubleshooting
- Ensure `chromadb` persistent path is writable.
- If PDF image extraction fails for specific files, check `PdfImagesLoader` logs and the PyMuPDF version.
- Llava model implementation is incomplete at the moment, therefore can be ignored
