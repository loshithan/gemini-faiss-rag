# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project uses `uv` (Python 3.14, `uv_build` backend, `src/` layout). Run these from the repo root:

```sh
uv sync                                  # install dependencies into .venv
uv run rag-pipeline                      # build the index from data/*.txt, then start the interactive Q&A loop
uv run python -m rag_pipeline.embedder   # quick check of embeddings and the API key (prints cosine similarities)
```

The project has no tests, linter config or CI yet.

A `GOOGLE_API_KEY` must be set in the environment or in a `.env` file at the repo root. `embedder.get_client()` loads it with `python-dotenv` the first time it is called.

## Architecture

This is a small retrieval-augmented generation pipeline over plain-text files. It uses Gemini for both embeddings and generation, and FAISS for vector search. Everything runs in memory: the index is rebuilt from scratch every time `main` runs, and nothing is saved to disk.

Data flow (`main.build_store` → `main.main` loop):

1. `loader.load_documents` reads every non-empty `data/*.txt` into `{"doc_id", "text"}` dicts. `doc_id` is the file stem.
2. `chunker.chunk_documents` splits them into overlapping chunks, measured in **words** rather than tokens. Each chunk is `{"chunk_id": "<doc_id>::<position>", "doc_id", "text", "position"}`.
3. `embedder.embed_chunks` calls the API in batches of `EMBED_BATCH_SIZE` with task type `RETRIEVAL_DOCUMENT`. Queries go through `embed_text` with task type `RETRIEVAL_QUERY`. If you change one side, change the other, because documents and queries must use the same model and dimension.
4. `store.VectorStore` wraps `faiss.IndexFlatIP` over L2-normalized vectors, so its scores are cosine similarities. It keeps a parallel `self.chunks` list, and FAISS row `i` maps to `self.chunks[i]`. Any new mutation (delete, persist, reload) has to keep the two in sync.
5. `pipeline.answer_question` retrieves `TOP_K` chunks, builds a prompt with each passage tagged `[chunk_id]`, and calls `GENERATION_MODEL` at temperature 0. The system instruction limits answers to the retrieved context and falls back to a fixed "I don't know based on the provided documents." reply. It returns `{"answer", "chunks": [(chunk, score), ...]}`.

Conventions across modules:
- All tunables live in `config.py`: chunk size and overlap, `TOP_K`, model names, `EMBEDDING_DIM`, batch size and the API key env var name. Modules import these as default argument values. `EMBEDDING_DIM` must match both the embedding request's `output_dimensionality` and the FAISS index dimension.
- `config.PROJECT_ROOT` is computed as `parents[2]` of `config.py`, so `DATA_DIR` depends on the `src/rag_pipeline/` layout.
- The Gemini client is a process-wide singleton (`@cache` on `get_client`) and is shared by the embedder and the pipeline.
- API errors (`google.genai.errors.APIError`) are re-raised as `RuntimeError`. `main` catches `RuntimeError` and `FileNotFoundError` to print a message instead of a traceback.
- Documents and chunks are plain dicts, not classes.
- `anthropic` and `sentence-transformers` are listed in `pyproject.toml` but no code imports them yet.
