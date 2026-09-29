# gemini-faiss-rag

A minimal Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python, using **Google Gemini** for embeddings and generation and **FAISS** for vector search.

It answers questions about your own `.txt` documents, and only from those documents. When the answer isn't in the documents, it says so instead of guessing, and every answer lists the chunks it was based on.

This is a learning project. It doesn't use a framework like LangChain or LlamaIndex, so every step of RAG is small, readable code you can follow from end to end.

## How it works

The pipeline runs in two phases.

```
INDEXING (once, at startup)
  data/*.txt ──► load ──► chunk ──► embed ──► FAISS index
                          (200-word chunks,   (Gemini,     (cosine similarity,
                           40-word overlap)    768 dims)    + chunk lookup list)

QUERYING (every question)
  question ──► embed ──► search index ──► top-k chunks ──► grounded prompt ──► Gemini ──► answer + sources
```

1. **Load:** reads every `.txt` file in `data/`.
2. **Chunk:** splits each document into overlapping word-based chunks, so a sentence at a boundary still appears whole in at least one chunk.
3. **Embed:** turns each chunk into a 768-number vector with `gemini-embedding-001`.
4. **Index:** normalizes the vectors to length 1 and stores them in a FAISS inner-product index, which makes the scores cosine similarity. A list stored alongside maps each vector position back to its chunk.
5. **Retrieve:** embeds the question with the same model and finds the `top_k` most similar chunks.
6. **Generate:** sends the retrieved chunks and the question to a Gemini chat model. The system instruction tells it to answer only from the context and to say it doesn't know otherwise.

## Project structure

```
src/rag_pipeline/
├── config.py     # every setting: paths, chunk size, top-k, model names
├── loader.py     # data/*.txt  →  [{doc_id, text}]
├── chunker.py    # text  →  [{chunk_id, doc_id, text, position}]
├── embedder.py   # Gemini client and embeddings (document and query modes)
├── store.py      # VectorStore: FAISS IndexFlatIP + chunk lookup list
├── pipeline.py   # answer_question(): retrieve → prompt → generate
└── main.py       # builds the index, then runs an interactive Q&A loop
data/
└── photosynthesis.txt   # sample document for testing
```

## Setup

You need Python 3.14+, [uv](https://docs.astral.sh/uv/), and a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone <your-repo-url>
cd gemini-faiss-rag
uv sync
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-key-here
```

> `.env` holds a secret. Make sure it's listed in `.gitignore` and never commit it.

## Usage

Put one or more `.txt` files in `data/`, then run:

```bash
uv run rag-pipeline
```

Example session:

```
Loaded 1 documents from .../data, split into 6 chunks.
Embedding chunks...
Ask a question (empty line or Ctrl+C to quit).

> what is photosynthasis

Based on the provided documents, photosynthesis is the process by which plants, algae,
and some bacteria convert light energy into chemical energy stored in sugar. It takes in
carbon dioxide and water and, using energy from sunlight, produces glucose and oxygen.

Sources:
  photosynthesis::0  (similarity 0.723)
  photosynthesis::2  (similarity 0.689)
  photosynthesis::5  (similarity 0.687)
  photosynthesis::1  (similarity 0.667)
```

Even with the misspelling, the right chunk was retrieved, because embeddings capture meaning rather than exact spelling.

If a question isn't covered by the documents, you get:

```
> Who discovered DNA?

I don't know based on the provided documents.
```

## Configuration

All settings are in `src/rag_pipeline/config.py`.

| Setting | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | `200` | Words per chunk |
| `CHUNK_OVERLAP` | `40` | Words shared by consecutive chunks |
| `TOP_K` | `4` | Chunks retrieved per question |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Used for both chunks and questions |
| `EMBEDDING_DIM` | `768` | Vector size (the model supports 768, 1536 or 3072) |
| `GENERATION_MODEL` | `gemini-3.8-flash` | Model that writes the answer |

## Design notes

- **Cosine similarity with FAISS:** FAISS's `IndexFlatIP` scores results by dot product. After normalizing every vector to length 1, the dot product is the cosine similarity. The same normalization runs when adding vectors and when searching.
- **One model, two modes:** chunks are embedded as `RETRIEVAL_DOCUMENT` and questions as `RETRIEVAL_QUERY`, both with the same model and dimension, so the vectors are comparable.
- **Grounding:** the prompt labels each chunk with its ID, and a system instruction limits the model to that context. `temperature=0` keeps answers as consistent as possible between runs.
- **Flat (exact) search:** every query is compared against every chunk. That's exact, and fast enough for small collections.

## Evaluating it

Ask questions whose answers you know are in `data/`, and some whose answers aren't. Then check two things:

- **Retrieval:** does the chunk containing the answer appear under Sources, with a clearly higher score than the rest?
- **Hallucination:** for questions the documents don't answer, does it say it doesn't know instead of using outside knowledge?

## Limitations

- The index lives in memory and is rebuilt, including re-embedding every chunk, on each run.
- Only plain `.txt` files are supported.
- Retrieval is vector-only, with no keyword search and no re-ranking.
- API errors are reported, but failed requests aren't retried.

## Possible next steps

- Save the index to disk (`faiss.write_index` plus the chunks as JSON) so it isn't rebuilt on every run.
- Retry rate-limited (429) and overloaded (503) requests automatically.
- Add an automated evaluation script with expected answers.
- Support PDF and Markdown files.
- Move to a managed service, such as Azure AI Search with Azure OpenAI, for a production system.
