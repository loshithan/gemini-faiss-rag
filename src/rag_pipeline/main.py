from .chunker import chunk_documents
from .config import DATA_DIR
from .embedder import embed_chunks
from .loader import load_documents
from .pipeline import answer_question
from .store import VectorStore


def build_store() -> VectorStore:
    """
    Load, chunk and embed every document in DATA_DIR into a new VectorStore.

    Returns:
        VectorStore: The populated store.
    """
    documents = load_documents()
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} documents from {DATA_DIR}, split into {len(chunks)} chunks.")

    store = VectorStore()
    if chunks:
        print("Embedding chunks...")
        store.add(chunks, embed_chunks(chunks))
    return store


def main() -> None:
    try:
        store = build_store()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        return

    if len(store) == 0:
        print(f"No .txt files with content found in {DATA_DIR}.")
        return

    print("Ask a question (empty line or Ctrl+C to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break

        try:
            result = answer_question(question, store)
        except RuntimeError as e:
            print(f"Error: {e}")
            continue

        print(f"\n{result['answer']}\n\nSources:")
        for chunk, score in result["chunks"]:
            print(f"  {chunk['chunk_id']}  (similarity {score:.3f})")


if __name__ == "__main__":
    main()
