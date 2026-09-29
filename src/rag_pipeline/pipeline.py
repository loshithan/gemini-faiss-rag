from google.genai import errors, types

from .config import GENERATION_MODEL, TOP_K
from .embedder import embed_text, get_client
from .store import VectorStore

SYSTEM_INSTRUCTION = (
    "You answer questions using only the context passages provided. "
    "Do not use outside knowledge. "
    "If the context does not contain the answer, reply exactly: \"I don't know based on the provided documents.\""
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build a prompt that puts the retrieved chunks before the question.

    Args:
        question (str): The user's question.
        chunks (list[dict]): The retrieved chunks.

    Returns:
        str: The prompt text.
    """
    context = "\n\n".join(f"[{chunk['chunk_id']}]\n{chunk['text']}" for chunk in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def answer_question(question: str, store: VectorStore, top_k: int = TOP_K) -> dict:
    """
    Answer a question using chunks retrieved from the store.

    Args:
        question (str): The question to be answered.
        store (VectorStore): The index to retrieve context from.
        top_k (int): The number of chunks to retrieve. Default is config.TOP_K.

    Returns:
        dict: {"answer": str, "chunks": list of (chunk, score) pairs used as context}.
    """
    results = store.search(embed_text(question), top_k=top_k)
    if not results:
        return {"answer": "No documents are indexed.", "chunks": []}

    prompt = build_prompt(question, [chunk for chunk, _ in results])
    try:
        response = get_client().models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=0),
        )
    except errors.APIError as e:
        raise RuntimeError(f"Generation request failed ({e.code}): {e.message}") from e

    # response.text is None when the model returns no text (e.g. a blocked response)
    answer = response.text or "The model returned no answer."
    return {"answer": answer.strip(), "chunks": results}
