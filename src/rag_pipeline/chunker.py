from .config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(doc_id: str, text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Splits the input text into overlapping word-based chunks.

    Args:
        doc_id (str): ID of the document the text comes from.
        text (str): The input text to be chunked.
        chunk_size (int): The maximum number of words in each chunk. Default is config.CHUNK_SIZE.
        chunk_overlap (int): The number of words shared by consecutive chunks. Default is config.CHUNK_OVERLAP.

    Returns:
        list[dict]: A list of {"chunk_id", "doc_id", "text", "position"} dicts.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    step = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(words), step):
        position = len(chunks)
        chunks.append({
            "chunk_id": f"{doc_id}::{position}",
            "doc_id": doc_id,
            "text": " ".join(words[start:start + chunk_size]),
            "position": position,
        })
        # Stop once a chunk reaches the end, so the tail isn't repeated in a tiny extra chunk
        if start + chunk_size >= len(words):
            break

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk every {"doc_id", "text"} document.

    Args:
        documents (list[dict]): Documents as returned by loader.load_documents.

    Returns:
        list[dict]: All chunks, in document order.
    """
    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc["doc_id"], doc["text"]))
    return chunks
