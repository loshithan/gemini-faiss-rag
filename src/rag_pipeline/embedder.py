import os
from functools import cache

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from .config import API_KEY_ENV_VAR, EMBED_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL


@cache
def get_client() -> genai.Client:
    """
    Create the Gemini client once, reading the API key from the environment (or .env).

    Returns:
        genai.Client: A shared Gemini client.
    """
    load_dotenv()
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV_VAR} is not set. Add it to your environment or to a .env file.")
    return genai.Client(api_key=api_key)


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    try:
        response = get_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIM),
        )
    except errors.APIError as e:
        raise RuntimeError(f"Embedding request failed ({e.code}): {e.message}") from e
    return [embedding.values for embedding in response.embeddings]


def embed_text(text: str) -> list[float]:
    """
    Embed a search query.

    Args:
        text (str): The query to embed.

    Returns:
        list[float]: The embedding vector as a list of floats.
    """
    return _embed([text], task_type="RETRIEVAL_QUERY")[0]


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """
    Embed chunk texts in batches, using the same model as embed_text.

    Args:
        chunks (list[dict]): Chunks as returned by chunker.chunk_documents.

    Returns:
        list[list[float]]: One embedding vector per chunk, in the same order.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        embeddings.extend(_embed(texts[i:i + EMBED_BATCH_SIZE], task_type="RETRIEVAL_DOCUMENT"))
    return embeddings


def compute_cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """
    Compute the cosine similarity between two embedding vectors.

    Args:
        embedding1 (list[float]): The first embedding vector.
        embedding2 (list[float]): The second embedding vector.

    Returns:
        float: The cosine similarity score between the two embeddings.
    """
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    norm1 = sum(a * a for a in embedding1) ** 0.5
    norm2 = sum(b * b for b in embedding2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

if __name__ == "__main__":
    # Example usage
    text1 = "the sky is blue"
    text2 = "the sky appears blue"

    embedding1 = embed_text(text1)
    embedding2 = embed_text(text2)

    cosine_similarity = compute_cosine_similarity(embedding1, embedding2)

    print(f"Cosine similarity between '{text1}' and '{text2}': {cosine_similarity}")

    text3 = "I like pizza"
    embedding3 = embed_text(text3)

    cosine_similarity3 = compute_cosine_similarity(embedding1, embedding3)

    print(f"Cosine similarity between '{text1}' and '{text3}': {cosine_similarity3}")
