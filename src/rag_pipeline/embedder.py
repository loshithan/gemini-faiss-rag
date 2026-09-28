import os
from google import genai
from openai import OpenAI

google_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def embed_text(text: str, model: str = "gemini-embedding-001") -> list[float]:
    """
    Embed text using the specified model.

    Args:
        text (str): The text to embed.
        model (str): The embedding model to use. Default is "gemini-embedding-001".

    Returns:
        list[float]: The embedding vector as a list of floats.
    """
    response = google_client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def embed_chunks(chunks: list[str], model: str = "gemini-embedding-001") -> list[list[float]]:
    """
    Embed a list of text chunks using the specified model.

    Args:
        chunks (list[str]): A list of text chunks to embed.
        model (str): The embedding model to use. Default is "gemini-embedding-001".

    Returns:
        list[list[float]]: A list of embedding vectors, each corresponding to a chunk.
    """
    embeddings = []
    for chunk in chunks:
        embedding = embed_text(chunk, model=model)
        embeddings.append(embedding)
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