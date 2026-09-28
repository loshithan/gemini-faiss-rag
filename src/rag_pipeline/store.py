import faiss,numpy as np
from .embedder import compute_cosine_similarity

def create_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    """
    Create a FAISS index from a list of embedding vectors.

    Args:
        embeddings (list[list[float]]): A list of embedding vectors.
        dimension (int): The dimension of the embedding vectors.

    Returns:
        faiss.IndexFlatL2: A FAISS index.
    """
    dimension = len(embeddings[0]) if embeddings else 0
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index    

def search_query_in_index(query_embedding: list[float], index: faiss.IndexFlatL2, top_k: int = 5) -> list[tuple[int, float]]:
    """
    Search for the top K most similar embeddings in the FAISS index for a given query embedding.

    Args:
        query_embedding (list[float]): The query embedding vector.
        index (faiss.IndexFlatL2): The FAISS index to search in.
        top_k (int): The number of top results to return. Default is 5.

    Returns:
        list[tuple[int, float]]: A list of tuples containing the indices and distances of the top K results.
    """
    query_vector = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)
    return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]
