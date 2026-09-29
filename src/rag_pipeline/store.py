import faiss
import numpy as np

from .config import EMBEDDING_DIM, TOP_K


class VectorStore:
    """
    A FAISS inner-product index over L2-normalized vectors (i.e. cosine similarity),
    plus a parallel list so that vector i in the index maps to self.chunks[i].
    """

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[dict] = []

    def __len__(self) -> int:
        return self.index.ntotal

    def _to_normalized_matrix(self, embeddings: list[list[float]]) -> np.ndarray:
        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got shape {vectors.shape}")
        faiss.normalize_L2(vectors)  # in place
        return vectors

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """
        Add chunks and their embeddings. Both lists must be in the same order.

        Args:
            chunks (list[dict]): The chunks that were embedded.
            embeddings (list[list[float]]): One embedding per chunk.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Got {len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return
        self.index.add(self._to_normalized_matrix(embeddings))
        self.chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = TOP_K) -> list[tuple[dict, float]]:
        """
        Find the chunks most similar to a query embedding.

        Args:
            query_embedding (list[float]): The query embedding vector.
            top_k (int): The number of results to return. Default is config.TOP_K.

        Returns:
            list[tuple[dict, float]]: (chunk, cosine similarity) pairs, best first.
        """
        if len(self) == 0:
            return []
        query = self._to_normalized_matrix([query_embedding])
        scores, indices = self.index.search(query, min(top_k, len(self)))
        # FAISS pads with -1 when it has fewer results than requested
        return [(self.chunks[i], float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]
