"""Embedding model and FAISS index management."""

from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:
    """Encode text chunks, store vectors in FAISS, and perform retrieval."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name, device="cpu")
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: List[str] = []

    @staticmethod
    def _to_similarity(distances: np.ndarray) -> np.ndarray:
        """Convert L2 distances to bounded similarity scores in (0, 1]."""
        return 1.0 / (1.0 + distances)

    def build_index(self, chunks: List[str]) -> None:
        """Build a FAISS IndexFlatL2 for the provided chunks."""
        if not chunks:
            raise ValueError("Cannot build index with empty chunks list.")

        self.chunks = chunks
        embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> Tuple[List[str], List[float]]:
        """Retrieve top-k chunks and similarity scores for a query."""
        if self.index is None:
            raise ValueError("Index has not been built. Call build_index first.")

        query_vector = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        query_vector = query_vector.astype("float32")

        k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_vector, k)

        retrieved_chunks: List[str] = []
        retrieved_distances = distances[0]

        for idx in indices[0]:
            if idx == -1:
                continue
            retrieved_chunks.append(self.chunks[idx])

        similarity_scores = self._to_similarity(retrieved_distances[: len(retrieved_chunks)]).tolist()
        return retrieved_chunks, similarity_scores
