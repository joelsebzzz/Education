"""Configuration values for the PDF RAG QA application."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    """Centralized constants for indexing, retrieval, and generation."""

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_model_name: str = "google/flan-t5-base"
    chunk_size_words: int = 500
    chunk_overlap_words: int = 50
    top_k: int = 3
    similarity_threshold: float = 0.45
    max_new_tokens: int = 150


CONFIG = RAGConfig()
