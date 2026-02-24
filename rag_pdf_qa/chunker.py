"""Word-based chunking logic for document text."""

from typing import List


class TextChunker:
    """Create overlapping word chunks from text."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0:
            raise ValueError("overlap must be 0 or greater")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split text into ordered overlapping chunks.

        Args:
            text: Source text.

        Returns:
            List of chunk strings.
        """
        words = text.split()
        if not words:
            return []

        chunks: List[str] = []
        step = self.chunk_size - self.overlap

        for start in range(0, len(words), step):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break

        return chunks
