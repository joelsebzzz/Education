"""Utilities to load and clean text from PDF files."""

from pathlib import Path
import re
from typing import List

import pdfplumber


class PDFLoader:
    """Load and normalize text content from a PDF document."""

    @staticmethod
    def _normalize_line(line: str) -> str:
        """Normalize whitespace in a line and strip edge spaces."""
        return re.sub(r"\s+", " ", line).strip()

    def load_text(self, pdf_path: str) -> str:
        """Extract cleaned text from a PDF file.

        Args:
            pdf_path: Path to the source PDF file.

        Returns:
            A single cleaned text string preserving page order.

        Raises:
            FileNotFoundError: If the provided path does not exist.
            ValueError: If no usable text was found in the PDF.
            RuntimeError: If PDF parsing fails.
        """
        path = Path(pdf_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        collected_lines: List[str] = []

        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    raw_text = page.extract_text() or ""
                    for line in raw_text.splitlines():
                        normalized = self._normalize_line(line)
                        if normalized:
                            collected_lines.append(normalized)
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            raise RuntimeError(f"Failed to read PDF: {exc}") from exc

        text = "\n".join(collected_lines).strip()
        if not text:
            raise ValueError("The PDF appears to be empty or contains no extractable text.")

        return text
