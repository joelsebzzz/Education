"""CLI entry point for CPU-only PDF RAG QA."""

from typing import List

from chunker import TextChunker
from config import CONFIG
from embedder import EmbeddingRetriever
from generator import AnswerGenerator
from pdf_loader import PDFLoader


class RAGApplication:
    """Coordinates document ingestion, retrieval, and answer generation."""

    def __init__(self) -> None:
        self.pdf_loader = PDFLoader()
        self.chunker = TextChunker(CONFIG.chunk_size_words, CONFIG.chunk_overlap_words)
        self.retriever = EmbeddingRetriever(CONFIG.embedding_model_name)
        self.generator = AnswerGenerator(CONFIG.generation_model_name, CONFIG.max_new_tokens)

    @staticmethod
    def _is_retrieval_confident(similarity_scores: List[float], threshold: float) -> bool:
        if not similarity_scores:
            return False
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        return avg_similarity >= threshold

    def run(self) -> None:
        """Run the interactive CLI workflow."""
        pdf_path = input("Enter PDF file path: ").strip()
        if not pdf_path:
            print("PDF path is required.")
            return

        try:
            text = self.pdf_loader.load_text(pdf_path)
            chunks = self.chunker.chunk_text(text)
            if not chunks:
                print("No chunks were created from the PDF text.")
                return

            self.retriever.build_index(chunks)
            print(f"Index built successfully with {len(chunks)} chunks.")
        except Exception as exc:
            print(f"Initialization failed: {exc}")
            return

        print("Ask questions about the document (type 'exit' to quit).")
        while True:
            question = input("\nQuestion: ").strip()
            if not question:
                print("Please enter a question.")
                continue
            if question.lower() == "exit":
                print("Goodbye!")
                break

            try:
                retrieved_chunks, similarity_scores = self.retriever.search(question, top_k=CONFIG.top_k)
                if not self._is_retrieval_confident(similarity_scores, CONFIG.similarity_threshold):
                    print("Answer: Not found in document.")
                    continue

                context = "\n\n".join(retrieved_chunks)
                answer = self.generator.generate_answer(context, question)
                print(f"Answer: {answer}")
            except Exception as exc:
                print(f"Error while answering question: {exc}")


if __name__ == "__main__":
    app = RAGApplication()
    app.run()
