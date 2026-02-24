"""Text generation module using Flan-T5."""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class AnswerGenerator:
    """Generate answers strictly from retrieved context using Flan-T5."""

    PROMPT_TEMPLATE = (
        "Answer the question ONLY using the context below.\n"
        'If the answer is not found in the context, say "Not found in document."\n\n'
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    )

    def __init__(self, model_name: str, max_new_tokens: int = 150) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer from supplied context and question."""
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return answer or "Not found in document."
