import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import GENERATION_MODEL
from src.logger import get_logger
from src.exceptions import GenerationError

logger = get_logger(__name__)

logger.info("Loading generation model")
tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)
model.eval()


def generate_answer(context: str, question: str) -> str:
    logger.info("Generating answer")

    try:
        prompt = f"""
        You are an assistant for question-answering tasks. Use the following pieces of 
        retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences 
        maximum and keep the answer concise.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200
            )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        logger.info("Answer generated successfully")
        return answer.strip()

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise GenerationError(str(e)) from e
