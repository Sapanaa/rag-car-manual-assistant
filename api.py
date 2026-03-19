from fastapi import FastAPI
from pydantic import BaseModel
from src.logger import get_logger

from src.loader import load_documents
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore
from src.generator import generate_answer
from src.rag_pipeline import retrieve_context

logger = get_logger(__name__)

app = FastAPI()

# 🔥 Initialize once (IMPORTANT)
docs = load_documents()
splits = split_documents(docs)

embeddings = get_embeddings()
vectorstore = build_vectorstore(splits, embeddings)
retriever = vectorstore.as_retriever()


class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        logger.info(f"Received question: {req.question}")

        context = retrieve_context(retriever, req.question)
        answer = generate_answer(context, req.question)

        return {
            "question": req.question,
            "answer": answer
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}