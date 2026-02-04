from src.loader import load_documents
from src.splitter import split_documents
from src.embeddings import get_embeddings
# from src.vectorstore import build_vectorstore
# from src.generator import generate_answer
# from src.rag_pipeline import retrieve_context
from src.exceptions import RAGException
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    try:
        logger.info("Starting RAG pipeline")

        docs = load_documents()
        splits = split_documents(docs)

        embeddings = get_embeddings()
        print(embeddings)

    except RAGException as e:
        logger.error(f"RAG pipeline failed: {e}")
        raise e