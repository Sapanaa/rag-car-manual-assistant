from langchain_community.vectorstores import FAISS
from src.logger import get_logger
from src.exceptions import VectorStoreError

logger = get_logger(__name__)


def build_vectorstore(documents, embeddings):
    logger.info("Building vector store")

    try:
        if not documents:
            raise VectorStoreError("No documents provided to vector store")

        vectorstore = FAISS.from_documents(documents, embeddings)

        logger.info("Vector store successfully created")
        return vectorstore

    except Exception as e:
        logger.error(f"Vector store creation failed: {e}")
        raise VectorStoreError(str(e)) from e
