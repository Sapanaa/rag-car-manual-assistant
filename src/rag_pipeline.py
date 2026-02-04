from src.logger import get_logger
from src.exceptions import VectorStoreError

logger = get_logger(__name__)


def retrieve_context(retriever, question: str) -> str:
    logger.info("Retrieving relevant context")

    try:
        docs = retriever.invoke(question)

        if not docs:
            raise VectorStoreError("No relevant documents retrieved")

        context = "\n".join(doc.page_content for doc in docs)

        logger.info(f"Retrieved {len(docs)} relevant documents")
        return context

    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise VectorStoreError(str(e)) from e
