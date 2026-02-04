from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL
from src.logger import get_logger
from src.exceptions import EmbeddingError

logger = get_logger(__name__)


def get_embeddings():
    logger.info("Initializing embedding model")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")
        return embeddings

    except Exception as e:
        logger.error(f"Embedding initialization failed: {e}")
        raise EmbeddingError(str(e)) from e
