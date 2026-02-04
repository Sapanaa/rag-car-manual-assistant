from langchain_community.document_loaders import UnstructuredHTMLLoader
from src.config import DATA_PATH
from src.logger import get_logger
from src.exceptions import DocumentLoadError

logger = get_logger(__name__)

def load_documents():
    logger.info("Starting document loading")
    try:
        loader = UnstructuredHTMLLoader(DATA_PATH)
        documents = loader.load()

        if not documents:
            raise DocumentLoadError("No documents were loaded.")

        logger.info(f"Loaded {len(documents)} document(s)")
        return documents


    except FileNotFoundError as e:
        logger.error(f"Failed to load documents: {e}")
        raise DocumentLoadError(str(e))
