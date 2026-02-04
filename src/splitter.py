from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.logger import get_logger
from src.exceptions import TextSplitError

logger = get_logger(__name__)


def split_documents(documents):
    logger.info("Starting text splitting")

    try:
        if not documents:
            raise TextSplitError("No documents provided for splitting")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        splits = splitter.split_documents(documents)

        if not splits:
            raise TextSplitError("Text splitting produced no chunks")

        logger.info(f"Created {len(splits)} text chunks")
        return splits

    except Exception as e:
        logger.error(f"Text splitting failed: {e}")
        raise TextSplitError(str(e)) from e
