class RAGException(Exception):
    """Base exception for the RAG system."""
    pass


class DocumentLoadError(RAGException):
    """Raised when document loading fails."""
    pass


class TextSplitError(RAGException):
    """Raised when text splitting fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector store creation or retrieval fails."""
    pass


class GenerationError(RAGException):
    """Raised when LLM generation fails."""
    pass
