from src.loader import load_documents
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore
from src.generator import generate_answer
from src.rag_pipeline import retrieve_context
from src.exceptions import RAGException
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    try:
        logger.info("Starting RAG pipeline")

        docs = load_documents()
        splits = split_documents(docs)

        embeddings = get_embeddings()
        vectorstore = build_vectorstore(splits, embeddings)
        retriever = vectorstore.as_retriever()

        question = (
            "The Gasoline Particulate Filter Full warning has appeared. "
            "What does this mean and what should I do?"
        )

        context = retrieve_context(retriever, question)
        answer = generate_answer(context, question)

        print("\nANSWER:\n", answer)

        logger.info("RAG pipeline completed successfully")

    except RAGException as e:
        logger.error(f"RAG pipeline failed: {e}")
        print("❌ RAG pipeline failed:", e)

    except Exception as e:
        logger.critical(f"Unexpected failure: {e}", exc_info=True)
        print("❌ Unexpected error:", e)


if __name__ == "__main__":
    main()
