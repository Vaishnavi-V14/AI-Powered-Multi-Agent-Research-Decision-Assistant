"""ChromaDB vector store — setup, ingestion, and similarity search."""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

from src.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "research_docs"


def _get_embeddings() -> OllamaEmbeddings:
    """Return Ollama-based embeddings (uses nomic-embed-text via Ollama)."""
    cfg = get_settings()
    return OllamaEmbeddings(
        base_url=cfg.ollama_base_url,
        model="nomic-embed-text",  # lightweight embedding model
    )


def get_vectorstore() -> Chroma:
    """
    Return a persistent ChromaDB vector store backed by Ollama embeddings.
    Creates the persistence directory if it does not exist.
    """
    cfg = get_settings()
    persist_dir = str(Path(cfg.chroma_persist_dir).resolve())
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    embeddings = _get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    logger.info("ChromaDB loaded from %s", persist_dir)
    return vectorstore


def ingest_documents(documents: list[Document]) -> int:
    """
    Add documents to the persistent vector store.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        Number of documents ingested.
    """
    if not documents:
        logger.warning("ingest_documents called with empty list.")
        return 0

    vs = get_vectorstore()
    vs.add_documents(documents)
    logger.info("Ingested %d documents into ChromaDB.", len(documents))
    return len(documents)


def similarity_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    Search the vector store for documents similar to *query*.

    Returns:
        List of dicts with 'content' and 'source' keys.
    """
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    results = []
    for doc in docs:
        results.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "local_doc"),
                "type": "local_doc",
            }
        )
    logger.info("ChromaDB similarity search returned %d results.", len(results))
    return results
