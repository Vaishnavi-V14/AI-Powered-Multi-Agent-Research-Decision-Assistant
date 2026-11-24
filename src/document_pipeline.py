"""
Document Ingestion Pipeline
============================
Handles ingestion of diverse document types (PDF, DOCX, TXT, MD, HTML)
into the ChromaDB vector store with:

- Format-aware text extraction
- Recursive character text splitting with overlap
- Rich metadata (source, page, chunk_index, file_type, ingested_at)
- Deduplication via content hash

Usage:
    from src.document_pipeline import ingest_file, ingest_text

    # From a file path
    count = ingest_file("/path/to/report.pdf")

    # From raw text (e.g. uploaded via Streamlit)
    count = ingest_text("AI is used in healthcare...", source="upload.txt")
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.vectorstore import ingest_documents, get_vectorstore

logger = logging.getLogger(__name__)

# Chunking configuration
_CHUNK_SIZE = 1000      # characters per chunk
_CHUNK_OVERLAP = 150    # overlap to preserve context across chunk boundaries


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _content_hash(text: str) -> str:
    """SHA-256 hash of content for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Extractors ────────────────────────────────────────────────────────────────

def _extract_pdf(file_path: Path) -> list[dict[str, Any]]:
    """Extract text from PDF, returning [{text, page}]."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": i + 1})
        return pages
    except ImportError:
        logger.error("pypdf not installed — cannot read PDF files.")
        return []
    except Exception as exc:
        logger.error("PDF extraction failed for %s: %s", file_path, exc)
        return []


def _extract_docx(file_path: Path) -> list[dict[str, Any]]:
    """Extract text from DOCX."""
    try:
        from docx import Document as DocxDoc
        doc = DocxDoc(str(file_path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"text": text, "page": 1}]
    except ImportError:
        logger.error("python-docx not installed — cannot read DOCX files.")
        return []
    except Exception as exc:
        logger.error("DOCX extraction failed for %s: %s", file_path, exc)
        return []


def _extract_text(file_path: Path) -> list[dict[str, Any]]:
    """Read plain text / markdown."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return [{"text": text, "page": 1}]
    except Exception as exc:
        logger.error("Text extraction failed for %s: %s", file_path, exc)
        return []


_EXTRACTOR_MAP = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".doc": _extract_docx,
    ".txt": _extract_text,
    ".md": _extract_text,
    ".markdown": _extract_text,
    ".rst": _extract_text,
}


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_file(file_path: str | Path) -> int:
    """
    Ingest a document file into ChromaDB.

    Supports: PDF, DOCX, TXT, MD.

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        Number of chunks ingested.
    """
    path = Path(file_path).resolve()
    suffix = path.suffix.lower()

    extractor = _EXTRACTOR_MAP.get(suffix)
    if extractor is None:
        logger.warning("Unsupported file type: %s", suffix)
        return 0

    logger.info("[DocPipeline] Extracting %s (%s)...", path.name, suffix)
    pages = extractor(path)

    if not pages:
        logger.warning("[DocPipeline] No content extracted from %s.", path.name)
        return 0

    splitter = _make_splitter()
    docs: list[Document] = []
    now = datetime.now(timezone.utc).isoformat()

    for page_data in pages:
        raw_text = page_data["text"]
        page_num = page_data.get("page", 1)

        chunks = splitter.split_text(raw_text)
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                        "file_type": suffix,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "content_hash": _content_hash(chunk),
                        "ingested_at": now,
                    },
                )
            )

    if not docs:
        return 0

    # Deduplicate by content hash before ingesting
    vs = get_vectorstore()
    existing_ids: set[str] = set()
    try:
        collection = vs._collection  # type: ignore[attr-defined]
        result = collection.get(include=[])
        existing_ids = set(result.get("ids", []))
    except Exception:
        pass

    unique_docs = []
    for doc in docs:
        doc_id = doc.metadata["content_hash"]
        if doc_id not in existing_ids:
            unique_docs.append(doc)

    if not unique_docs:
        logger.info("[DocPipeline] All chunks already in DB — skipping.")
        return 0

    count = ingest_documents(unique_docs)
    logger.info("[DocPipeline] Ingested %d chunks from %s.", count, path.name)
    return count


def ingest_text(text: str, source: str = "upload") -> int:
    """
    Ingest raw text directly (e.g. from Streamlit file upload).

    Args:
        text:   The document text.
        source: A label to use as the source metadata.

    Returns:
        Number of chunks ingested.
    """
    if not text.strip():
        return 0

    splitter = _make_splitter()
    chunks = splitter.split_text(text)
    now = datetime.now(timezone.utc).isoformat()

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": source,
                "file_name": source,
                "file_type": "text",
                "page": 1,
                "chunk_index": i,
                "content_hash": _content_hash(chunk),
                "ingested_at": now,
            },
        )
        for i, chunk in enumerate(chunks)
        if chunk.strip()
    ]

    count = ingest_documents(docs)
    logger.info("[DocPipeline] Ingested %d chunks from text source '%s'.", count, source)
    return count


def get_ingestion_stats() -> dict[str, Any]:
    """Return statistics about the current ChromaDB collection."""
    try:
        vs = get_vectorstore()
        collection = vs._collection  # type: ignore[attr-defined]
        result = collection.get(include=["metadatas"])
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])

        sources: set[str] = set()
        file_types: dict[str, int] = {}
        for m in metadatas:
            if m:
                src = m.get("file_name", "unknown")
                sources.add(src)
                ft = m.get("file_type", "unknown")
                file_types[ft] = file_types.get(ft, 0) + 1

        return {
            "total_chunks": len(ids),
            "unique_sources": len(sources),
            "source_names": sorted(sources),
            "file_types": file_types,
        }
    except Exception as exc:
        logger.warning("Could not fetch ingestion stats: %s", exc)
        return {"total_chunks": 0, "unique_sources": 0, "source_names": [], "file_types": {}}
