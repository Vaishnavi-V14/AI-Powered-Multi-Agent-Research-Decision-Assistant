"""LangChain tool definitions wrapping Tavily web search and ChromaDB retrieval."""

import logging
from typing import Any

from langchain_core.tools import tool
from tavily import TavilyClient

from src.config import get_settings
from src.vectorstore import similarity_search

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> list[dict[str, Any]]:
    """
    Search the web for recent, relevant information using Tavily.

    Args:
        query: The search query string.

    Returns:
        List of result dicts with 'content', 'url', and 'title' keys.
    """
    cfg = get_settings()
    if not cfg.tavily_api_key:
        logger.warning("TAVILY_API_KEY not set — web search disabled.")
        return []

    try:
        client = TavilyClient(api_key=cfg.tavily_api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=7,
            include_answer=False,
        )
        results = []
        for r in response.get("results", []):
            results.append(
                {
                    "content": r.get("content", ""),
                    "source": r.get("url", ""),
                    "title": r.get("title", ""),
                    "type": "web",
                }
            )
        logger.info("Tavily returned %d results for query: %s", len(results), query)
        return results
    except Exception as exc:  # noqa: BLE001
        logger.error("Tavily search error: %s", exc)
        return []


@tool
def local_doc_search(query: str) -> list[dict[str, Any]]:
    """
    Search the local ChromaDB vector store for relevant document chunks.

    Args:
        query: The search query string.

    Returns:
        List of result dicts with 'content' and 'source' keys.
    """
    try:
        return similarity_search(query, k=5)
    except Exception as exc:  # noqa: BLE001
        logger.error("ChromaDB search error: %s", exc)
        return []


# Convenience list for binding to agents
ALL_TOOLS = [web_search, local_doc_search]
