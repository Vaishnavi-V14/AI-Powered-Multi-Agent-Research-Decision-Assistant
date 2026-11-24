"""
Searcher (Scout) Agent
======================
Runs web search + local ChromaDB search in parallel and merges the raw
results into the shared pipeline state.
"""

import logging
from typing import Any

from src.state import ResearchState
from src.tools import web_search, local_doc_search

logger = logging.getLogger(__name__)


def searcher_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Searcher.

    Fetches raw data from:
      1. Tavily web search
      2. ChromaDB local document store

    Updates state fields:
      - raw_data  (merged list of result dicts)
      - sources   (deduplicated list of source strings)
    """
    question = state.get("question", "")
    logger.info("[Searcher] Running search for: %s", question)

    # --- Web search ----------------------------------------------------------
    web_results: list[dict] = []
    try:
        web_results = web_search.invoke(question)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Searcher] Web search failed: %s", exc)

    # --- Local doc search ----------------------------------------------------
    doc_results: list[dict] = []
    try:
        doc_results = local_doc_search.invoke(question)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Searcher] Local doc search failed: %s", exc)

    all_results = web_results + doc_results

    # Collect unique sources
    sources = list(
        {r.get("source", "") for r in all_results if r.get("source")}
    )

    logger.info(
        "[Searcher] Found %d results (%d web, %d local).",
        len(all_results),
        len(web_results),
        len(doc_results),
    )

    return {
        "raw_data": all_results,
        "sources": sources,
    }
