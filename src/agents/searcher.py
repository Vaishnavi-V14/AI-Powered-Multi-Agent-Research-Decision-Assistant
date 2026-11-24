"""
Searcher (Scout) Agent — Upgraded
==================================
Executes all sub-queries from the Planner in PARALLEL using a thread pool,
merging and deduplicating results across web search and local vector search.

Key improvements over v1:
- Consumes Planner's sub_queries (not just the raw question)
- ThreadPoolExecutor for concurrent search across all sub-queries
- Deduplication by content fingerprint to avoid redundant chunks
- Per-result metadata tagging (which sub-query produced it)
"""

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any

from src.state import ResearchState
from src.tools import web_search, local_doc_search

logger = logging.getLogger(__name__)

_MAX_WORKERS = 6
_SEARCH_TIMEOUT = 25  # seconds per search call


def _fingerprint(content: str) -> str:
    """Short hash to deduplicate near-identical snippets."""
    return hashlib.md5(content[:200].encode()).hexdigest()[:12]


def _search_one_query(query: str, query_index: int) -> list[dict[str, Any]]:
    """Run web + local search for a single sub-query. Returns merged results."""
    results: list[dict] = []

    # Web search
    try:
        web_results = web_search.invoke(query)
        for r in web_results:
            r["sub_query"] = query
            r["sub_query_idx"] = query_index
        results.extend(web_results)
    except Exception as exc:
        logger.warning("[Searcher] Web search failed for query '%s': %s", query, exc)

    # Local ChromaDB search
    try:
        doc_results = local_doc_search.invoke(query)
        for r in doc_results:
            r["sub_query"] = query
            r["sub_query_idx"] = query_index
        results.extend(doc_results)
    except Exception as exc:
        logger.warning("[Searcher] Local search failed for query '%s': %s", query, exc)

    logger.debug("[Searcher] Query '%s' → %d results", query, len(results))
    return results


def searcher_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Searcher.

    Executes all Planner sub-queries in parallel and merges results.

    Updates state fields:
      - raw_data  (deduplicated list of result dicts)
      - sources   (deduplicated source list)
    """
    sub_queries = state.get("sub_queries", [])
    question = state.get("question", "")

    # Fallback: if Planner didn't run, use the original question
    if not sub_queries:
        sub_queries = [question]

    logger.info(
        "[Searcher] Executing %d sub-queries in parallel...", len(sub_queries)
    )

    all_results: list[dict] = []
    seen_fingerprints: set[str] = set()

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(sub_queries))) as executor:
        future_map = {
            executor.submit(_search_one_query, query, idx): (idx, query)
            for idx, query in enumerate(sub_queries)
        }

        for future in as_completed(future_map, timeout=60):
            idx, query = future_map[future]
            try:
                results = future.result(timeout=_SEARCH_TIMEOUT)
                # Deduplicate by content fingerprint
                for r in results:
                    fp = _fingerprint(r.get("content", ""))
                    if fp not in seen_fingerprints:
                        seen_fingerprints.add(fp)
                        all_results.append(r)
            except TimeoutError:
                logger.warning("[Searcher] Query %d timed out: '%s'", idx, query)
            except Exception as exc:
                logger.error("[Searcher] Query %d failed: %s", idx, exc)

    # Collect unique sources
    sources = list({r.get("source", "") for r in all_results if r.get("source")})

    logger.info(
        "[Searcher] Total: %d unique results, %d sources from %d sub-queries.",
        len(all_results), len(sources), len(sub_queries),
    )

    return {
        "raw_data": all_results,
        "sources": sources,
    }
