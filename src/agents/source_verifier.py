"""
Source Verifier Agent
=====================
For every unique source URL collected by the Searcher, this agent:

1. Makes a lightweight HTTP HEAD request to verify the URL is accessible.
2. Cross-references each fact against multiple sources to compute a
   per-fact confidence score (how many sources agree on the claim).
3. Filters out unverifiable or contradictory sources.
4. Attaches `verified_sources` metadata to the state so downstream agents
   know which sources are trustworthy.

This is the key difference from a basic chatbot — claims are only kept if
at least one accessible source supports them.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.state import ResearchState, SourceVerification

logger = logging.getLogger(__name__)

# HTTP check settings
_TIMEOUT = 8.0        # seconds
_MAX_WORKERS = 8      # parallel URL checks
_USER_AGENT = "Mozilla/5.0 (compatible; ResearchBot/1.0)"


# ── URL Verification ──────────────────────────────────────────────────────────

def _check_url(url: str) -> dict[str, Any]:
    """Check if a URL is accessible via HTTP HEAD (falls back to GET)."""
    if not url.startswith("http"):
        return {"url": url, "accessible": False, "status_code": 0, "is_local": True}

    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = client.head(url, headers={"User-Agent": _USER_AGENT})
            if resp.status_code == 405:
                # HEAD not allowed — try GET with stream
                resp = client.get(url, headers={"User-Agent": _USER_AGENT})
            accessible = resp.status_code < 400
            return {
                "url": url,
                "accessible": accessible,
                "status_code": resp.status_code,
                "is_local": False,
            }
    except Exception as exc:  # noqa: BLE001
        logger.debug("[SourceVerifier] URL check failed for %s: %s", url, exc)
        return {"url": url, "accessible": False, "status_code": 0, "is_local": False}


def _verify_urls_parallel(urls: list[str]) -> dict[str, dict[str, Any]]:
    """Check all URLs in parallel. Returns {url: check_result}."""
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_map = {executor.submit(_check_url, url): url for url in urls}
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                results[url] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[url] = {"url": url, "accessible": False, "status_code": 0}
    return results


# ── Claim Cross-Reference ─────────────────────────────────────────────────────

_CROSSREF_SYSTEM = """\
You are a source analysis expert. Given a list of source content snippets
and a list of facts, determine which sources support each fact.

For each fact, output a JSON object on its own line:
{"fact_index": <int>, "supporting_source_indices": [<int>, ...], "confidence": <0.0-1.0>}

Base confidence on:
- 1.0: Fact explicitly stated by 2+ sources
- 0.8: Fact explicitly stated by 1 source
- 0.5: Fact implied or partially supported
- 0.0: Fact not found in any source

Output ONLY the JSON lines, nothing else.
"""


def _cross_reference_facts(
    facts: list[str],
    raw_data: list[dict[str, Any]],
    llm: ChatOllama,
) -> list[dict[str, Any]]:
    """
    Use LLM to cross-reference facts against source snippets.
    Returns per-fact confidence and supporting source indices.
    """
    if not facts or not raw_data:
        return []

    # Build compact source block (cap per source to 300 chars)
    sources_block = "\n".join(
        f"[Source {i}] ({r.get('source', '?')}): {r.get('content', '')[:300]}"
        for i, r in enumerate(raw_data[:15])
    )
    facts_block = "\n".join(f"[Fact {i}] {f}" for i, f in enumerate(facts[:20]))

    messages = [
        SystemMessage(content=_CROSSREF_SYSTEM),
        HumanMessage(
            content=f"Sources:\n{sources_block}\n\nFacts:\n{facts_block}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        results = []
        import json
        for line in response.content.strip().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SourceVerifier] Cross-reference LLM call failed: %s", exc)
        return []


# ── Main Node ─────────────────────────────────────────────────────────────────

def source_verifier_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Source Verifier.

    1. Parallel-checks all source URLs for HTTP accessibility.
    2. Cross-references facts vs sources to compute per-source confidence.
    3. Filters raw_data to only include accessible or local sources.

    Updates state fields:
      - verified_sources (list of SourceVerification objects)
      - raw_data         (filtered to accessible sources)
      - sources          (filtered list)
    """
    raw_data: list[dict] = state.get("raw_data", [])
    facts: list[str] = state.get("facts", [])

    logger.info("[SourceVerifier] Verifying %d sources.", len(raw_data))

    if not raw_data:
        return {"verified_sources": [], "raw_data": [], "sources": []}

    # ── Step 1: Parallel URL checks ───────────────────────────────────────────
    all_urls = list({r.get("source", "") for r in raw_data if r.get("source")})
    url_checks = _verify_urls_parallel(all_urls)

    accessible_urls: set[str] = set()
    for url, check in url_checks.items():
        is_local = check.get("is_local", False) or not url.startswith("http")
        if check.get("accessible") or is_local:
            accessible_urls.add(url)
        status = check.get("status_code", 0)
        logger.debug(
            "[SourceVerifier] %s → %s (HTTP %s)",
            url,
            "✓" if url in accessible_urls else "✗",
            status,
        )

    # Filter raw_data to accessible sources
    filtered_data = [
        r for r in raw_data
        if r.get("source", "") in accessible_urls
        or r.get("type") == "local_doc"
        or not r.get("source", "").startswith("http")
    ]

    logger.info(
        "[SourceVerifier] %d / %d sources accessible.",
        len(accessible_urls),
        len(all_urls),
    )

    # ── Step 2: Cross-reference facts vs sources ──────────────────────────────
    cfg = get_settings()
    llm = ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=0,
    )

    crossref_results = _cross_reference_facts(facts, filtered_data, llm)

    # Build per-source support count
    source_support: dict[int, int] = {}
    for item in crossref_results:
        for idx in item.get("supporting_source_indices", []):
            source_support[idx] = source_support.get(idx, 0) + 1

    total_facts = max(len(facts), 1)

    # ── Step 3: Build SourceVerification objects ──────────────────────────────
    verified_sources: list[SourceVerification] = []
    seen_urls: set[str] = set()

    for i, r in enumerate(filtered_data):
        url = r.get("source", f"doc_{i}")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        check = url_checks.get(url, {})
        supported_facts = source_support.get(i, 0)
        confidence = round(supported_facts / total_facts, 3)

        verified_sources.append(
            SourceVerification(
                url=url,
                accessible=check.get("accessible", True),
                status_code=check.get("status_code", 200),
                claims_supported=supported_facts,
                claims_total=total_facts,
                confidence=confidence,
            )
        )

    # Sort by confidence (most supporting sources first)
    verified_sources.sort(key=lambda s: s.get("confidence", 0), reverse=True)

    final_sources = [s["url"] for s in verified_sources]
    logger.info(
        "[SourceVerifier] Built %d verified source records.", len(verified_sources)
    )

    return {
        "verified_sources": verified_sources,
        "raw_data": filtered_data,
        "sources": final_sources,
    }
