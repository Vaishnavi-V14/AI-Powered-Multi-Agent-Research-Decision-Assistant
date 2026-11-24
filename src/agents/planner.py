"""
Planner Agent
=============
Decomposes a complex research question into 3-5 independent, targeted
sub-queries that the Searcher can execute in parallel. Each sub-query
targets a different angle or aspect of the original question, maximising
recall and minimising overlap between search results.
"""

import json
import logging
import re
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.state import ResearchState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research strategist. Your job is to decompose a user's research
question into 3 to 5 specific, independent sub-queries.

Rules:
- Each sub-query must target a DIFFERENT aspect of the question
- Sub-queries must be specific enough to retrieve distinct information
- Avoid overlap between sub-queries
- Sub-queries should be optimised for web search (not conversational)
- Do NOT answer the question — only generate sub-queries

Output ONLY valid JSON in this exact format:
{
  "sub_queries": [
    "sub-query 1",
    "sub-query 2",
    "sub-query 3"
  ]
}
"""


def planner_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Planner.

    Decomposes the user question into targeted sub-queries.

    Updates state fields:
      - sub_queries (list of search query strings)
    """
    question = state.get("question", "")
    logger.info("[Planner] Decomposing question: %s", question)

    cfg = get_settings()
    llm = ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=0.2,
        format="json",
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Research question: {question}\n\nGenerate 3-5 targeted sub-queries."
        ),
    ]

    sub_queries = []
    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Parse JSON response
        data = json.loads(raw)
        sub_queries = data.get("sub_queries", [])

        # Validate — ensure we got non-empty strings
        sub_queries = [q.strip() for q in sub_queries if isinstance(q, str) and q.strip()]

        if not sub_queries:
            raise ValueError("LLM returned empty sub_queries list")

    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("[Planner] JSON parse failed (%s) — extracting from raw text.", exc)
        # Fallback: extract quoted strings or numbered lines
        sub_queries = _extract_fallback_queries(response.content if 'response' in dir() else "", question)

    except Exception as exc:  # noqa: BLE001
        logger.error("[Planner] LLM call failed: %s — using question as sole query.", exc)
        sub_queries = [question]

    # Always include the original question as a safety net
    if question not in sub_queries:
        sub_queries.insert(0, question)

    # Cap at 5 sub-queries to control cost/latency
    sub_queries = sub_queries[:5]

    logger.info("[Planner] Generated %d sub-queries: %s", len(sub_queries), sub_queries)
    return {"sub_queries": sub_queries}


def _extract_fallback_queries(raw_text: str, original: str) -> list[str]:
    """Extract queries from non-JSON LLM output."""
    queries = []

    # Try numbered list: "1. ...", "2. ..."
    for match in re.finditer(r'^\s*\d+[\.\)]\s*(.+)$', raw_text, re.MULTILINE):
        q = match.group(1).strip().strip('"').strip("'")
        if q:
            queries.append(q)

    # Try quoted strings if no numbered list found
    if not queries:
        for match in re.finditer(r'"([^"]{10,})"', raw_text):
            q = match.group(1).strip()
            if q:
                queries.append(q)

    return queries if queries else [original]
