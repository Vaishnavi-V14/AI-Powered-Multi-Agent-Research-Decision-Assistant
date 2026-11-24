"""
Retriever (Filter) Agent
========================
Uses an Ollama LLM to rank and filter the raw search results, keeping
only the most relevant facts and deduplicated sources.
"""

import logging
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.state import ResearchState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert research filter.
You will receive a user question and a list of raw search results.
Your job is to:
1. Read every result carefully.
2. Keep only the facts that are DIRECTLY relevant to the question.
3. Discard duplicate or low-quality information.
4. Return a numbered list of distinct, concise factual statements (max 15).
   Each fact should be one sentence.

Output format (plain text, no extra commentary):
1. <fact>
2. <fact>
...
"""


def retriever_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Retriever.

    Filters raw_data into a concise list of relevant facts using Ollama.

    Updates state fields:
      - facts (list of relevant factual strings)
    """
    question = state.get("question", "")
    raw_data = state.get("raw_data", [])
    logger.info("[Retriever] Filtering %d raw results.", len(raw_data))

    if not raw_data:
        logger.warning("[Retriever] No raw data to filter.")
        return {"facts": []}

    # Build a compact text block from raw results (truncate each to 500 chars)
    results_text = "\n\n".join(
        f"[{i+1}] Source: {r.get('source','?')}\n{r.get('content','')[:500]}"
        for i, r in enumerate(raw_data[:20])  # cap at 20 to avoid context overflow
    )

    cfg = get_settings()
    llm = ChatOllama(base_url=cfg.ollama_base_url, model=cfg.ollama_model, temperature=0)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Raw search results:\n{results_text}\n\n"
                "Return only the numbered list of relevant facts."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw_facts = response.content.strip()

        # Parse numbered list into individual strings
        facts = []
        for line in raw_facts.splitlines():
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                facts.append(line.split(". ", 1)[1].strip())

        if not facts:
            # Fallback: use lines directly
            facts = [l.strip() for l in raw_facts.splitlines() if l.strip()]

        logger.info("[Retriever] Extracted %d relevant facts.", len(facts))
        return {"facts": facts}

    except Exception as exc:  # noqa: BLE001
        logger.error("[Retriever] LLM call failed: %s", exc)
        # Graceful degradation: extract snippets directly
        facts = [r.get("content", "")[:200] for r in raw_data[:10] if r.get("content")]
        return {"facts": facts}
