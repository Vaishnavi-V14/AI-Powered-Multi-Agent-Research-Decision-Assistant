"""
Summarizer (Compressor) Agent
==============================
Calls the local Ollama LLM to compress the retrieved facts into concise,
well-structured key points ready for quality-checking.
"""

import logging
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.state import ResearchState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a world-class research summarizer.
Given a user question and a list of relevant facts, produce a clear,
well-structured summary that:
- Directly answers the question
- Is written in plain English (no jargon without explanation)
- Groups related ideas into short paragraphs
- Is concise but comprehensive (200–400 words)
- Does NOT invent information — stick only to the provided facts

Do NOT include any preamble like "Here is a summary:". Just output the summary.
"""


def summarizer_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Summarizer.

    Compresses facts into a coherent summary using Ollama.

    Updates state fields:
      - summary (string)
    """
    question = state.get("question", "")
    facts = state.get("facts", [])
    retry_count = state.get("retry_count", 0)
    critique_notes = state.get("critique_notes", "")

    logger.info(
        "[Summarizer] Summarizing %d facts (attempt %d).", len(facts), retry_count + 1
    )

    if not facts:
        logger.warning("[Summarizer] No facts to summarize — returning empty summary.")
        return {"summary": "No relevant information was found for this question."}

    facts_text = "\n".join(f"- {f}" for f in facts)

    # On retry, include critique feedback so the LLM can self-correct
    retry_note = ""
    if retry_count > 0 and critique_notes:
        retry_note = (
            f"\n\nPrevious attempt was rejected for the following reasons:\n"
            f"{critique_notes}\n"
            f"Please correct these issues in your new summary."
        )

    cfg = get_settings()
    llm = ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=0.3,
        num_predict=600,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Facts:\n{facts_text}"
                f"{retry_note}"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        summary = response.content.strip()
        logger.info("[Summarizer] Summary generated (%d chars).", len(summary))
        return {"summary": summary}
    except Exception as exc:  # noqa: BLE001
        logger.error("[Summarizer] LLM call failed: %s", exc)
        # Fallback: join facts as-is
        fallback = " ".join(facts[:5])
        return {"summary": fallback}
