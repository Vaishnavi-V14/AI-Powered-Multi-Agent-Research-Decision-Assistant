"""
Finalizer (Synthesizer) Agent
==============================
Synthesizes the final polished answer with inline source citations,
then persists the session record to SQLite.
"""

import logging
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.database import save_session
from src.state import ResearchState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a professional research analyst delivering a final report.
You will receive a user question, a summary of verified facts, and a list of sources.

Your task is to produce a final, polished answer that:
1. Opens with a direct answer to the question (1–2 sentences)
2. Elaborates with supporting details organized into clear paragraphs
3. Adds inline citations like [1], [2] wherever a specific claim is backed by a source
4. Ends with a "## Sources" section listing numbered URLs / document names
5. Is written professionally but accessibly

Do NOT fabricate information. Use only what is in the summary and facts.
"""


def finalizer_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Finalizer.

    Produces the final cited answer and stores the session in SQLite.

    Updates state fields:
      - final_answer (string)
      - session_id   (UUID from SQLite)
    """
    question = state.get("question", "")
    summary = state.get("summary", "")
    facts = state.get("facts", [])
    sources = state.get("sources", [])
    validation_status = state.get("validation_status", "PENDING")

    logger.info("[Finalizer] Synthesizing final answer. Status: %s", validation_status)

    cfg = get_settings()
    llm = ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=0.4,
        num_predict=800,
    )

    # Build numbered source reference list
    source_list = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sources))
    facts_text = "\n".join(f"- {f}" for f in facts[:10])

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Verified Summary:\n{summary}\n\n"
                f"Key Facts:\n{facts_text}\n\n"
                f"Available Sources:\n{source_list or 'No external sources.'}"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        final_answer = response.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("[Finalizer] LLM call failed: %s — using summary as answer.", exc)
        final_answer = summary

    # Persist session
    new_state = {**state, "final_answer": final_answer}
    try:
        session_id = save_session(new_state)
    except Exception as exc:  # noqa: BLE001
        logger.error("[Finalizer] Failed to save session: %s", exc)
        session_id = ""

    logger.info("[Finalizer] Done. Session ID: %s", session_id)
    return {
        "final_answer": final_answer,
        "session_id": session_id,
    }
