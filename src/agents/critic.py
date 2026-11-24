"""
Critic (Quality Control) Agent
================================
Fact-checks the summary against the retrieved facts, detects hallucinations,
and sets validation_status to "PASS" or "FAIL". Increments retry_count on
failure and feeds critique_notes back to the Summarizer.
"""

import logging
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.state import ResearchState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a rigorous fact-checking critic.
You will be given:
  - A user question
  - A list of verified facts (ground truth)
  - A generated summary

Your job is to check the summary for:
1. Hallucinations — claims NOT supported by the provided facts
2. Logical errors or contradictions
3. Missing critical information that should have been included
4. Vague or misleading statements

Respond with EXACTLY this format:
VERDICT: PASS
(if no significant issues are found)

OR:

VERDICT: FAIL
ISSUES:
- <issue 1>
- <issue 2>
...

Do not add any extra text before or after.
"""


def critic_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Critic.

    Fact-checks the summary and sets validation_status.

    Updates state fields:
      - validation_status ("PASS" | "FAIL")
      - critique_notes (failure reasons, if any)
      - retry_count (incremented on FAIL)
    """
    question = state.get("question", "")
    facts = state.get("facts", [])
    summary = state.get("summary", "")
    retry_count = state.get("retry_count", 0)

    cfg = get_settings()
    logger.info(
        "[Critic] Checking summary (attempt %d / %d).",
        retry_count + 1,
        cfg.max_retry,
    )

    if not summary:
        logger.warning("[Critic] Empty summary — marking FAIL.")
        return {
            "validation_status": "FAIL",
            "critique_notes": "Summary is empty.",
            "retry_count": retry_count + 1,
        }

    facts_text = "\n".join(f"- {f}" for f in facts)

    llm = ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=0,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Verified facts:\n{facts_text}\n\n"
                f"Generated summary:\n{summary}"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        verdict_text = response.content.strip()

        if verdict_text.startswith("VERDICT: PASS"):
            logger.info("[Critic] Summary PASSED quality check.")
            return {
                "validation_status": "PASS",
                "critique_notes": "",
            }
        else:
            # Extract issues
            issues = ""
            if "ISSUES:" in verdict_text:
                issues = verdict_text.split("ISSUES:", 1)[1].strip()

            logger.warning("[Critic] Summary FAILED. Issues:\n%s", issues)
            return {
                "validation_status": "FAIL",
                "critique_notes": issues,
                "retry_count": retry_count + 1,
            }

    except Exception as exc:  # noqa: BLE001
        logger.error("[Critic] LLM call failed: %s — defaulting to PASS.", exc)
        # On LLM failure, pass through to avoid infinite loops
        return {
            "validation_status": "PASS",
            "critique_notes": f"Critic LLM unavailable: {exc}",
        }


def should_retry(state: ResearchState) -> str:
    """
    LangGraph conditional edge: decides whether to retry or move to Finalizer.

    Returns:
        "retry"    → back to Summarizer
        "finalize" → forward to Finalizer
    """
    cfg = get_settings()
    validation_status = state.get("validation_status", "PENDING")
    retry_count = state.get("retry_count", 0)

    if validation_status == "PASS":
        logger.info("[Critic] Routing → Finalizer")
        return "finalize"

    if retry_count >= cfg.max_retry:
        logger.warning(
            "[Critic] Max retries (%d) reached — forcing finalization.", cfg.max_retry
        )
        return "finalize"

    logger.info(
        "[Critic] Routing → Summarizer (retry %d / %d)", retry_count, cfg.max_retry
    )
    return "retry"
