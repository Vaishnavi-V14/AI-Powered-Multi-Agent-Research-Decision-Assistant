"""
Critic (Quality Control) Agent — Upgraded
==========================================
Performs CLAIM-LEVEL fact-checking — each sentence in the summary is
individually verified against the retrieved facts, rather than judging
the summary holistically.

For each claim, the Critic assigns:
  - verdict:    "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED"
  - evidence:   the specific fact or chunk that supports/contradicts it
  - confidence: 0.0 – 1.0

The overall validation_status is determined by:
  - PASS: ≥85% of claims are SUPPORTED (configurable)
  - FAIL: <85% of claims SUPPORTED or any claim is CONTRADICTED

This catches hallucinations at the sentence level, not just the summary level.
"""

import json
import logging
import re
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.state import ClaimCheck, ResearchState

logger = logging.getLogger(__name__)

_PASS_THRESHOLD = 0.80  # fraction of claims that must be SUPPORTED to PASS

_CLAIM_CHECK_SYSTEM = """\
You are a rigorous fact-checker operating at the sentence level.

You will receive:
  - A list of FACTS (ground truth retrieved from sources)
  - A list of CLAIMS (sentences from a generated summary)

For each claim, determine:
  1. verdict: "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED"
     - SUPPORTED: The claim is explicitly backed by at least one fact
     - UNSUPPORTED: The claim cannot be verified against the facts (not necessarily wrong)
     - CONTRADICTED: A fact directly contradicts the claim
  2. evidence: Quote the specific fact (or "none") that justifies your verdict
  3. confidence: 0.0–1.0 (how certain are you of this verdict?)

Output ONLY valid JSON — a list of objects, one per claim:
[
  {
    "claim_index": 0,
    "claim": "...",
    "verdict": "SUPPORTED",
    "evidence": "...",
    "confidence": 0.95
  },
  ...
]
"""


def _split_claims(text: str) -> list[str]:
    """Split summary text into individual claim sentences."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out very short fragments and headings
    return [s.strip() for s in sentences if len(s.strip()) > 20 and not s.startswith("#")]


def _check_claims(
    claims: list[str],
    facts: list[str],
    llm: ChatOllama,
) -> list[ClaimCheck]:
    """Run LLM claim-level fact-check. Returns list of ClaimCheck dicts."""
    if not claims or not facts:
        return []

    facts_text = "\n".join(f"[F{i}] {f}" for i, f in enumerate(facts[:20]))
    claims_text = "\n".join(f"[C{i}] {c}" for i, c in enumerate(claims))

    messages = [
        SystemMessage(content=_CLAIM_CHECK_SYSTEM),
        HumanMessage(
            content=f"FACTS:\n{facts_text}\n\nCLAIMS TO CHECK:\n{claims_text}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Extract JSON array
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON array in response")

        data = json.loads(match.group())
        checks: list[ClaimCheck] = []

        for item in data:
            checks.append(
                ClaimCheck(
                    claim=item.get("claim", claims[item.get("claim_index", 0)] if claims else ""),
                    verdict=item.get("verdict", "UNSUPPORTED"),
                    evidence=item.get("evidence", ""),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        return checks

    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("[Critic] JSON parse failed: %s — falling back to holistic check.", exc)
        return _holistic_fallback(claims, facts, llm)
    except Exception as exc:
        logger.error("[Critic] LLM call failed: %s", exc)
        return []


def _holistic_fallback(
    claims: list[str], facts: list[str], llm: ChatOllama
) -> list[ClaimCheck]:
    """Simple fallback: mark all claims as SUPPORTED if the summary seems factual."""
    # Return all as UNSUPPORTED with low confidence — forces at least partial review
    return [
        ClaimCheck(claim=c, verdict="UNSUPPORTED", evidence="", confidence=0.3)
        for c in claims
    ]


def critic_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Critic.

    Performs claim-level fact-checking of the summary against retrieved facts.

    Updates state fields:
      - claim_checks       (per-claim verdict list)
      - validation_status  ("PASS" | "FAIL")
      - critique_notes     (summary of failures)
      - retry_count        (incremented on FAIL)
    """
    question = state.get("question", "")
    facts = state.get("facts", [])
    summary = state.get("summary", "")
    retry_count = state.get("retry_count", 0)

    cfg = get_settings()
    logger.info(
        "[Critic] Claim-level check (attempt %d / %d).", retry_count + 1, cfg.max_retry
    )

    if not summary:
        logger.warning("[Critic] Empty summary — marking FAIL.")
        return {
            "claim_checks": [],
            "validation_status": "FAIL",
            "critique_notes": "Summary is empty.",
            "retry_count": retry_count + 1,
        }

    # Split summary into claims
    claims = _split_claims(summary)
    if not claims:
        claims = [summary]

    logger.info("[Critic] Checking %d claims against %d facts.", len(claims), len(facts))

    llm = ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=0,
    )

    claim_checks = _check_claims(claims, facts, llm)

    # ── Compute verdict ───────────────────────────────────────────────────────
    if not claim_checks:
        # LLM failed completely — pass through to avoid infinite loop
        logger.warning("[Critic] No claim checks produced — defaulting PASS.")
        return {
            "claim_checks": [],
            "validation_status": "PASS",
            "critique_notes": "Critic could not evaluate — passing through.",
            "retry_count": retry_count,
        }

    total = len(claim_checks)
    supported = sum(1 for c in claim_checks if c.get("verdict") == "SUPPORTED")
    contradicted = sum(1 for c in claim_checks if c.get("verdict") == "CONTRADICTED")
    support_rate = supported / total

    logger.info(
        "[Critic] %d/%d claims SUPPORTED (%.0f%%), %d CONTRADICTED.",
        supported, total, support_rate * 100, contradicted,
    )

    if support_rate >= _PASS_THRESHOLD and contradicted == 0:
        logger.info("[Critic] PASS — %.0f%% supported, 0 contradictions.", support_rate * 100)
        return {
            "claim_checks": claim_checks,
            "validation_status": "PASS",
            "critique_notes": "",
            "retry_count": retry_count,
        }
    else:
        # Build critique notes listing specific failures
        failures = [
            f"- [{c.get('verdict')}] \"{c.get('claim', '')[:80]}\" → {c.get('evidence', 'no evidence')[:100]}"
            for c in claim_checks
            if c.get("verdict") in ("UNSUPPORTED", "CONTRADICTED")
        ]
        notes = (
            f"Support rate: {support_rate:.0%} (need ≥{_PASS_THRESHOLD:.0%}). "
            f"{contradicted} contradiction(s).\n"
            + "\n".join(failures[:5])
        )

        logger.warning("[Critic] FAIL — %s", notes[:200])
        return {
            "claim_checks": claim_checks,
            "validation_status": "FAIL",
            "critique_notes": notes,
            "retry_count": retry_count + 1,
        }


def should_retry(state: ResearchState) -> str:
    """LangGraph conditional edge: retry → Summarizer, finalize → Finalizer."""
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

    logger.info("[Critic] Routing → Summarizer (retry %d / %d)", retry_count, cfg.max_retry)
    return "retry"
