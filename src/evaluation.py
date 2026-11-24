"""
RAGAS-Style Evaluation Pipeline
=================================
Implements four measurable evaluation metrics for each research session.
These run entirely locally (no external API) using Ollama as the LLM judge.

Metrics:
  1. Faithfulness        — What % of answer claims are grounded in the retrieved context?
  2. Answer Relevancy    — How well does the answer address the original question?
  3. Context Precision   — What % of retrieved chunks actually helped answer the question?
  4. Context Recall      — Did the retrieved context contain all necessary information?
  5. Source Coverage     — What % of cited sources are accessible/verified?

Each metric is a float in [0.0, 1.0]. An overall_score is their weighted average.

Weights:
  faithfulness       0.35  (most critical — prevents hallucination)
  answer_relevancy   0.25
  context_precision  0.20
  context_recall     0.15
  source_coverage    0.05
"""

import json
import logging
import re
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np

from src.config import get_settings
from src.state import EvalScores, ResearchState

logger = logging.getLogger(__name__)

_WEIGHTS = {
    "faithfulness": 0.35,
    "answer_relevancy": 0.25,
    "context_precision": 0.20,
    "context_recall": 0.15,
    "source_coverage": 0.05,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_llm(cfg, temperature: float = 0):
    return ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=temperature,
    )


def _ask_score(llm: ChatOllama, system: str, human: str) -> float:
    """Ask LLM for a numeric score 0.0–1.0, with robust extraction."""
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = response.content.strip()
        # Extract first float-like number in [0,1]
        matches = re.findall(r'\b(0?\.\d+|1\.0|0|1)\b', raw)
        for m in matches:
            val = float(m)
            if 0.0 <= val <= 1.0:
                return round(val, 3)
        # Fallback: look for integer (convert 0-10 scale)
        int_matches = re.findall(r'\b([0-9]|10)\b', raw)
        if int_matches:
            return round(int(int_matches[0]) / 10, 3)
    except Exception as exc:
        logger.warning("[Eval] LLM score extraction failed: %s", exc)
    return 0.5  # neutral fallback


# ── Metric Implementations ────────────────────────────────────────────────────

_FAITHFULNESS_SYSTEM = """\
You are evaluating whether an AI answer is faithful to its source context.

Task: For each sentence/claim in the ANSWER, check if it is supported by the CONTEXT.
Count how many claims are supported vs total claims.

Output ONLY a JSON object:
{"supported_claims": <int>, "total_claims": <int>, "faithfulness_score": <float 0-1>}
"""

def _faithfulness(answer: str, context_chunks: list[str], llm: ChatOllama) -> float:
    """Fraction of answer claims grounded in retrieved context."""
    if not answer or not context_chunks:
        return 0.0

    context_text = "\n---\n".join(context_chunks[:8])
    human = f"ANSWER:\n{answer}\n\nCONTEXT:\n{context_text}"

    try:
        resp = llm.invoke([SystemMessage(content=_FAITHFULNESS_SYSTEM), HumanMessage(content=human)])
        data = json.loads(resp.content.strip())
        total = max(data.get("total_claims", 1), 1)
        supported = data.get("supported_claims", 0)
        return round(min(supported / total, 1.0), 3)
    except Exception:
        return _ask_score(llm, _FAITHFULNESS_SYSTEM, human)


_RELEVANCY_SYSTEM = """\
Score how well the ANSWER addresses the QUESTION.
- 1.0: Answer directly and completely addresses the question
- 0.7: Answer mostly relevant but misses some aspects
- 0.4: Answer is tangentially related
- 0.0: Answer does not address the question at all

Output ONLY a single float between 0.0 and 1.0.
"""

def _answer_relevancy(question: str, answer: str, llm: ChatOllama) -> float:
    """Semantic relevance of answer to the original question."""
    if not answer or not question:
        return 0.0
    return _ask_score(
        llm,
        _RELEVANCY_SYSTEM,
        f"QUESTION: {question}\n\nANSWER: {answer}"
    )


_PRECISION_SYSTEM = """\
You are evaluating retrieval quality. For each CONTEXT CHUNK, judge if it
was useful for answering the QUESTION (relevant=1) or not (relevant=0).

Output ONLY a JSON array, one entry per chunk:
[{"chunk_index": 0, "relevant": 1}, {"chunk_index": 1, "relevant": 0}, ...]
"""

def _context_precision(
    question: str, chunks: list[str], llm: ChatOllama
) -> float:
    """Fraction of retrieved chunks that are actually relevant to the question."""
    if not chunks:
        return 0.0

    chunks_text = "\n".join(
        f"[Chunk {i}]: {c[:300]}" for i, c in enumerate(chunks[:10])
    )
    human = f"QUESTION: {question}\n\nCHUNKS:\n{chunks_text}"

    try:
        resp = llm.invoke([SystemMessage(content=_PRECISION_SYSTEM), HumanMessage(content=human)])
        raw = resp.content.strip()
        # Extract JSON array
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if data:
                relevant_count = sum(1 for d in data if d.get("relevant") == 1)
                return round(relevant_count / len(data), 3)
    except Exception:
        pass
    return _ask_score(llm, _PRECISION_SYSTEM, human)


_RECALL_SYSTEM = """\
You are evaluating whether the retrieved CONTEXT contains all information
needed to answer the QUESTION comprehensively.

- 1.0: Context fully contains everything needed
- 0.7: Context contains most needed information
- 0.4: Context is missing important aspects
- 0.0: Context is completely insufficient

Output ONLY a single float between 0.0 and 1.0.
"""

def _context_recall(question: str, chunks: list[str], llm: ChatOllama) -> float:
    """Whether the retrieved context covers all needed information."""
    if not chunks:
        return 0.0
    context_text = "\n---\n".join(chunks[:8])
    return _ask_score(
        llm,
        _RECALL_SYSTEM,
        f"QUESTION: {question}\n\nCONTEXT:\n{context_text}"
    )


def _source_coverage(verified_sources: list[dict]) -> float:
    """Fraction of cited sources that are HTTP-accessible."""
    if not verified_sources:
        return 0.0
    web_sources = [s for s in verified_sources if str(s.get("url", "")).startswith("http")]
    if not web_sources:
        return 1.0  # all local — full coverage by definition
    accessible = sum(1 for s in web_sources if s.get("accessible", False))
    return round(accessible / len(web_sources), 3)


# ── Main Evaluator Node ───────────────────────────────────────────────────────

def evaluator_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Evaluator.

    Computes RAGAS-style metrics for the completed research session.

    Updates state fields:
      - eval_scores (EvalScores dict with all metric values)
    """
    question = state.get("question", "")
    final_answer = state.get("final_answer", "")
    ranked_chunks = state.get("ranked_chunks", [])
    facts = state.get("facts", [])
    verified_sources = state.get("verified_sources", [])

    logger.info("[Evaluator] Computing evaluation metrics...")

    cfg = get_settings()
    llm = _safe_llm(cfg)

    # Build context strings from ranked chunks
    context_strings = [c.get("content", "") for c in ranked_chunks if c.get("content")]
    if not context_strings:
        context_strings = facts  # fallback to facts

    # ── Compute all metrics ───────────────────────────────────────────────────
    scores: dict[str, float] = {}

    try:
        scores["faithfulness"] = _faithfulness(final_answer, context_strings, llm)
        logger.info("[Evaluator] Faithfulness: %.3f", scores["faithfulness"])
    except Exception as exc:
        logger.error("[Evaluator] Faithfulness failed: %s", exc)
        scores["faithfulness"] = 0.0

    try:
        scores["answer_relevancy"] = _answer_relevancy(question, final_answer, llm)
        logger.info("[Evaluator] Answer Relevancy: %.3f", scores["answer_relevancy"])
    except Exception as exc:
        logger.error("[Evaluator] Answer Relevancy failed: %s", exc)
        scores["answer_relevancy"] = 0.0

    try:
        scores["context_precision"] = _context_precision(question, context_strings, llm)
        logger.info("[Evaluator] Context Precision: %.3f", scores["context_precision"])
    except Exception as exc:
        logger.error("[Evaluator] Context Precision failed: %s", exc)
        scores["context_precision"] = 0.0

    try:
        scores["context_recall"] = _context_recall(question, context_strings, llm)
        logger.info("[Evaluator] Context Recall: %.3f", scores["context_recall"])
    except Exception as exc:
        logger.error("[Evaluator] Context Recall failed: %s", exc)
        scores["context_recall"] = 0.0

    try:
        scores["source_coverage"] = _source_coverage(verified_sources)
        logger.info("[Evaluator] Source Coverage: %.3f", scores["source_coverage"])
    except Exception as exc:
        logger.error("[Evaluator] Source Coverage failed: %s", exc)
        scores["source_coverage"] = 0.0

    # Weighted overall score
    overall = sum(
        scores.get(metric, 0.0) * weight
        for metric, weight in _WEIGHTS.items()
    )
    scores["overall_score"] = round(overall, 3)

    logger.info(
        "[Evaluator] Overall score: %.3f | F=%.2f AR=%.2f CP=%.2f CR=%.2f SC=%.2f",
        scores["overall_score"],
        scores.get("faithfulness", 0),
        scores.get("answer_relevancy", 0),
        scores.get("context_precision", 0),
        scores.get("context_recall", 0),
        scores.get("source_coverage", 0),
    )

    return {"eval_scores": EvalScores(**scores)}
