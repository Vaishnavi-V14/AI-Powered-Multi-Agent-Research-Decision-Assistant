"""
LangGraph shared state schema — upgraded for production multi-agent pipeline.

State progression:
  question       → user input
  sub_queries    → Planner output (3-5 targeted search queries)
  raw_data       → Searcher output (parallel sub-query results)
  verified_sources → SourceVerifier output (URL checks + claim cross-refs)
  ranked_chunks  → Retriever output (hybrid BM25+vector, reranked)
  facts          → Retriever extracted relevant facts
  summary        → Summarizer output (compressed key points)
  claim_checks   → Critic output (per-claim verification results)
  validation_status → Critic verdict ("PASS" | "FAIL")
  critique_notes → Critic failure reasons
  retry_count    → incremented by Critic on FAIL
  sources        → deduplicated source list
  final_answer   → Finalizer synthesized answer
  eval_scores    → Evaluator RAGAS-style metric scores
  session_id     → SQLite record UUID
  error          → unrecoverable exception message
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


class SourceVerification(TypedDict, total=False):
    """Result of verifying a single source."""
    url: str
    accessible: bool
    status_code: int
    claims_supported: int    # number of facts this source supports
    claims_total: int
    confidence: float        # 0.0 – 1.0


class ClaimCheck(TypedDict, total=False):
    """Per-claim fact-check result from Critic."""
    claim: str
    verdict: str             # "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED"
    evidence: str            # supporting or contradicting snippet
    confidence: float        # 0.0 – 1.0


class EvalScores(TypedDict, total=False):
    """RAGAS-style evaluation metrics."""
    faithfulness: float          # % answer claims grounded in context
    answer_relevancy: float      # semantic sim(question, answer)
    context_precision: float     # % retrieved chunks that are relevant
    context_recall: float        # % relevant facts captured
    source_coverage: float       # % sources accessible
    overall_score: float         # weighted average


class ResearchState(TypedDict, total=False):
    """Full shared state for the multi-agent research pipeline."""

    # Input
    question: str

    # Planner output
    sub_queries: list[str]

    # Searcher output (append-reducible across parallel runs)
    raw_data: Annotated[list[dict[str, Any]], operator.add]

    # Source Verifier output
    verified_sources: list[SourceVerification]

    # Retriever output
    ranked_chunks: list[dict[str, Any]]   # reranked, deduplicated
    facts: list[str]                       # extracted factual sentences

    # Summarizer output
    summary: str

    # Critic output
    claim_checks: list[ClaimCheck]
    validation_status: str               # "PASS" | "FAIL" | "PENDING"
    critique_notes: str
    retry_count: int

    # Sources
    sources: list[str]                   # deduplicated URL/doc names

    # Finalizer output
    final_answer: str

    # Evaluator output
    eval_scores: EvalScores

    # Persistence
    session_id: str

    # Error tracking
    error: str


def initial_state(question: str) -> ResearchState:
    """Create a fresh state for a new research session."""
    return ResearchState(
        question=question,
        sub_queries=[],
        raw_data=[],
        verified_sources=[],
        ranked_chunks=[],
        facts=[],
        summary="",
        claim_checks=[],
        validation_status="PENDING",
        critique_notes="",
        retry_count=0,
        sources=[],
        final_answer="",
        eval_scores={},
        session_id="",
        error="",
    )
