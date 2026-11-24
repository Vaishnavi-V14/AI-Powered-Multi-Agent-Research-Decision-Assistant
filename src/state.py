"""LangGraph shared state schema for the multi-agent research pipeline."""

from typing import Annotated, Any
from typing_extensions import TypedDict

import operator


class ResearchState(TypedDict, total=False):
    """
    Shared mutable state passed between all agents in the LangGraph pipeline.

    Fields are progressively populated as the pipeline executes:
      question        → Set by the user at entry point
      raw_data        → Set by Searcher (web + doc results)
      facts           → Set by Retriever (filtered relevant facts)
      summary         → Set by Summarizer (compressed key points)
      validation_status → Set by Critic ("PASS" | "FAIL")
      critique_notes  → Set by Critic (reasons for failure, if any)
      retry_count     → Incremented by Critic on each retry
      sources         → Set by Retriever (list of source URLs / doc names)
      final_answer    → Set by Finalizer (polished, cited answer)
      session_id      → Set by Finalizer (UUID stored in SQLite)
      error           → Set on any unrecoverable exception
    """

    question: str
    raw_data: Annotated[list[dict[str, Any]], operator.add]
    facts: list[str]
    summary: str
    validation_status: str          # "PASS" | "FAIL" | "PENDING"
    critique_notes: str
    retry_count: int
    sources: list[str]
    final_answer: str
    session_id: str
    error: str


def initial_state(question: str) -> ResearchState:
    """Create a fresh state for a new research session."""
    return ResearchState(
        question=question,
        raw_data=[],
        facts=[],
        summary="",
        validation_status="PENDING",
        critique_notes="",
        retry_count=0,
        sources=[],
        final_answer="",
        session_id="",
        error="",
    )
