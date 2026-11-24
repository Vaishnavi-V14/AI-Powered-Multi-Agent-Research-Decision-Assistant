"""
LangGraph Pipeline — Full production pipeline with:
  Planner → Searcher (parallel) → SourceVerifier → Retriever (hybrid+rerank)
  → Summarizer → Critic (claim-level) → [retry|finalize] → Finalizer → Evaluator → END
"""

import logging

from langgraph.graph import StateGraph, END

from src.state import ResearchState, initial_state
from src.agents.planner import planner_node
from src.agents.searcher import searcher_node
from src.agents.source_verifier import source_verifier_node
from src.agents.retriever import retriever_node
from src.agents.summarizer import summarizer_node
from src.agents.critic import critic_node, should_retry
from src.agents.finalizer import finalizer_node
from src.evaluation import evaluator_node

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Construct and compile the full LangGraph research pipeline."""

    graph = StateGraph(ResearchState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    graph.add_node("planner",         planner_node)
    graph.add_node("searcher",        searcher_node)
    graph.add_node("source_verifier", source_verifier_node)
    graph.add_node("retriever",       retriever_node)
    graph.add_node("summarizer",      summarizer_node)
    graph.add_node("critic",          critic_node)
    graph.add_node("finalizer",       finalizer_node)
    graph.add_node("evaluator",       evaluator_node)

    # ── Edges — linear pipeline ───────────────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner",         "searcher")
    graph.add_edge("searcher",        "source_verifier")
    graph.add_edge("source_verifier", "retriever")
    graph.add_edge("retriever",       "summarizer")
    graph.add_edge("summarizer",      "critic")

    # ── Conditional critique loop ─────────────────────────────────────────────
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry":    "summarizer",   # loop back for self-correction
            "finalize": "finalizer",    # proceed to synthesis
        },
    )

    graph.add_edge("finalizer", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Return the cached compiled graph, building on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Building LangGraph research pipeline...")
        _compiled_graph = build_graph()
        logger.info("Pipeline compiled successfully.")
    return _compiled_graph


def run_research(question: str) -> ResearchState:
    """Execute the full pipeline synchronously."""
    graph = get_graph()
    state = initial_state(question)
    logger.info("Starting research pipeline for: %s", question)
    result = graph.invoke(state)
    logger.info(
        "Pipeline complete. Session: %s | Score: %.2f",
        result.get("session_id"),
        result.get("eval_scores", {}).get("overall_score", 0),
    )
    return result


def stream_research(question: str):
    """
    Stream intermediate state updates as each agent completes.

    Yields:
        Tuples of (node_name, partial_state_dict).
    """
    graph = get_graph()
    state = initial_state(question)

    for chunk in graph.stream(state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            logger.debug("Node '%s' completed.", node_name)
            yield node_name, node_output
