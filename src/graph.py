"""
LangGraph Pipeline — wires all agents into a stateful graph with a
self-correcting critique loop.

Graph topology:
  START
    └─► searcher
          └─► retriever
                └─► summarizer
                      └─► critic ──┐
                           │       │ FAIL (retry)
                           │       └──► summarizer (loop back)
                           │
                           └─► finalizer (PASS / max retries)
                                 └─► END
"""

import logging

from langgraph.graph import StateGraph, END

from src.state import ResearchState, initial_state
from src.agents.searcher import searcher_node
from src.agents.retriever import retriever_node
from src.agents.summarizer import summarizer_node
from src.agents.critic import critic_node, should_retry
from src.agents.finalizer import finalizer_node

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph research pipeline."""

    graph = StateGraph(ResearchState)

    # ── Add nodes ────────────────────────────────────────────────────────────
    graph.add_node("searcher", searcher_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalizer", finalizer_node)

    # ── Linear edges ─────────────────────────────────────────────────────────
    graph.set_entry_point("searcher")
    graph.add_edge("searcher", "retriever")
    graph.add_edge("retriever", "summarizer")
    graph.add_edge("summarizer", "critic")

    # ── Conditional critique loop ─────────────────────────────────────────────
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry": "summarizer",    # loop back
            "finalize": "finalizer",  # proceed
        },
    )

    graph.add_edge("finalizer", END)

    return graph.compile()


# Module-level compiled graph (cached)
_compiled_graph = None


def get_graph():
    """Return the cached compiled graph, building it on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Building LangGraph research pipeline...")
        _compiled_graph = build_graph()
        logger.info("Pipeline compiled successfully.")
    return _compiled_graph


def run_research(question: str) -> ResearchState:
    """
    Execute the full multi-agent research pipeline for a given question.

    Args:
        question: The user's research question.

    Returns:
        The final ResearchState after all agents have run.
    """
    graph = get_graph()
    state = initial_state(question)
    logger.info("Starting research pipeline for: %s", question)
    result = graph.invoke(state)
    logger.info("Research pipeline completed. Session ID: %s", result.get("session_id"))
    return result


def stream_research(question: str):
    """
    Stream intermediate state updates from the research pipeline.

    Yields:
        Tuples of (node_name, partial_state_dict) as each agent completes.
    """
    graph = get_graph()
    state = initial_state(question)

    for chunk in graph.stream(state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            logger.debug("Node '%s' produced update.", node_name)
            yield node_name, node_output
