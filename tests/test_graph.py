"""Tests for the multi-agent research pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from unittest.mock import patch, MagicMock

from src.state import initial_state, ResearchState
from src.agents.searcher import searcher_node
from src.agents.retriever import retriever_node
from src.agents.summarizer import summarizer_node
from src.agents.critic import critic_node, should_retry
from src.agents.finalizer import finalizer_node


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> ResearchState:
    return initial_state("How is AI used in healthcare?")


@pytest.fixture
def state_with_raw_data(base_state) -> ResearchState:
    state = dict(base_state)
    state["raw_data"] = [
        {"content": "AI is used in medical imaging.", "source": "https://example.com/1", "type": "web"},
        {"content": "Machine learning detects cancer in scans.", "source": "https://example.com/2", "type": "web"},
        {"content": "NLP extracts info from medical records.", "source": "local_doc", "type": "local_doc"},
    ]
    state["sources"] = ["https://example.com/1", "https://example.com/2", "local_doc"]
    return state


@pytest.fixture
def state_with_facts(state_with_raw_data) -> ResearchState:
    state = dict(state_with_raw_data)
    state["facts"] = [
        "AI is used in medical imaging to detect diseases.",
        "Machine learning helps identify cancer in CT scans.",
        "NLP is used to extract data from medical records.",
    ]
    return state


@pytest.fixture
def state_with_summary(state_with_facts) -> ResearchState:
    state = dict(state_with_facts)
    state["summary"] = (
        "AI plays a transformative role in healthcare. It is used in medical imaging "
        "to detect diseases, helps identify cancer in CT scans via machine learning, "
        "and extracts structured data from records using NLP."
    )
    return state


# ── State Tests ───────────────────────────────────────────────────────────────

class TestInitialState:
    def test_creates_valid_state(self, base_state):
        assert base_state["question"] == "How is AI used in healthcare?"
        assert base_state["raw_data"] == []
        assert base_state["facts"] == []
        assert base_state["validation_status"] == "PENDING"
        assert base_state["retry_count"] == 0


# ── Searcher Tests ────────────────────────────────────────────────────────────

class TestSearcherNode:
    @patch("src.agents.searcher.web_search")
    @patch("src.agents.searcher.local_doc_search")
    def test_merges_results(self, mock_local, mock_web, base_state):
        mock_web.invoke.return_value = [
            {"content": "Web result 1", "source": "https://a.com", "type": "web"}
        ]
        mock_local.invoke.return_value = [
            {"content": "Local doc result", "source": "doc.txt", "type": "local_doc"}
        ]

        result = searcher_node(base_state)

        assert len(result["raw_data"]) == 2
        assert len(result["sources"]) == 2

    @patch("src.agents.searcher.web_search")
    @patch("src.agents.searcher.local_doc_search")
    def test_handles_empty_results(self, mock_local, mock_web, base_state):
        mock_web.invoke.return_value = []
        mock_local.invoke.return_value = []

        result = searcher_node(base_state)

        assert result["raw_data"] == []
        assert result["sources"] == []

    @patch("src.agents.searcher.web_search")
    @patch("src.agents.searcher.local_doc_search")
    def test_handles_search_failure(self, mock_local, mock_web, base_state):
        mock_web.invoke.side_effect = Exception("API error")
        mock_local.invoke.return_value = []

        result = searcher_node(base_state)
        # Should not raise, should return empty
        assert result["raw_data"] == []


# ── Critic / Router Tests ─────────────────────────────────────────────────────

class TestShouldRetry:
    def test_routes_to_finalize_on_pass(self, state_with_summary):
        state = dict(state_with_summary)
        state["validation_status"] = "PASS"
        assert should_retry(state) == "finalize"

    def test_routes_to_retry_on_fail(self, state_with_summary):
        state = dict(state_with_summary)
        state["validation_status"] = "FAIL"
        state["retry_count"] = 1
        assert should_retry(state) == "retry"

    def test_forces_finalize_at_max_retries(self, state_with_summary):
        state = dict(state_with_summary)
        state["validation_status"] = "FAIL"
        state["retry_count"] = 3  # = MAX_RETRY default
        assert should_retry(state) == "finalize"


# ── Critic Node Tests ─────────────────────────────────────────────────────────

class TestCriticNode:
    @patch("src.agents.critic.ChatOllama")
    def test_returns_pass_verdict(self, mock_ollama_class, state_with_summary):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="VERDICT: PASS")
        mock_ollama_class.return_value = mock_llm

        result = critic_node(state_with_summary)
        assert result["validation_status"] == "PASS"

    @patch("src.agents.critic.ChatOllama")
    def test_returns_fail_verdict(self, mock_ollama_class, state_with_summary):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="VERDICT: FAIL\nISSUES:\n- Hallucinated statistic"
        )
        mock_ollama_class.return_value = mock_llm

        result = critic_node(state_with_summary)
        assert result["validation_status"] == "FAIL"
        assert "Hallucinated" in result["critique_notes"]
        assert result["retry_count"] == 1

    def test_fails_on_empty_summary(self, base_state):
        result = critic_node(base_state)
        assert result["validation_status"] == "FAIL"


# ── Finalizer Tests ───────────────────────────────────────────────────────────

class TestFinalizerNode:
    @patch("src.agents.finalizer.ChatOllama")
    @patch("src.agents.finalizer.save_session")
    def test_produces_answer_and_saves_session(
        self, mock_save, mock_ollama_class, state_with_summary
    ):
        state = dict(state_with_summary)
        state["validation_status"] = "PASS"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="AI is transforming healthcare in multiple ways [1].\n\n## Sources\n[1] https://example.com/1"
        )
        mock_ollama_class.return_value = mock_llm
        mock_save.return_value = "test-session-uuid"

        result = finalizer_node(state)

        assert "AI is transforming" in result["final_answer"]
        assert result["session_id"] == "test-session-uuid"
        mock_save.assert_called_once()
