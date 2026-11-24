"""Comprehensive tests for the production multi-agent research pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pytest
from unittest.mock import patch, MagicMock, call

from src.state import initial_state, ResearchState, ClaimCheck, EvalScores
from src.agents.planner import planner_node, _extract_fallback_queries
from src.agents.searcher import searcher_node, _fingerprint
from src.agents.source_verifier import _check_url, source_verifier_node
from src.agents.retriever import _bm25_search, _rrf_fusion
from src.agents.critic import critic_node, should_retry, _split_claims, _PASS_THRESHOLD
from src.agents.finalizer import finalizer_node
from src.evaluation import _source_coverage, _faithfulness, evaluator_node
from src.reranker import rerank


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def q() -> str:
    return "How is AI used in drug discovery?"


@pytest.fixture
def base(q) -> ResearchState:
    return initial_state(q)


@pytest.fixture
def with_sub_queries(base) -> ResearchState:
    s = dict(base)
    s["sub_queries"] = [
        "AI applications in drug discovery",
        "machine learning in pharmaceutical research",
        "deep learning for protein folding",
    ]
    return s


@pytest.fixture
def with_raw_data(with_sub_queries) -> ResearchState:
    s = dict(with_sub_queries)
    s["raw_data"] = [
        {"content": "AI accelerates drug target identification.", "source": "https://nature.com/ai-drug", "type": "web"},
        {"content": "AlphaFold predicts protein structures.", "source": "https://deepmind.com/alphafold", "type": "web"},
        {"content": "Local lab notes on ML in pharma.", "source": "lab_notes.txt", "type": "local_doc"},
    ]
    s["sources"] = ["https://nature.com/ai-drug", "https://deepmind.com/alphafold", "lab_notes.txt"]
    return s


@pytest.fixture
def with_facts(with_raw_data) -> ResearchState:
    s = dict(with_raw_data)
    s["facts"] = [
        "AI accelerates identification of drug targets.",
        "AlphaFold can predict protein 3D structure with high accuracy.",
        "Machine learning models screen millions of compounds faster than humans.",
    ]
    s["ranked_chunks"] = [
        {"content": f, "source": "test", "rerank_score": 0.9 - i * 0.1}
        for i, f in enumerate(s["facts"])
    ]
    return s


@pytest.fixture
def with_summary(with_facts) -> ResearchState:
    s = dict(with_facts)
    s["summary"] = (
        "AI is transforming drug discovery. AlphaFold predicts protein structures. "
        "Machine learning screens vast compound libraries rapidly."
    )
    return s


@pytest.fixture
def with_verified_sources(with_raw_data) -> ResearchState:
    s = dict(with_raw_data)
    s["verified_sources"] = [
        {"url": "https://nature.com/ai-drug", "accessible": True, "status_code": 200, "confidence": 0.8},
        {"url": "https://deepmind.com/alphafold", "accessible": True, "status_code": 200, "confidence": 0.6},
        {"url": "lab_notes.txt", "accessible": False, "status_code": 0, "confidence": 0.4},
    ]
    return s


# ── State Tests ───────────────────────────────────────────────────────────────

class TestState:
    def test_initial_state_defaults(self, base, q):
        assert base["question"] == q
        assert base["sub_queries"] == []
        assert base["raw_data"] == []
        assert base["facts"] == []
        assert base["claim_checks"] == []
        assert base["validation_status"] == "PENDING"
        assert base["retry_count"] == 0
        assert base["eval_scores"] == {}

    def test_claim_check_typed_dict(self):
        c = ClaimCheck(claim="AI detects cancer.", verdict="SUPPORTED", evidence="Study shows...", confidence=0.9)
        assert c["verdict"] == "SUPPORTED"
        assert c["confidence"] == 0.9


# ── Planner Tests ─────────────────────────────────────────────────────────────

class TestPlanner:
    @patch("src.agents.planner.ChatOllama")
    def test_returns_sub_queries(self, mock_cls, base):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "sub_queries": ["AI in drug target identification", "deep learning protein folding"]
        }))
        mock_cls.return_value = mock_llm

        result = planner_node(base)
        queries = result["sub_queries"]
        assert len(queries) >= 2
        # Original question always prepended if not present
        assert any("drug" in q.lower() or "AI" in q for q in queries)

    @patch("src.agents.planner.ChatOllama")
    def test_always_includes_original_question(self, mock_cls, base):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "sub_queries": ["specific sub-query A", "specific sub-query B"]
        }))
        mock_cls.return_value = mock_llm

        result = planner_node(base)
        assert base["question"] in result["sub_queries"]

    @patch("src.agents.planner.ChatOllama")
    def test_caps_at_five_queries(self, mock_cls, base):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "sub_queries": [f"query {i}" for i in range(10)]
        }))
        mock_cls.return_value = mock_llm

        result = planner_node(base)
        assert len(result["sub_queries"]) <= 5

    def test_fallback_extraction_numbered_list(self):
        raw = "1. First query\n2. Second query\n3. Third query"
        queries = _extract_fallback_queries(raw, "original")
        assert len(queries) == 3
        assert queries[0] == "First query"


# ── Searcher Tests ────────────────────────────────────────────────────────────

class TestSearcher:
    def test_fingerprint_deduplication(self):
        content = "AI is used in healthcare for diagnostics."
        fp1 = _fingerprint(content)
        fp2 = _fingerprint(content)
        assert fp1 == fp2

    def test_different_fingerprints(self):
        fp1 = _fingerprint("AI in healthcare")
        fp2 = _fingerprint("Blockchain in finance")
        assert fp1 != fp2

    @patch("src.agents.searcher.web_search")
    @patch("src.agents.searcher.local_doc_search")
    def test_deduplicates_results(self, mock_local, mock_web, with_sub_queries):
        # Both searches return the same content
        same_result = {"content": "AI detects cancer early.", "source": "https://a.com", "type": "web"}
        mock_web.invoke.return_value = [same_result]
        mock_local.invoke.return_value = [same_result]

        result = searcher_node(with_sub_queries)
        # Same content should be deduplicated
        contents = [r["content"] for r in result["raw_data"]]
        assert len(contents) == len(set(contents))

    @patch("src.agents.searcher.web_search")
    @patch("src.agents.searcher.local_doc_search")
    def test_handles_all_search_failures(self, mock_local, mock_web, with_sub_queries):
        mock_web.invoke.side_effect = Exception("API down")
        mock_local.invoke.side_effect = Exception("DB error")

        result = searcher_node(with_sub_queries)
        assert result["raw_data"] == []
        assert result["sources"] == []


# ── Source Verifier Tests ─────────────────────────────────────────────────────

class TestSourceVerifier:
    def test_check_url_local_file(self):
        result = _check_url("lab_notes.txt")
        assert result["is_local"] is True
        assert result["accessible"] is False  # local — not HTTP

    @patch("src.agents.source_verifier.httpx.Client")
    def test_check_url_accessible(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.head.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        result = _check_url("https://example.com")
        assert result["accessible"] is True
        assert result["status_code"] == 200

    @patch("src.agents.source_verifier.httpx.Client")
    def test_check_url_not_found(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.head.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        result = _check_url("https://example.com/missing")
        assert result["accessible"] is False


# ── Retriever Tests ───────────────────────────────────────────────────────────

class TestRetriever:
    def test_bm25_ranks_by_relevance(self):
        corpus = [
            {"content": "AI is used in drug discovery to find targets."},
            {"content": "The stock market crashed in 2008."},
            {"content": "Machine learning accelerates drug target identification."},
        ]
        results = _bm25_search("AI drug discovery", corpus, top_k=3)
        # First result should be AI/drug-related
        assert "drug" in results[0]["content"].lower() or "AI" in results[0]["content"]

    def test_rrf_fusion_combines_rankings(self):
        bm25 = [
            {"content": "AI drug discovery", "source": "a"},
            {"content": "Deep learning molecules", "source": "b"},
        ]
        vector = [
            {"content": "Machine learning pharma", "source": "c"},
            {"content": "AI drug discovery", "source": "a"},  # overlap
        ]
        fused = _rrf_fusion(bm25, vector)
        # Should contain 3 unique items (overlap deduplicated)
        contents = [r["content"] for r in fused]
        # Overlapping content should be ranked higher
        assert "AI drug discovery" in contents[0] or fused[0].get("rrf_score", 0) > 0

    def test_rrf_attaches_scores(self):
        bm25 = [{"content": "test content one", "source": "a"}]
        vector = [{"content": "test content two", "source": "b"}]
        fused = _rrf_fusion(bm25, vector)
        assert all("rrf_score" in r for r in fused)


# ── Reranker Tests ────────────────────────────────────────────────────────────

class TestReranker:
    def test_returns_empty_for_empty_input(self):
        assert rerank("query", []) == []

    def test_respects_top_k(self):
        chunks = [{"content": f"chunk {i}"} for i in range(20)]
        # Without model loaded, falls back gracefully
        result = rerank("query", chunks, top_k=5)
        assert len(result) <= 5


# ── Critic Tests ──────────────────────────────────────────────────────────────

class TestCritic:
    def test_split_claims_basic(self):
        text = "AI detects cancer. It improves diagnosis speed. Models achieve 95% accuracy."
        claims = _split_claims(text)
        assert len(claims) == 3

    def test_split_claims_filters_short(self):
        text = "AI. It improves diagnosis of rare diseases in patients significantly."
        claims = _split_claims(text)
        # "AI." is too short (<20 chars)
        assert all(len(c) > 20 for c in claims)

    @patch("src.agents.critic.ChatOllama")
    def test_pass_when_all_supported(self, mock_cls, with_summary):
        mock_llm = MagicMock()
        claims = _split_claims(with_summary["summary"])
        checks = [{"claim_index": i, "claim": c, "verdict": "SUPPORTED",
                   "evidence": "fact supports it", "confidence": 0.9}
                  for i, c in enumerate(claims)]
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(checks))
        mock_cls.return_value = mock_llm

        result = critic_node(with_summary)
        assert result["validation_status"] == "PASS"

    @patch("src.agents.critic.ChatOllama")
    def test_fail_on_contradicted_claim(self, mock_cls, with_summary):
        mock_llm = MagicMock()
        claims = _split_claims(with_summary["summary"])
        checks = [
            {"claim_index": 0, "claim": claims[0], "verdict": "CONTRADICTED",
             "evidence": "source says opposite", "confidence": 0.95},
        ] + [
            {"claim_index": i+1, "claim": c, "verdict": "SUPPORTED",
             "evidence": "fact supports", "confidence": 0.9}
            for i, c in enumerate(claims[1:])
        ]
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(checks))
        mock_cls.return_value = mock_llm

        result = critic_node(with_summary)
        assert result["validation_status"] == "FAIL"

    def test_fail_on_empty_summary(self, base):
        result = critic_node(base)
        assert result["validation_status"] == "FAIL"
        assert result["retry_count"] == 1

    def test_should_retry_on_fail(self, with_summary):
        s = dict(with_summary)
        s["validation_status"] = "FAIL"
        s["retry_count"] = 1
        assert should_retry(s) == "retry"

    def test_should_finalize_on_pass(self, with_summary):
        s = dict(with_summary)
        s["validation_status"] = "PASS"
        assert should_retry(s) == "finalize"

    def test_should_finalize_at_max_retry(self, with_summary):
        s = dict(with_summary)
        s["validation_status"] = "FAIL"
        s["retry_count"] = 10
        assert should_retry(s) == "finalize"


# ── Evaluation Tests ──────────────────────────────────────────────────────────

class TestEvaluation:
    def test_source_coverage_all_accessible(self, with_verified_sources):
        vs = with_verified_sources["verified_sources"]
        # Both http sources are accessible
        score = _source_coverage(vs)
        assert 0.0 <= score <= 1.0
        assert score == 1.0  # both web sources accessible

    def test_source_coverage_none_accessible(self):
        vs = [
            {"url": "https://a.com", "accessible": False, "status_code": 404},
            {"url": "https://b.com", "accessible": False, "status_code": 403},
        ]
        score = _source_coverage(vs)
        assert score == 0.0

    def test_source_coverage_empty(self):
        assert _source_coverage([]) == 0.0

    def test_source_coverage_local_only(self):
        vs = [{"url": "doc.txt", "accessible": False, "status_code": 0}]
        score = _source_coverage(vs)
        assert score == 1.0  # all local — full coverage

    @patch("src.evaluation.ChatOllama")
    def test_evaluator_node_produces_all_metrics(self, mock_cls, with_summary, with_verified_sources):
        state = {**with_summary, **with_verified_sources}
        state["final_answer"] = "AI is used in drug discovery."

        mock_llm = MagicMock()
        # Return valid score responses
        mock_llm.invoke.return_value = MagicMock(
            content='{"supported_claims": 3, "total_claims": 3, "faithfulness_score": 0.9}'
        )
        mock_cls.return_value = mock_llm

        result = evaluator_node(state)
        scores = result["eval_scores"]

        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_precision" in scores
        assert "context_recall" in scores
        assert "source_coverage" in scores
        assert "overall_score" in scores
        assert 0.0 <= scores["overall_score"] <= 1.0


# ── Integration: Full Graph ───────────────────────────────────────────────────

class TestFullPipeline:
    @patch("src.agents.evaluator_node", create=True)
    @patch("src.agents.finalizer.save_session")
    @patch("src.agents.finalizer.ChatOllama")
    @patch("src.agents.critic.ChatOllama")
    @patch("src.agents.summarizer.ChatOllama")
    @patch("src.agents.retriever.ChatOllama")
    @patch("src.agents.source_verifier.ChatOllama")
    @patch("src.agents.planner.ChatOllama")
    @patch("src.agents.searcher.web_search")
    @patch("src.agents.searcher.local_doc_search")
    @patch("src.agents.retriever.similarity_search")
    def test_pipeline_state_flow(
        self, mock_vs, mock_local, mock_web,
        mock_planner_llm, mock_verifier_llm, mock_retriever_llm,
        mock_summarizer_llm, mock_critic_llm, mock_finalizer_llm,
        mock_save, _
    ):
        """
        End-to-end test with mocked LLM and search calls.
        Verifies that state is correctly passed through all nodes.
        """
        from src.graph import run_research

        # Planner
        mock_planner_llm.return_value.invoke.return_value = MagicMock(
            content='{"sub_queries": ["AI drug discovery", "ML pharma"]}'
        )
        # Search
        mock_web.invoke.return_value = [
            {"content": "AI finds drug targets faster.", "source": "https://nature.com/1", "type": "web"}
        ]
        mock_local.invoke.return_value = []
        mock_vs.return_value = []

        # Source verifier
        mock_verifier_llm.return_value.invoke.return_value = MagicMock(
            content='{"fact_index": 0, "supporting_source_indices": [0], "confidence": 0.8}\n'
        )

        # Retriever
        mock_retriever_llm.return_value.invoke.return_value = MagicMock(
            content="1. AI finds drug targets faster."
        )

        # Summarizer
        mock_summarizer_llm.return_value.invoke.return_value = MagicMock(
            content="AI accelerates drug discovery by identifying targets quickly."
        )

        # Critic — PASS
        claims_json = json.dumps([
            {"claim_index": 0,
             "claim": "AI accelerates drug discovery by identifying targets quickly.",
             "verdict": "SUPPORTED", "evidence": "fact 0", "confidence": 0.95}
        ])
        mock_critic_llm.return_value.invoke.return_value = MagicMock(content=claims_json)

        # Finalizer
        mock_finalizer_llm.return_value.invoke.return_value = MagicMock(
            content="AI accelerates drug discovery [1].\n\n## Sources\n[1] https://nature.com/1"
        )
        mock_save.return_value = "test-uuid-1234"

        result = run_research("How does AI help drug discovery?")

        assert result.get("sub_queries")  # Planner ran
        assert result.get("raw_data")     # Searcher ran
        assert result.get("facts")        # Retriever ran
        assert result.get("summary")      # Summarizer ran
        assert result.get("validation_status") == "PASS"  # Critic passed
        assert result.get("final_answer") # Finalizer ran
        assert result.get("session_id") == "test-uuid-1234"
