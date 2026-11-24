"""
Retriever (Filter) Agent — Upgraded
=====================================
Implements a full hybrid retrieval pipeline:

1. BM25 keyword search  — exact term matching over raw_data content
2. Vector similarity    — ChromaDB semantic search over the original question
3. Score fusion         — Reciprocal Rank Fusion (RRF) of BM25 + vector rankings
4. Cross-encoder rerank — sentence-transformers for precise relevance scoring
5. Fact extraction      — LLM extracts factual sentences from top-ranked chunks

Key improvements over v1:
- BM25 catches exact terminology that vector search misses (e.g. drug names, model numbers)
- Cross-encoder reranker dramatically improves precision
- Explicit ranked_chunks field for Evaluator to measure context precision/recall
"""

import logging
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.state import ResearchState
from src.vectorstore import similarity_search
from src.reranker import rerank

logger = logging.getLogger(__name__)

# BM25 config
_BM25_TOP_K = 15
_VECTOR_TOP_K = 10
_RERANK_TOP_K = 10
_RRF_K = 60  # constant for Reciprocal Rank Fusion


# ── BM25 Search ───────────────────────────────────────────────────────────────

def _bm25_search(query: str, corpus: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """
    Run BM25 keyword search over raw_data content.
    Returns corpus entries sorted by BM25 score.
    """
    if not corpus:
        return []

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("[Retriever] rank-bm25 not installed — skipping BM25.")
        return corpus[:top_k]

    # Tokenise content (simple whitespace split, lowercase)
    tokenised = [c.get("content", "").lower().split() for c in corpus]
    query_tokens = query.lower().split()

    bm25 = BM25Okapi(tokenised)
    scores = bm25.get_scores(query_tokens)

    # Sort indices by score descending
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for idx in ranked_indices[:top_k]:
        entry = {**corpus[idx], "bm25_score": float(scores[idx])}
        results.append(entry)

    logger.debug("[Retriever] BM25 top score: %.4f", scores[ranked_indices[0]] if ranked_indices else 0)
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_fusion(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    k: int = _RRF_K,
) -> list[dict[str, Any]]:
    """
    Merge BM25 and vector search results via Reciprocal Rank Fusion.

    RRF score = Σ 1/(k + rank_i)
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    def _id(r: dict) -> str:
        return r.get("content", "")[:100]

    for rank, r in enumerate(bm25_results):
        rid = _id(r)
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        items[rid] = r

    for rank, r in enumerate(vector_results):
        rid = _id(r)
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        items.setdefault(rid, r)

    merged = sorted(items.values(), key=lambda r: scores.get(_id(r), 0.0), reverse=True)
    for r in merged:
        r["rrf_score"] = round(scores.get(_id(r), 0.0), 6)

    return merged


# ── Fact Extraction ───────────────────────────────────────────────────────────

_FACT_SYSTEM = """\
You are a precise fact extractor. Given a user question and relevant text
chunks, extract 10-20 distinct, factual sentences that are directly relevant
to answering the question.

Rules:
- Each fact must be a complete, standalone sentence
- Do NOT invent or infer beyond the provided text
- Remove duplicates and near-duplicates
- Prefer specific, concrete facts over vague statements

Output a numbered list:
1. <fact>
2. <fact>
...
"""

def _extract_facts(
    question: str,
    top_chunks: list[dict[str, Any]],
    llm: ChatOllama,
) -> list[str]:
    """Extract atomic factual sentences from top-ranked chunks."""
    if not top_chunks:
        return []

    chunks_text = "\n\n".join(
        f"[Chunk {i+1}] {c.get('content', '')[:600]}"
        for i, c in enumerate(top_chunks[:8])
    )

    messages = [
        SystemMessage(content=_FACT_SYSTEM),
        HumanMessage(
            content=f"Question: {question}\n\nText chunks:\n{chunks_text}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        facts = []
        for line in response.content.strip().splitlines():
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                facts.append(line.split(". ", 1)[1].strip())
        if not facts:
            facts = [l.strip() for l in response.content.splitlines() if l.strip()]
        return facts[:20]
    except Exception as exc:
        logger.error("[Retriever] Fact extraction failed: %s", exc)
        return [c.get("content", "")[:200] for c in top_chunks[:5]]


# ── Main Node ─────────────────────────────────────────────────────────────────

def retriever_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Retriever.

    Hybrid BM25 + vector search → RRF fusion → cross-encoder rerank → fact extraction.

    Updates state fields:
      - ranked_chunks (reranked list for context precision/recall evaluation)
      - facts         (extracted factual sentences)
    """
    question = state.get("question", "")
    sub_queries = state.get("sub_queries", [question])
    raw_data = state.get("raw_data", [])

    logger.info("[Retriever] Running hybrid retrieval on %d raw chunks.", len(raw_data))

    if not raw_data:
        logger.warning("[Retriever] No raw data to retrieve from.")
        return {"ranked_chunks": [], "facts": []}

    # Use the original question as the retrieval query (most focused)
    query = question

    # ── BM25 over raw_data ────────────────────────────────────────────────────
    bm25_results = _bm25_search(query, raw_data, top_k=_BM25_TOP_K)

    # ── Vector search over ChromaDB ───────────────────────────────────────────
    vector_results: list[dict] = []
    try:
        vector_results = similarity_search(query, k=_VECTOR_TOP_K)
    except Exception as exc:
        logger.warning("[Retriever] ChromaDB vector search failed: %s", exc)
        # Fall back to BM25 results only
        vector_results = []

    logger.info(
        "[Retriever] BM25=%d results, Vector=%d results — fusing...",
        len(bm25_results), len(vector_results),
    )

    # ── RRF Fusion ────────────────────────────────────────────────────────────
    fused = _rrf_fusion(bm25_results, vector_results)
    logger.info("[Retriever] After RRF fusion: %d unique candidates.", len(fused))

    # ── Cross-encoder Rerank ──────────────────────────────────────────────────
    ranked_chunks = rerank(query, fused, top_k=_RERANK_TOP_K)
    logger.info("[Retriever] After reranking: %d top chunks.", len(ranked_chunks))

    # ── Fact Extraction ───────────────────────────────────────────────────────
    cfg = get_settings()
    llm = ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        temperature=0,
    )

    facts = _extract_facts(question, ranked_chunks, llm)
    logger.info("[Retriever] Extracted %d facts from top chunks.", len(facts))

    return {
        "ranked_chunks": ranked_chunks,
        "facts": facts,
    }
