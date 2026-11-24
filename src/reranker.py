"""
Cross-Encoder Reranker
======================
Uses a sentence-transformers cross-encoder model to score (query, chunk)
pairs and rerank retrieved chunks by relevance. This runs entirely locally
via sentence-transformers — no API calls needed.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M params, runs on CPU in ~100ms per batch
  - Trained on MS MARCO passage ranking
"""

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MAX_RERANK = 20   # max candidates to pass to cross-encoder


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load and cache the cross-encoder model (loaded once per process)."""
    try:
        from sentence_transformers import CrossEncoder
        logger.info("[Reranker] Loading cross-encoder model: %s", _DEFAULT_RERANKER_MODEL)
        model = CrossEncoder(_DEFAULT_RERANKER_MODEL, max_length=512)
        logger.info("[Reranker] Cross-encoder loaded.")
        return model
    except ImportError:
        logger.warning(
            "[Reranker] sentence-transformers not installed — reranking disabled."
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("[Reranker] Failed to load cross-encoder: %s", exc)
        return None


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Rerank *chunks* by cross-encoder relevance to *query*.

    Each chunk dict must have a 'content' key.
    Returns the top_k chunks sorted by descending score, with
    a 'rerank_score' field added.

    If the cross-encoder is unavailable, returns chunks unchanged.
    """
    if not chunks:
        return []

    model = _get_cross_encoder()

    if model is None:
        # Graceful degradation: return first top_k as-is
        logger.warning("[Reranker] No model available — skipping rerank.")
        return chunks[:top_k]

    # Cap candidates to avoid OOM on large batches
    candidates = chunks[:_MAX_RERANK]

    pairs = [(query, c.get("content", "")[:512]) for c in candidates]

    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        logger.error("[Reranker] Cross-encoder predict failed: %s", exc)
        return chunks[:top_k]

    # Attach scores and sort
    scored = []
    for chunk, score in zip(candidates, scores):
        scored.append({**chunk, "rerank_score": float(score)})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    logger.info(
        "[Reranker] Reranked %d → %d chunks. Top score: %.4f",
        len(candidates),
        min(top_k, len(scored)),
        scored[0]["rerank_score"] if scored else 0.0,
    )

    return scored[:top_k]
