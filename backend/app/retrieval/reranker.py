"""Reranker – cross-encoder scoring for passage relevance.

Uses BAAI/bge-reranker-base (or configurable model) to rerank
candidate passages against the user query.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from app.config import RERANKER_DEVICE, RERANKER_MODEL

log = logging.getLogger(__name__)

_reranker = None


def warmup():
    """Pre-load the reranker model at startup."""
    _get_reranker()


def _get_reranker():
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)
        log.info("Loaded reranker '%s' on %s.", RERANKER_MODEL, RERANKER_DEVICE)
    return _reranker


def rerank(
    query: str,
    candidates: List[Dict],
    top_k: int = 5,
    text_key: str = "text_chunk",
) -> List[Dict]:
    """Rerank candidate passages using a cross-encoder model.

    Args:
        query: The user's question.
        candidates: List of dicts, each must have ``text_key`` field.
        top_k: Number of top results to return after reranking.
        text_key: Key in each candidate dict that holds the passage text.

    Returns:
        Top-K candidates sorted by reranker score (highest first).
        Each dict gets an added ``rerank_score`` field.
    """
    if not candidates:
        return []

    model = _get_reranker()

    # Build (query, passage) pairs
    pairs: List[Tuple[str, str]] = []
    for c in candidates:
        passage = c.get(text_key, "")
        pairs.append((query, passage))

    # Score all pairs
    scores = model.predict(pairs, show_progress_bar=False)

    # Attach scores
    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    # Sort by rerank_score descending
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return candidates[:top_k]
