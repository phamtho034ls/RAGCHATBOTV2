"""Reranker – cross-encoder / FlagReranker scoring for passage relevance.

Ưu tiên BAAI/bge-reranker-v2-m3 qua FlagEmbedding (đa ngôn ngữ, tiếng Việt tốt hơn).
Fallback: CrossEncoder với BAAI/bge-reranker-base nếu không tải được model mới.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Union

from app.config import RERANKER_BATCH_SIZE, RERANKER_DEVICE, RERANKER_FALLBACK_MODEL, RERANKER_MODEL

log = logging.getLogger(__name__)

_reranker: Union[object, None] = None
_reranker_backend: str = ""


def warmup():
    """Pre-load the reranker model at startup."""
    _get_reranker()


def _get_reranker():
    """Lazy-load reranker: FlagReranker (v2-m3) hoặc CrossEncoder (fallback)."""
    global _reranker, _reranker_backend
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker

            _reranker = FlagReranker(RERANKER_MODEL, use_fp16=False, device=RERANKER_DEVICE)
            _reranker_backend = f"FlagReranker:{RERANKER_MODEL}"
            log.info(
                "Loaded reranker '%s' via FlagEmbedding on %s (multilingual; tốt cho tiếng Việt).",
                RERANKER_MODEL,
                RERANKER_DEVICE,
            )
        except Exception as exc:
            log.warning(
                "FlagReranker load failed for '%s' (%s); falling back to CrossEncoder '%s'.",
                RERANKER_MODEL,
                exc,
                RERANKER_FALLBACK_MODEL,
            )
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(RERANKER_FALLBACK_MODEL, device=RERANKER_DEVICE)
            _reranker_backend = f"CrossEncoder:{RERANKER_FALLBACK_MODEL}"
            log.info("Loaded fallback reranker '%s' on %s.", RERANKER_FALLBACK_MODEL, RERANKER_DEVICE)
    return _reranker


def rerank(
    query: str,
    candidates: List[Dict],
    top_k: int = 5,
    text_key: str = "text_chunk",
) -> List[Dict]:
    """Rerank candidate passages. Each dict gets ``rerank_score`` field."""
    if not candidates:
        return []

    model = _get_reranker()
    pairs: List[Tuple[str, str]] = []
    for c in candidates:
        passage = c.get(text_key, "")
        pairs.append((query, passage))

    if _reranker_backend.startswith("FlagReranker"):
        try:
            scores = model.compute_score(pairs, batch_size=RERANKER_BATCH_SIZE)
        except TypeError:
            scores = model.compute_score(pairs)
        if not isinstance(scores, list):
            scores = list(scores)
    else:
        scores = model.predict(pairs, show_progress_bar=False, batch_size=RERANKER_BATCH_SIZE)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
