"""Embedding generation for the ingestion pipeline.

Wraps sentence-transformers for batch embedding of text chunks.
Provides pipeline-specific batch processing with lazy model loading.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from app.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE, EMBEDDING_MODEL, HF_TOKEN

log = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        _model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE, **kwargs)
        log.info("Loaded embedding model '%s' on %s.", EMBEDDING_MODEL, EMBEDDING_DEVICE)
    return _model


def embed_texts(texts: List[str], batch_size: int | None = None) -> np.ndarray:
    """Embed a list of texts. Returns numpy array of shape (N, dim)."""
    if not texts:
        return np.array([])

    model = _get_model()
    bs = batch_size or EMBEDDING_BATCH_SIZE

    embeddings = model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> List[float]:
    """Embed a single query. Returns list of floats."""
    model = _get_model()
    vec = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32).tolist()


def warmup() -> None:
    """Pre-load the model (call at startup)."""
    _get_model()
    log.info("Embedding model warmed up.")
