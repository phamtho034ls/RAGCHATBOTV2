"""Embedding generation for the ingestion pipeline.

Wraps sentence-transformers for batch embedding of text chunks.
Provides pipeline-specific batch processing with lazy model loading.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_FALLBACK_MODEL,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL,
    HF_TOKEN,
)

log = logging.getLogger(__name__)

_model = None
_effective_model_name: str | None = None
# ST ≥ 3.x: một số model (vd. keepitreal/vietnamese-sbert) khai báo không nhận kwargs thêm → encode(truncation=True) lỗi (không phải TypeError).
_truncation_encode_cache: dict[int, bool] = {}


def _encode_truncation_supported(model) -> bool:
    """Trả True nếu được phép truyền truncation=True vào SentenceTransformer.encode."""
    mid = id(model)
    if mid in _truncation_encode_cache:
        return _truncation_encode_cache[mid]

    supported = True
    gmk = getattr(model, "get_model_kwargs", None)
    if callable(gmk):
        try:
            allowed = gmk()
        except Exception:
            allowed = None
        if allowed is not None:
            if len(allowed) == 0:
                supported = False
            elif isinstance(allowed, (list, tuple, set)) and "truncation" not in allowed:
                supported = False

    if not supported:
        log.info(
            "SentenceTransformer.encode: bỏ truncation=True (model không khai báo trong get_model_kwargs).",
        )
    _truncation_encode_cache[mid] = supported
    return supported


def _apply_safe_max_seq_length(model) -> int:
    """Không set max_seq_length vượt quá max_position_embeddings (tránh CUDA index OOB)."""
    cap = int(EMBEDDING_MAX_LENGTH)
    try:
        inner = model[0]
        am = getattr(inner, "auto_model", None)
        cfg = getattr(am, "config", None) if am is not None else None
        mp = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
        if mp is not None:
            # BERT/RoBERTa/XLM-R: chừa chỗ cho special tokens
            safe = min(cap, max(8, int(mp) - 2))
            model.max_seq_length = safe
            log.info(
                "embedding max_seq_length=%d (model max_position_embeddings=%d, env cap=%d)",
                safe,
                mp,
                cap,
            )
            return safe
    except Exception as exc:
        log.warning("Could not align max_seq_length to model config: %s", exc)
    model.max_seq_length = cap
    log.info("embedding max_seq_length=%d (fallback, no config)", cap)
    return cap


def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    return SentenceTransformer(model_name, device=EMBEDDING_DEVICE, **kwargs)


def _get_model():
    """Lazy-load the sentence-transformer model (primary, fallback nếu lỗi)."""
    global _model, _effective_model_name
    if _model is None:
        try:
            _model = _load_sentence_transformer(EMBEDDING_MODEL)
            _effective_model_name = EMBEDDING_MODEL
            log.info("Loaded embedding model '%s' on %s.", EMBEDDING_MODEL, EMBEDDING_DEVICE)
        except Exception as exc:
            log.warning(
                "Primary embedding model '%s' failed (%s); falling back to '%s'.",
                EMBEDDING_MODEL,
                exc,
                EMBEDDING_FALLBACK_MODEL,
            )
            _model = _load_sentence_transformer(EMBEDDING_FALLBACK_MODEL)
            _effective_model_name = EMBEDDING_FALLBACK_MODEL
            log.info(
                "Loaded fallback embedding '%s' on %s — set EMBEDDING_DIM in .env to match.",
                EMBEDDING_FALLBACK_MODEL,
                EMBEDDING_DEVICE,
            )
        _apply_safe_max_seq_length(_model)
        dim = _model.get_sentence_embedding_dimension()
        log.info("Embedding dimension (model output) = %d", dim)
    return _model


def get_embedding_dimension() -> int:
    """Kích thước vector thực tế của model đang load."""
    return int(_get_model().get_sentence_embedding_dimension())


def embed_texts(texts: List[str], batch_size: int | None = None) -> np.ndarray:
    """Embed a list of texts. Returns numpy array of shape (N, dim)."""
    if not texts:
        return np.array([])

    model = _get_model()
    bs = batch_size or EMBEDDING_BATCH_SIZE
    # Tránh lỗi tokenizer / CUDA với chuỗi rỗng hoặc chỉ khoảng trắng
    cleaned = [(t.strip() if isinstance(t, str) else "") or " " for t in texts]

    enc_kw = dict(
        batch_size=bs,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    if _encode_truncation_supported(model):
        try:
            embeddings = model.encode(cleaned, truncation=True, **enc_kw)
        except TypeError:
            embeddings = model.encode(cleaned, **enc_kw)
    else:
        embeddings = model.encode(cleaned, **enc_kw)
    return embeddings.astype(np.float32)


def embed_query(query: str) -> List[float]:
    """Embed a single query. Returns list of floats."""
    model = _get_model()
    q = (query or "").strip() or " "
    base_kw = dict(normalize_embeddings=True, convert_to_numpy=True)
    if _encode_truncation_supported(model):
        try:
            vec = model.encode(q, truncation=True, **base_kw)
        except TypeError:
            vec = model.encode(q, **base_kw)
    else:
        vec = model.encode(q, **base_kw)
    return vec.astype(np.float32).tolist()


def warmup() -> None:
    """Pre-load the model (call at startup)."""
    _get_model()
    log.info("Embedding model warmed up (%s).", _effective_model_name or EMBEDDING_MODEL)
