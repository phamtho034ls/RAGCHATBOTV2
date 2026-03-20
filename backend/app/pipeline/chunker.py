"""Backward-compatible shim to legal-aware chunker."""

from app.pipeline.legal_chunker import (
    Chunk,
    attach_context_prefix,
    chunk_articles,
    chunk_by_article,
    chunk_by_clause_primary,
    chunk_by_clause_if_needed,
    chunk_preamble,
)

__all__ = [
    "Chunk",
    "attach_context_prefix",
    "chunk_by_article",
    "chunk_by_clause_primary",
    "chunk_by_clause_if_needed",
    "chunk_articles",
    "chunk_preamble",
]
