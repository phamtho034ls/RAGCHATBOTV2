"""Retrieval package – hybrid search with reranking."""

from __future__ import annotations

__all__ = ["hybrid_search"]


def __getattr__(name: str):
    if name == "hybrid_search":
        from app.retrieval.hybrid_retriever import hybrid_search

        return hybrid_search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
