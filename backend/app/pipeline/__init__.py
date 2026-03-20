"""Pipeline package for legal ingestion and preprocessing."""

from __future__ import annotations


async def ingest_document(*args, **kwargs):
    """Lazy proxy to avoid importing heavy vector dependencies at package import."""
    from app.pipeline.ingestor import ingest_document as _ingest_document

    return await _ingest_document(*args, **kwargs)


__all__ = ["ingest_document"]
