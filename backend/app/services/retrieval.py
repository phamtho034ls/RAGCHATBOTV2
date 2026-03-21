"""Retrieval adapter – bridges legacy API to v2 hybrid retrieval.

Provides backward-compatible search functions (``search_all``,
``search_with_fallback``, ``search_by_metadata``, ``has_any_dataset``)
using the v2 infrastructure (Qdrant + PostgreSQL + Hybrid Retriever).

All functions auto-manage database sessions so callers don't need to
inject ``AsyncSession`` themselves.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, or_

from app.config import QUERY_REWRITE_PROMPT, RETRIEVAL_TOP_K
from app.database.models import Document, VectorChunk
from app.database.session import _session_factory
from app.retrieval.hybrid_retriever import hybrid_search

log = logging.getLogger(__name__)


# ── Conversion helper ─────────────────────────────────────

def _v2_to_legacy(results: List[Dict]) -> List[Dict]:
    """Convert v2 hybrid_search output to legacy format.

    Legacy format: {"text", "score", "rerank_score", "dataset_id", "metadata", ...}
    """
    return [
        {
            "id": r.get("id", ""),
            "text": r.get("text_chunk", ""),
            "score": r.get("rerank_score", r.get("rrf_score", r.get("score", 0.0))),
            "rerank_score": r.get("rerank_score", 0.0),
            "hybrid_score": r.get("rrf_score", 0.0),
            "dataset_id": str(r.get("document_id", "")),
            "metadata": {
                "document_id": r.get("document_id"),
                "article_id": r.get("article_id"),
                "clause_id": r.get("clause_id"),
                "doc_number": r.get("doc_number", ""),
                "law_name": r.get("document_title") or r.get("law_name") or r.get("doc_number", ""),
                "article_number": r.get("article_number"),
                "clause_number": r.get("clause_number"),
            },
        }
        for r in results
    ]


# ── Public API (backward-compatible) ──────────────────────

async def search_all(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    query_variants: Optional[List[str]] = None,
) -> List[Dict]:
    """Search using v2 hybrid retrieval (Qdrant + PostgreSQL).

    Drop-in replacement for ``app.services.vector_store.search_all``.
    """
    doc_number = (filters or {}).get("doc_number")

    async with _session_factory() as db:
        results = await hybrid_search(
            query=query,
            db=db,
            top_k=top_k,
            retrieval_k=max(top_k, RETRIEVAL_TOP_K),
            doc_number=doc_number,
        )

        # If variants provided and not enough results, supplement
        if query_variants and len(results) < top_k:
            seen_ids = {r.get("id") for r in results}
            for variant in query_variants[:3]:
                extra = await hybrid_search(
                    query=variant,
                    db=db,
                    top_k=top_k,
                    doc_number=doc_number,
                    doc_number_source_query=query,
                )
                for r in extra:
                    if r.get("id") not in seen_ids:
                        results.append(r)
                        seen_ids.add(r.get("id"))
                if len(results) >= top_k:
                    break

    return _v2_to_legacy(results[:top_k])


async def search_with_fallback(
    query: str,
    keywords: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    query_variants: Optional[List[str]] = None,
    min_results: int = 3,
) -> List[Dict]:
    """Search with fallback – drop-in replacement for legacy version.

    The v2 hybrid retriever already combines vector + keyword search,
    so fallback logic is inherently satisfied.
    """
    results = await search_all(
        query=query,
        filters=filters,
        top_k=max(top_k, min_results),
        query_variants=query_variants,
    )

    # If still below min_results and we have keywords, do keyword-only search
    if len(results) < min_results and keywords:
        kw_query = " ".join(keywords[:6])
        extra = await search_all(query=kw_query, filters=filters, top_k=top_k)
        seen_ids = {r.get("id") for r in results}
        for r in extra:
            if r.get("id") not in seen_ids:
                results.append(r)
                seen_ids.add(r.get("id"))
            if len(results) >= top_k:
                break

    return results


async def search_by_metadata(
    linh_vuc: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    only_effective: bool = True,
    limit: int = 20,
) -> List[Dict]:
    """Search documents by metadata in PostgreSQL.

    Drop-in replacement for ``app.services.db.search_by_metadata``.
    """
    async with _session_factory() as db:
        stmt = (
            select(
                VectorChunk.id,
                VectorChunk.document_id,
                VectorChunk.article_id,
                VectorChunk.clause_id,
                VectorChunk.vector_id,
                VectorChunk.chunk_text,
                Document.doc_number,
                Document.title,
                Document.document_type,
                Document.issuer,
            )
            .join(Document, VectorChunk.document_id == Document.id)
        )

        filters = []
        if linh_vuc:
            filters.append(Document.title.ilike(f"%{linh_vuc}%"))
        if keywords:
            kw_filters = [VectorChunk.chunk_text.ilike(f"%{kw}%") for kw in keywords]
            filters.append(or_(*kw_filters))
        if filters:
            stmt = stmt.where(*filters)

        stmt = stmt.limit(limit)
        result = await db.execute(stmt)
        rows = result.all()

    return [
        {
            "id": str(row.vector_id or row.id),
            "chunk_text": row.chunk_text,
            "dataset_id": str(row.document_id),
            "document_id": row.document_id,
            "doc_number": row.doc_number,
            "law_name": row.title or row.doc_number,
            "document_type": row.document_type,
            "source_file": row.title,
        }
        for row in rows
    ]


async def has_any_dataset() -> bool:
    """Check whether any documents have been ingested."""
    async with _session_factory() as db:
        count = (await db.execute(
            select(func.count()).select_from(Document)
        )).scalar() or 0
    return count > 0


def format_sources(docs: List[dict]) -> List[dict]:
    """Format search results as source citations (legacy format)."""
    return [
        {
            "content": doc.get("text", doc.get("text_chunk", "")),
            "score": round(
                doc.get("rerank_score", doc.get("hybrid_score", doc.get("score", 0))),
                4,
            ),
            "dataset_id": doc.get("dataset_id", ""),
            "metadata": doc.get("metadata", {}),
        }
        for doc in docs
    ]


async def rewrite_query(question: str) -> str:
    """Rewrite a query for better retrieval."""
    from app.services.llm_client import generate

    try:
        prompt = QUERY_REWRITE_PROMPT.format(question=question)
        rewritten = (await generate(prompt, temperature=0.0)).strip()
        if rewritten and len(rewritten) > 5:
            return rewritten
    except Exception:
        pass
    return question
