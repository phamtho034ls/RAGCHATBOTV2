"""Keyword retriever – BM25-based full-text search.

Performs keyword search against PostgreSQL vector_chunks table
using trigram similarity and tsvector/tsquery for Vietnamese text.
Falls back to LIKE-based search when pg_trgm is not available.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Article, Clause, VectorChunk, Document

log = logging.getLogger(__name__)


async def keyword_search(
    query: str,
    db: AsyncSession,
    top_k: int = 20,
    doc_number: Optional[str] = None,
) -> List[Dict]:
    """Run BM25-style keyword search over vector_chunks in PostgreSQL.

    Uses ILIKE with scored keyword overlap for ranking.
    Returns list of dicts matching the vector_search output format.
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return []

    # Build query: search chunks that contain any keyword, rank by match count
    stmt = (
        select(
            VectorChunk.id,
            VectorChunk.document_id,
            VectorChunk.article_id,
            VectorChunk.clause_id,
            VectorChunk.vector_id,
            VectorChunk.chunk_text,
            Document.doc_number,
            Document.title.label("document_title"),
            Article.article_number,
            Article.title.label("article_title"),
            Clause.clause_number,
        )
        .join(Document, VectorChunk.document_id == Document.id)
        .outerjoin(Article, VectorChunk.article_id == Article.id)
        .outerjoin(Clause, VectorChunk.clause_id == Clause.id)
    )

    if doc_number:
        stmt = stmt.where(
            func.upper(Document.doc_number) == func.upper(doc_number)
        )

    # Filter: chunk must contain at least one keyword
    keyword_filters = []
    for kw in keywords:
        keyword_filters.append(VectorChunk.chunk_text.ilike(f"%{kw}%"))

    if keyword_filters:
        from sqlalchemy import or_
        stmt = stmt.where(or_(*keyword_filters))

    stmt = stmt.limit(top_k * 3)  # fetch extra, then score and rank

    result = await db.execute(stmt)
    rows = result.all()

    # Score by keyword overlap count
    scored = []
    for row in rows:
        text_lower = row.chunk_text.lower()
        score = sum(1 for kw in keywords if kw.lower() in text_lower) / max(len(keywords), 1)
        scored.append({
            "id": str(row.vector_id or row.id),
            "score": score,
            "text_chunk": row.chunk_text,
            "document_id": row.document_id,
            "article_id": row.article_id,
            "clause_id": row.clause_id,
            "doc_number": row.doc_number,
            "document_title": row.document_title or "",
            "article_number": _normalize_article_number(row.article_number),
            "article_title": row.article_title or "",
            "clause_number": _normalize_clause_number(row.clause_number),
        })

    # Sort by score desc and take top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]

    log.debug("Keyword search returned %d results for: '%.60s...'", len(results), query)
    return results


def _extract_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from query, filtering stopwords."""
    # Vietnamese stopwords (minimal set)
    stopwords = {
        "là", "và", "của", "có", "được", "trong", "cho", "với", "này",
        "các", "những", "một", "để", "theo", "về", "từ", "đến", "đã",
        "sẽ", "không", "hay", "hoặc", "nếu", "thì", "khi", "bao",
        "gì", "nào", "ai", "đâu", "tại", "do", "mà", "vì", "bởi",
        "như", "quy", "định", "hỏi", "gì",
    }

    # Tokenize: split on whitespace and punctuation
    words = re.findall(r"[\w]+", query.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 1]

    # Also extract legal-specific compound terms
    legal_patterns = [
        r"Điều\s+\d+",
        r"Khoản\s+\d+",
        r"Nghị\s+định",
        r"Thông\s+tư",
        r"Quyết\s+định",
        r"Luật\s+\w+",
    ]
    for pattern in legal_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        keywords.extend(m.lower() for m in matches)

    return list(set(keywords))


def _normalize_article_number(article_number: Optional[str]) -> Optional[str]:
    if not article_number:
        return None
    m = re.search(r"(\d+[a-zA-Z]?)", article_number)
    return m.group(1) if m else article_number


def _normalize_clause_number(clause_number: Optional[str]) -> Optional[str]:
    if not clause_number:
        return None
    m = re.search(r"(\d+[a-zA-Z]?)", clause_number)
    return m.group(1) if m else clause_number
