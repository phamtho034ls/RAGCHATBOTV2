"""Keyword retriever – BM25-based full-text search.

Performs keyword search against PostgreSQL vector_chunks table
using trigram similarity and tsvector/tsquery for Vietnamese text.
Falls back to LIKE-based search when pg_trgm is not available.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from sqlalchemy import select, text, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Article, Clause, VectorChunk, Document

log = logging.getLogger(__name__)

# Cụm từ ưu tiên ILIKE trên toàn bộ nội dung Điều (khi chưa có search_vector).
_LEGAL_ILIKE_PHRASES: tuple[str, ...] = (
    "trợ giúp xã hội",
    "hoạt động trợ giúp",
    "đăng ký hoạt động",
    "cơ sở trợ giúp",
    "tiếng ồn",
    "karaoke",
    "đo độ ồn",
    "xử phạt hành chính",
)


def _extract_ilike_terms(query: str) -> List[str]:
    q = (query or "").strip().lower()
    if not q:
        return []
    terms = list(_extract_keywords(query))
    for ph in _LEGAL_ILIKE_PHRASES:
        if ph in q:
            terms.append(ph)
    # Ưu tiên chuỗi dài (khớp cụm) trước từ rời
    uniq: List[str] = []
    seen = set()
    for t in sorted(set(terms), key=len, reverse=True):
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:12]


async def _article_content_ilike_search(
    query: str,
    db: AsyncSession,
    limit: int = 40,
    doc_number: Optional[str] = None,
) -> List[Dict]:
    """Khi FTS (search_vector) không dùng được: ILIKE trực tiếp trên ``articles.content``."""
    terms = _extract_ilike_terms(query)
    if not terms:
        return []

    conds = []
    for kw in terms:
        if len(kw) >= 2:
            conds.append(Article.content.ilike(f"%{kw}%"))

    if not conds:
        return []

    stmt = (
        select(
            Article.id.label("article_id"),
            Article.document_id,
            Article.article_number,
            Article.title.label("article_title"),
            func.substr(Article.content, 1, 6000).label("chunk_text"),
            Document.doc_number,
            Document.title.label("document_title"),
        )
        .join(Document, Article.document_id == Document.id)
        .where(or_(*conds))
    )
    if doc_number:
        stmt = stmt.where(func.upper(func.trim(Document.doc_number)) == func.upper(func.trim(doc_number)))
    stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    rows = result.mappings().all()
    out: List[Dict] = []
    for row in rows:
        txt = row["chunk_text"] or ""
        score = sum(1 for t in terms if t.lower() in txt.lower()) / max(len(terms), 1)
        out.append(
            {
                "id": f"ailike-{row['article_id']}",
                "score": float(score),
                "text_chunk": txt,
                "document_id": row["document_id"],
                "article_id": row["article_id"],
                "clause_id": None,
                "doc_number": row["doc_number"],
                "document_title": row["document_title"] or "",
                "article_number": _normalize_article_number(row["article_number"]),
                "article_title": row["article_title"] or "",
                "clause_number": None,
            }
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    log.debug("Article ILIKE fallback returned %d rows", len(out))
    return out


async def full_text_search(
    query: str,
    db: AsyncSession,
    limit: int = 20,
    doc_number: Optional[str] = None,
) -> List[Dict]:
    """Full-text search trên bảng ``articles`` (tsvector + unaccent).

    Trả về dict cùng format ``keyword_search`` / ``vector_search`` (text_chunk, scores...).
    Nếu cột ``search_vector`` chưa có (chưa migration) → [].
    """
    q = (query or "").strip()
    if not q:
        return []

    stmt = text(
        """
        SELECT
          a.id AS article_id,
          a.document_id,
          a.article_number,
          a.title AS article_title,
          LEFT(a.content, 6000) AS chunk_text,
          d.doc_number,
          d.title AS document_title,
          ts_rank_cd(
            a.search_vector,
            plainto_tsquery('simple', public.unaccent(:q))
          ) AS fts_rank
        FROM articles a
        JOIN documents d ON d.id = a.document_id
        WHERE a.search_vector @@ plainto_tsquery('simple', public.unaccent(:q))
          AND (
            :doc_number IS NULL
            OR UPPER(TRIM(d.doc_number)) = UPPER(TRIM(:doc_number))
          )
        ORDER BY fts_rank DESC
        LIMIT :lim
        """
    )
    try:
        # SAVEPOINT: FTS lỗi → chỉ rollback savepoint, ILIKE fallback vẫn chạy được.
        async with db.begin_nested():
            res = await db.execute(
                stmt,
                {"q": q, "lim": limit, "doc_number": doc_number},
            )
            rows = res.mappings().all()
    except Exception as exc:
        log.warning(
            "full_text_search skipped (FTS/error — dùng ILIKE): %s",
            exc,
        )
        return []

    out: List[Dict] = []
    for row in rows:
        rank = float(row["fts_rank"] or 0.0)
        out.append(
            {
                "id": f"fts-article-{row['article_id']}",
                "score": rank,
                "text_chunk": row["chunk_text"] or "",
                "document_id": row["document_id"],
                "article_id": row["article_id"],
                "clause_id": None,
                "doc_number": row["doc_number"],
                "document_title": row["document_title"] or "",
                "article_number": _normalize_article_number(row["article_number"]),
                "article_title": row["article_title"] or "",
                "clause_number": None,
            }
        )
    return out


async def keyword_search(
    query: str,
    db: AsyncSession,
    top_k: int = 20,
    doc_number: Optional[str] = None,
) -> List[Dict]:
    """Keyword search: ưu tiên FTS ``articles`` (unaccent + tsvector), fallback ILIKE trên chunks."""
    keywords = _extract_keywords(query)
    if not keywords and not (query or "").strip():
        return []

    fts_hits = await full_text_search(query, db, limit=top_k * 2, doc_number=doc_number)
    if fts_hits:
        log.debug("Keyword path: FTS articles returned %d hits", len(fts_hits))
        return fts_hits[:top_k]

    article_ilike = await _article_content_ilike_search(
        query, db, limit=top_k * 2, doc_number=doc_number
    )
    if article_ilike:
        log.debug("Keyword path: article ILIKE returned %d hits", len(article_ilike))
        return article_ilike[:top_k]

    if not keywords:
        return []

    # Fallback: ILIKE trên vector_chunks
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
