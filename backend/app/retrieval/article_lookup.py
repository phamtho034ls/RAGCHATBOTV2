"""Direct DB lookup for article/clause and topic-based queries.

When users ask about a specific Điều/Khoản or a topic within a named law,
this module queries the PostgreSQL articles/clauses tables directly instead
of relying on vector similarity search.

Handles queries like:
  - "Điều 7, Khoản 4 của Luật Di sản văn hóa 2024 quy định gì?"
  - "Điều 7 Luật Di sản văn hóa 2024 quy định gì?"
  - "Luật Quảng cáo Điều 12"
  - "Các hành vi bị nghiêm cấm trong Luật Di sản văn hóa"  (topic-based)
  - "Quyền của tổ chức trong Luật Quảng cáo"                 (topic-based)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, List, Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Article, Clause, Document

log = logging.getLogger(__name__)

# ── Query parsing ─────────────────────────────────────────

_ARTICLE_RE = re.compile(r"[Đđ]iều\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"[Kk]hoản\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
_POINT_RE = re.compile(r"[Đđ]iểm\s+([a-zđ])", re.IGNORECASE)

_DOC_TYPE_KW = r"(?:Luật|Nghị\s*định|Thông\s*tư|Quyết\s*định|Chỉ\s*thị|Nghị\s*quyết)"

# Detect general/summary queries (tóm tắt, nội dung, tổng quan, ...)
_GENERAL_QUERY_RE = re.compile(
    r"(?:tóm\s*tắt|nội\s*dung|tổng\s*quan|cho\s*biết|giới\s*thiệu"
    r"|trình\s*bày|liệt\s*kê|bao\s*gồm\s*(?:những|các)\s*gì)",
    re.IGNORECASE,
)
_DOC_NAME_STOP_RE = re.compile(
    r"\s+(?:"
    r"điều|khoản|điểm"
    r"|quy\s*định\s+(?:gì|về)"
    r"|nói\s+gì|là\s+gì"
    r"|xử\s*phạt|hướng\s*dẫn|sửa\s*đổi|bổ\s*sung"
    r"|có\s+bao\s+nhiêu|gồm\s+(?:những|các)"
    r"|theo|bao\s+gồm"
    r")\b",
    re.IGNORECASE,
)


def parse_article_clause_query(query: str) -> Optional[Dict]:
    """Parse a query to extract article number, clause number, and document name.

    Returns None if the query doesn't reference a specific article.
    """
    article_match = _ARTICLE_RE.search(query)
    if not article_match:
        return None

    article_number = article_match.group(1)
    clause_number = None
    point_letter = None

    clause_match = _CLAUSE_RE.search(query)
    if clause_match:
        clause_number = clause_match.group(1)

    point_match = _POINT_RE.search(query)
    if point_match:
        point_letter = point_match.group(1)

    doc_name = _extract_document_name(query)
    year = _extract_year(query)

    result = {
        "article_number": article_number,
        "clause_number": clause_number,
        "point_letter": point_letter,
        "doc_name": doc_name,
        "year": year,
    }
    log.info("[ARTICLE_LOOKUP] Parsed query: %s", result)
    return result


def _extract_document_name(query: str) -> Optional[str]:
    """Extract the document name/title from the query.

    Finds the doc type keyword (Luật, Nghị định, ...) and captures everything
    after it until a stop phrase (Điều, Khoản, quy định gì, ...) or year or end.
    """
    doc_type_match = re.search(_DOC_TYPE_KW, query, re.IGNORECASE)
    if not doc_type_match:
        return None

    start = doc_type_match.start()
    tail = query[start:]

    stop = _DOC_NAME_STOP_RE.search(tail)
    if stop:
        name = tail[:stop.start()].strip()
    else:
        name = tail.strip()

    name = re.sub(r"\s+(?:năm\s+)?\d{4}\s*$", "", name).strip()
    name = re.sub(r"\s*[,?].*$", "", name).strip()

    # Reject interrogative phrases that aren't actual document names
    interrogative_endings = ["nào", "gì", "nào đó", "nào khác"]
    name_lower = name.lower().strip()
    for ending in interrogative_endings:
        if name_lower.endswith(ending):
            name = name[:name_lower.rfind(ending.split()[0])].strip()
            break

    if len(name) > 5:
        return name
    return None


def _extract_year(query: str) -> Optional[int]:
    """Extract year from query (e.g., '2024', 'năm 2024')."""
    m = re.search(r"(?:năm\s+)?(\d{4})", query)
    if m:
        year = int(m.group(1))
        if 1990 <= year <= 2030:
            return year
    return None


def _extract_doc_number_ref(query: str) -> Optional[str]:
    """Extract a document number reference from the query.

    Handles: 49/2025/QĐ-UBND, 06/CT-UBND, 144/2021/NĐ-CP,
             06CT-UBND, 06CT – UBND (no slash variants).
    """
    # num/year/type-issuer: 49/2025/QĐ-UBND, 144/2021/NĐ-CP
    m = re.search(
        r"(\d{1,4}/\d{4}/[A-ZĐa-zđ]{1,5}[-–][A-Z0-9Đa-zđ]{1,20})", query
    )
    if m:
        return m.group(1)

    # num/type-issuer (no year): 06/CT-UBND
    m = re.search(
        r"(\d{1,4}/[A-ZĐa-zđ]{1,5}[-–][A-Z0-9Đa-zđ]{1,20})", query
    )
    if m:
        return m.group(1)

    # No-slash variant: 06CT-UBND, 06CT – UBND → normalize to 06/CT-UBND
    m = re.search(
        r"(\d{1,4})([A-ZĐ]{1,5})\s*[-–]\s*([A-Z0-9Đ]{1,20})", query
    )
    if m:
        return f"{m.group(1)}/{m.group(2)}-{m.group(3)}"

    return None


def _is_general_query(query: str) -> bool:
    """True when the user wants a summary/overview of a document."""
    return bool(_GENERAL_QUERY_RE.search(query))


def _strip_vn_diacritics(text: str) -> str:
    """Strip Vietnamese diacritics for fuzzy comparison (Đ→D, etc.)."""
    text = text.replace("Đ", "D").replace("đ", "d")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ── DB lookup ─────────────────────────────────────────────

async def lookup_article_from_db(
    db: AsyncSession,
    query: str,
) -> List[Dict]:
    """Look up article/clause content directly from PostgreSQL.

    Three modes:
      1. Specific article: "Điều 7 Luật Di sản văn hóa" → exact article lookup
      2. Doc-number ref: "Tóm tắt 49/2025/QĐ-UBND" → find doc by number,
         return all articles (summary) or search by topic
      3. Topic-based: "Hành vi bị nghiêm cấm trong Luật Di sản văn hóa"
         → search articles by title/content keywords within the named law
    """
    # ── Mode 1: Specific article reference ──
    parsed = parse_article_clause_query(query)
    if parsed:
        result = await _lookup_specific_article(db, parsed)
        if result:
            return result

    # ── Mode 2: Doc-number reference (49/2025/QĐ-UBND, 06/CT-UBND) ──
    doc_num_ref = _extract_doc_number_ref(query)
    if doc_num_ref:
        result = await _lookup_by_doc_number(db, query, doc_num_ref)
        if result:
            return result

    # ── Mode 3: Topic-based search within a named law ──
    doc_name = _extract_document_name(query)
    if doc_name:
        result = await _lookup_topic_in_law(db, query, doc_name)
        if result:
            return result

    return []


async def _lookup_specific_article(
    db: AsyncSession,
    parsed: Dict,
) -> List[Dict]:
    """Mode 1: Look up a specific article by number within a named law."""
    article_num = parsed["article_number"]
    clause_num = parsed.get("clause_number")
    point_letter = parsed.get("point_letter")
    doc_name = parsed.get("doc_name")
    year = parsed.get("year")

    doc_ids = await _find_documents(db, doc_name, year)
    if not doc_ids:
        log.info("[ARTICLE_LOOKUP] No matching documents for name='%s' year=%s", doc_name, year)
        return []

    articles = await _find_articles(db, doc_ids, article_num)
    if not articles:
        log.info("[ARTICLE_LOOKUP] No article '%s' found in %d documents", article_num, len(doc_ids))
        return []

    passages = []
    for art_row in articles:
        if clause_num or point_letter:
            clause_passages = await _get_clause_content(
                db, art_row, clause_num, point_letter
            )
            passages.extend(clause_passages)
        else:
            article_passage = _build_article_passage(art_row)
            passages.append(article_passage)
            clause_passages = await _get_all_clauses(db, art_row)
            passages.extend(clause_passages)

    log.info(
        "[ARTICLE_LOOKUP] Found %d passages for Điều %s%s in '%s'",
        len(passages),
        article_num,
        f", Khoản {clause_num}" if clause_num else "",
        doc_name or "?",
    )
    return passages


async def _lookup_by_doc_number(
    db: AsyncSession,
    query: str,
    doc_num_ref: str,
) -> List[Dict]:
    """Mode 2: Look up content by explicit document number reference.

    For general queries (tóm tắt, nội dung) returns all articles.
    For specific topics, searches articles within the document.
    """
    doc_ids = await _find_document_by_number(db, doc_num_ref)
    if not doc_ids:
        log.info("[DOC_NUM_LOOKUP] No document found for '%s'", doc_num_ref)
        return []

    if _is_general_query(query):
        log.info("[DOC_NUM_LOOKUP] General query → returning all articles for '%s'", doc_num_ref)
        return await _get_all_articles_passages(db, doc_ids)

    topic_kw = _extract_topic_from_doc_query(query, doc_num_ref)
    if topic_kw:
        articles = await _find_articles_by_topic(db, doc_ids, topic_kw)
        if articles:
            passages: List[Dict] = []
            for art_row in articles:
                passages.append(_build_article_passage(art_row))
                passages.extend(await _get_all_clauses(db, art_row))
            log.info(
                "[DOC_NUM_LOOKUP] Topic match (%s) in '%s': %d passages",
                topic_kw, doc_num_ref, len(passages),
            )
            return passages

    log.info("[DOC_NUM_LOOKUP] No topic match → returning all articles for '%s'", doc_num_ref)
    return await _get_all_articles_passages(db, doc_ids)


async def _find_document_by_number(
    db: AsyncSession,
    doc_num: str,
) -> List[int]:
    """Find documents by doc_number (exact → contains → diacritics-stripped)."""
    # 1. Exact match
    stmt = (
        select(Document.id, Document.doc_number)
        .where(Document.doc_number == doc_num)
        .limit(5)
    )
    rows = (await db.execute(stmt)).all()
    if rows:
        log.info("[DOC_NUM_LOOKUP] Exact: '%s' → %s", doc_num, [(r.id, r.doc_number) for r in rows])
        return [r.id for r in rows]

    # 2. Case-insensitive contains
    stmt = (
        select(Document.id, Document.doc_number)
        .where(func.lower(Document.doc_number).contains(doc_num.lower()))
        .limit(5)
    )
    rows = (await db.execute(stmt)).all()
    if rows:
        log.info("[DOC_NUM_LOOKUP] Contains: '%s' → %s", doc_num, [(r.id, r.doc_number) for r in rows])
        return [r.id for r in rows]

    # 3. Diacritics-stripped comparison (QĐ↔QD, NĐ↔ND)
    stripped = _strip_vn_diacritics(doc_num).upper()
    stmt = select(Document.id, Document.doc_number).limit(200)
    all_rows = (await db.execute(stmt)).all()
    matches = [
        r for r in all_rows
        if stripped in _strip_vn_diacritics(r.doc_number).upper()
    ]
    if matches:
        log.info("[DOC_NUM_LOOKUP] Diacritics: '%s' → %s", doc_num, [(r.id, r.doc_number) for r in matches[:5]])
        return [r.id for r in matches[:5]]

    # 4. Numeric prefix fallback (e.g. "49" in "49/2025/QĐ-UBND")
    num_prefix = re.match(r"(\d+)", doc_num)
    if num_prefix:
        prefix = num_prefix.group(1)
        stmt = (
            select(Document.id, Document.doc_number)
            .where(Document.doc_number.like(f"{prefix}/%"))
            .limit(10)
        )
        rows = (await db.execute(stmt)).all()
        if rows:
            log.info("[DOC_NUM_LOOKUP] Prefix '%s/' → %s", prefix, [(r.id, r.doc_number) for r in rows[:5]])
            return [r.id for r in rows]

    return []


async def _get_all_articles_passages(
    db: AsyncSession,
    doc_ids: List[int],
    max_articles: int = 30,
) -> List[Dict]:
    """Return all articles (and their clauses) from the given documents."""
    stmt = (
        select(
            Article.id,
            Article.document_id,
            Article.article_number,
            Article.title,
            Article.content,
            Document.doc_number,
            Document.title.label("document_title"),
            Document.document_type,
        )
        .join(Document, Article.document_id == Document.id)
        .where(Article.document_id.in_(doc_ids))
        .order_by(Article.document_id, Article.id)
        .limit(max_articles)
    )
    result = await db.execute(stmt)
    articles = result.all()

    passages: List[Dict] = []
    for art_row in articles:
        passages.append(_build_article_passage(art_row))
        passages.extend(await _get_all_clauses(db, art_row))

    log.info(
        "[DOC_NUM_LOOKUP] All articles: %d articles → %d passages",
        len(articles), len(passages),
    )
    return passages


def _extract_topic_from_doc_query(query: str, doc_num: str) -> List[str]:
    """Extract topic keywords from a query that references a doc number.

    Strips the doc number, doc type keywords, and common noise words.
    """
    cleaned = query
    cleaned = re.sub(re.escape(doc_num), " ", cleaned)
    cleaned = re.sub(_DOC_TYPE_KW, " ", cleaned, flags=re.IGNORECASE)

    noise = {
        "các", "của", "trong", "theo", "về", "và", "những", "được",
        "có", "cho", "với", "tại", "từ", "đến", "này", "đó",
        "gì", "nào", "thế", "nào", "như", "là", "bị", "được",
        "quy", "định", "nói", "hỏi", "cách", "việt", "nam",
        "năm", "số", "ngày", "tháng",
        "điều", "khoản", "điểm", "luật", "nghị",
        "tóm", "tắt", "nội", "dung", "tổng", "quan",
        "cho", "biết", "giới", "thiệu", "trình", "bày",
        "sau", "trước", "mới",
    }
    words = re.findall(r"[a-zA-ZÀ-ỹĐđ]+", cleaned)
    keywords = [w for w in words if w.lower() not in noise and len(w) > 1]
    return keywords


async def _lookup_topic_in_law(
    db: AsyncSession,
    query: str,
    doc_name: str,
) -> List[Dict]:
    """Mode 2: Search articles by topic keywords within a named law.

    For queries like "Các hành vi bị nghiêm cấm trong Luật Di sản văn hóa",
    finds articles whose title or content matches the topic keywords.
    """
    year = _extract_year(query)
    doc_ids = await _find_documents(db, doc_name, year)
    if not doc_ids:
        log.info("[TOPIC_LOOKUP] No matching documents for name='%s'", doc_name)
        return []

    topic_keywords = _extract_topic_keywords(query, doc_name)
    if not topic_keywords:
        log.info("[TOPIC_LOOKUP] No topic keywords extracted from query")
        return []

    articles = await _find_articles_by_topic(db, doc_ids, topic_keywords)
    if not articles:
        log.info("[TOPIC_LOOKUP] No articles matching topic '%s' in %d docs", topic_keywords, len(doc_ids))
        return []

    passages = []
    for art_row in articles:
        article_passage = _build_article_passage(art_row)
        passages.append(article_passage)
        clause_passages = await _get_all_clauses(db, art_row)
        passages.extend(clause_passages)

    log.info(
        "[TOPIC_LOOKUP] Found %d passages for topic '%s' in '%s'",
        len(passages),
        " ".join(topic_keywords),
        doc_name,
    )
    return passages


async def _find_documents(
    db: AsyncSession,
    doc_name: Optional[str],
    year: Optional[int],
) -> List[int]:
    """Find document IDs matching the given name and/or year."""
    if not doc_name and not year:
        return []

    stmt = select(Document.id, Document.doc_number, Document.title)

    conditions = []
    if doc_name:
        keywords = _tokenize_doc_name(doc_name)
        for kw in keywords:
            conditions.append(
                or_(
                    func.lower(Document.title).contains(kw.lower()),
                    func.lower(Document.doc_number).contains(kw.lower()),
                )
            )

    if year:
        year_str = str(year)
        conditions.append(
            or_(
                Document.doc_number.contains(year_str),
                Document.title.contains(year_str),
                func.extract("year", Document.issued_date) == year,
                func.extract("year", Document.effective_date) == year,
            )
        )

    if conditions:
        stmt = stmt.where(*conditions)

    stmt = stmt.limit(10)
    result = await db.execute(stmt)
    rows = result.all()

    if not rows and doc_name:
        stmt2 = select(Document.id, Document.doc_number, Document.title)
        core_keywords = [kw for kw in _tokenize_doc_name(doc_name) if len(kw) > 2]
        if core_keywords:
            for kw in core_keywords[:3]:
                stmt2 = stmt2.where(
                    or_(
                        func.lower(Document.title).contains(kw.lower()),
                        func.lower(Document.doc_number).contains(kw.lower()),
                    )
                )
            stmt2 = stmt2.limit(10)
            result = await db.execute(stmt2)
            rows = result.all()

    doc_ids = [r.id for r in rows]
    if doc_ids:
        log.info(
            "[ARTICLE_LOOKUP] Found %d documents: %s",
            len(doc_ids),
            [(r.id, r.doc_number[:40]) for r in rows[:5]],
        )
    return doc_ids


def _tokenize_doc_name(doc_name: str) -> List[str]:
    """Split document name into meaningful search tokens."""
    noise = {
        "của", "trong", "theo", "về", "và", "các", "những", "được",
        "có", "cho", "với", "tại", "từ", "đến", "này", "đó",
        "luật", "nghị", "định", "thông", "tư", "quyết", "chỉ", "thị",
        "số", "năm", "ngày", "tháng",
    }
    words = re.findall(r"[a-zA-ZÀ-ỹĐđ]+", doc_name)
    return [w for w in words if w.lower() not in noise and len(w) > 1]


def _extract_topic_keywords(query: str, doc_name: str) -> List[str]:
    """Extract topic keywords from query, excluding the document name portion."""
    doc_name_lower = doc_name.lower()
    query_lower = query.lower()

    topic_part = query_lower
    idx = topic_part.find(doc_name_lower)
    if idx >= 0:
        before = topic_part[:idx].strip()
        after = topic_part[idx + len(doc_name_lower):].strip()
        topic_part = f"{before} {after}".strip()

    noise = {
        "các", "của", "trong", "theo", "về", "và", "những", "được",
        "có", "cho", "với", "tại", "từ", "đến", "này", "đó",
        "gì", "nào", "thế", "nào", "như", "là", "bị", "được",
        "quy", "định", "nói", "hỏi", "cách", "việt", "nam",
        "năm", "số", "ngày", "tháng",
        "điều", "khoản", "điểm", "luật", "nghị",
    }
    words = re.findall(r"[a-zA-ZÀ-ỹĐđ]+", topic_part)
    keywords = [w for w in words if w.lower() not in noise and len(w) > 1]
    return keywords


async def _find_articles_by_topic(
    db: AsyncSession,
    doc_ids: List[int],
    topic_keywords: List[str],
) -> List:
    """Find articles whose title or content matches topic keywords."""
    if not topic_keywords:
        return []

    base_stmt = (
        select(
            Article.id,
            Article.document_id,
            Article.article_number,
            Article.title,
            Article.content,
            Document.doc_number,
            Document.title.label("document_title"),
            Document.document_type,
        )
        .join(Document, Article.document_id == Document.id)
        .where(Article.document_id.in_(doc_ids))
    )

    # First: try matching all keywords in article title
    title_conditions = []
    for kw in topic_keywords:
        title_conditions.append(func.lower(Article.title).contains(kw.lower()))

    stmt = base_stmt.where(*title_conditions).limit(10)
    result = await db.execute(stmt)
    rows = result.all()

    if rows:
        log.info("[TOPIC_LOOKUP] Title match found %d articles", len(rows))
        return rows

    # Second: try matching any keyword in title
    any_title_conditions = []
    for kw in topic_keywords:
        any_title_conditions.append(func.lower(Article.title).contains(kw.lower()))

    stmt = base_stmt.where(or_(*any_title_conditions)).limit(10)
    result = await db.execute(stmt)
    rows = result.all()

    if rows:
        log.info("[TOPIC_LOOKUP] Partial title match found %d articles", len(rows))
        return rows

    # Third: search in article content (broader, slower)
    content_conditions = []
    for kw in topic_keywords:
        content_conditions.append(func.lower(Article.content).contains(kw.lower()))

    stmt = base_stmt.where(*content_conditions).limit(5)
    result = await db.execute(stmt)
    rows = result.all()

    if rows:
        log.info("[TOPIC_LOOKUP] Content match found %d articles", len(rows))
    return rows


async def _find_articles(
    db: AsyncSession,
    doc_ids: List[int],
    article_number: str,
) -> List:
    """Find articles matching the article number in given documents."""
    norm_num = article_number.strip()

    stmt = (
        select(
            Article.id,
            Article.document_id,
            Article.article_number,
            Article.title,
            Article.content,
            Document.doc_number,
            Document.title.label("document_title"),
            Document.document_type,
        )
        .join(Document, Article.document_id == Document.id)
        .where(Article.document_id.in_(doc_ids))
        .where(
            or_(
                Article.article_number == f"Điều {norm_num}",
                Article.article_number == norm_num,
                Article.article_number == f"điều {norm_num}",
            )
        )
    )
    result = await db.execute(stmt)
    return result.all()


async def _get_clause_content(
    db: AsyncSession,
    art_row,
    clause_number: Optional[str],
    point_letter: Optional[str],
) -> List[Dict]:
    """Get specific clause/point content from an article."""
    passages = []
    conditions = [Clause.article_id == art_row.id]

    if clause_number:
        norm_clause = clause_number.strip()
        conditions.append(
            or_(
                Clause.clause_number == f"Khoản {norm_clause}",
                Clause.clause_number == norm_clause,
            )
        )

    if point_letter:
        conditions.append(
            or_(
                Clause.clause_number == f"Điểm {point_letter}",
                Clause.clause_number == point_letter,
            )
        )

    stmt = select(Clause.id, Clause.clause_number, Clause.content).where(*conditions)
    result = await db.execute(stmt)
    rows = result.all()

    article_num = _normalize_article_num(art_row.article_number)

    for row in rows:
        text_parts = [
            (art_row.document_title or "").strip(),
            f"Điều {article_num}",
        ]
        if art_row.title:
            text_parts[-1] += f". {art_row.title}"
        text_parts.append(f"{row.clause_number}")
        text_parts.append(row.content)

        passages.append({
            "text_chunk": "\n".join(text_parts),
            "document_id": art_row.document_id,
            "article_id": art_row.id,
            "clause_id": row.id,
            "doc_number": art_row.doc_number,
            "document_title": art_row.document_title or "",
            "article_number": article_num,
            "article_title": art_row.title or "",
            "clause_number": _normalize_clause_display(row.clause_number),
            "score": 1.0,
            "rerank_score": 1.0,
            "_db_lookup": True,
        })

    if not rows:
        passages.append(_build_article_passage(art_row))

    return passages


async def _get_all_clauses(db: AsyncSession, art_row) -> List[Dict]:
    """Get all clauses for an article."""
    stmt = (
        select(Clause.id, Clause.clause_number, Clause.content)
        .where(Clause.article_id == art_row.id)
        .order_by(Clause.id)
    )
    result = await db.execute(stmt)
    rows = result.all()

    article_num = _normalize_article_num(art_row.article_number)
    passages = []
    for row in rows:
        text_parts = [
            (art_row.document_title or "").strip(),
            f"Điều {article_num}",
        ]
        if art_row.title:
            text_parts[-1] += f". {art_row.title}"
        text_parts.append(f"{row.clause_number}")
        text_parts.append(row.content)

        passages.append({
            "text_chunk": "\n".join(text_parts),
            "document_id": art_row.document_id,
            "article_id": art_row.id,
            "clause_id": row.id,
            "doc_number": art_row.doc_number,
            "document_title": art_row.document_title or "",
            "article_number": article_num,
            "article_title": art_row.title or "",
            "clause_number": _normalize_clause_display(row.clause_number),
            "score": 0.95,
            "rerank_score": 0.95,
            "_db_lookup": True,
        })
    return passages


def _build_article_passage(art_row) -> Dict:
    """Build a passage dict from an article DB row."""
    article_num = _normalize_article_num(art_row.article_number)
    text_parts = [
        (art_row.document_title or "").strip(),
        f"Điều {article_num}",
    ]
    if art_row.title:
        text_parts[-1] += f". {art_row.title}"
    text_parts.append(art_row.content)

    return {
        "text_chunk": "\n".join(text_parts),
        "document_id": art_row.document_id,
        "article_id": art_row.id,
        "clause_id": None,
        "doc_number": art_row.doc_number,
        "document_title": art_row.document_title or "",
        "article_number": article_num,
        "article_title": art_row.title or "",
        "clause_number": None,
        "score": 1.0,
        "rerank_score": 1.0,
        "_db_lookup": True,
    }


def _normalize_article_num(article_number: str) -> str:
    m = re.search(r"(\d+[a-zA-Z]?)", str(article_number or ""))
    return m.group(1) if m else str(article_number).strip()


def _normalize_clause_display(clause_number: str) -> Optional[str]:
    m = re.search(r"(\d+[a-zA-Z]?)", str(clause_number or ""))
    return m.group(1) if m else clause_number
