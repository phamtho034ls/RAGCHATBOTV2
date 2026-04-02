"""Hybrid retriever with article-aware legal retrieval."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import (
    MULTI_ARTICLE_MAX_ARTICLES,
    RAG_AMENDMENT_FULL_DOC_EXPAND,
    RAG_AMENDMENT_MAX_ARTICLES,
    RAG_AMENDMENT_MAX_CHUNKS,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    TOPIC_MISMATCH_PENALTY,
    TOPIC_MISMATCH_QUERY_CONF_MIN,
)
from app.services.domain_classifier import classify_query_domain
from app.database.models import VectorChunk, Document, Article, Clause, Chapter, Section
from app.retrieval.article_lookup import lookup_article_from_db
from app.retrieval.keyword_retriever import keyword_search
from app.retrieval.reranker import rerank
from app.retrieval.vector_retriever import vector_search
from app.retrieval.article_selection import diversify_by_article, dynamic_max_articles

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Văn bản sửa đổi / bổ sung — lấy đủ Điều trong DB
# ---------------------------------------------------------------------------

_AMEND_DOC_TITLE_RE = re.compile(
    r"sửa\s*đổi\s*,?\s*bổ\s*sung|bổ\s*sung\s*,?\s*sửa\s*đổi|"
    r"một\s*số\s*điều\s*(?:của|các\s*luật|các\s*nghị\s*định)|"
    r"một\s*số\s*điều\s*khoản\s*của|"
    r"(?:luật|nghị\s*định|thông\s*tư|quyết\s*định)\s+"
    r"(?:số\s*)?(?:sửa\s*đổi|sửa\s*đổi\s*,?\s*bổ\s*sung|bổ\s*sung)|"
    r"đính\s*chính",
    re.IGNORECASE,
)

_AMEND_QUERY_HINT_RE = re.compile(
    r"sửa\s*đổi|bổ\s*sung|thay\s*thế|đính\s*chính|"
    r"những\s*điều\s*(?:được\s*)?(?:sửa|bổ\s*sung|thay)|"
    r"điều\s*nào\s*(?:được\s*)?(?:sửa|bổ\s*sung|thay)|"
    r"nội\s*dung\s*(?:các\s*)?(?:thay\s*đổi|sửa\s*đổi)",
    re.IGNORECASE,
)


def _passage_or_doc_title_hints_amendment(p: Dict) -> bool:
    t = f"{p.get('document_title') or ''} {p.get('doc_number') or ''}"
    return bool(_AMEND_DOC_TITLE_RE.search(t))


def _document_title_str_hints_amendment(title: Optional[str]) -> bool:
    return bool(_AMEND_DOC_TITLE_RE.search(title or ""))


def _query_hints_amendment_scope(query: str) -> bool:
    return bool(_AMEND_QUERY_HINT_RE.search(query or ""))


async def _list_article_ids_for_documents(
    db: AsyncSession,
    doc_ids: List[int],
    max_articles: int,
) -> List[int]:
    """Chia quota đều mỗi văn bản để không một luật sửa đổi nuốt hết limit."""
    if not doc_ids or max_articles <= 0:
        return []
    uniq = list(dict.fromkeys(int(d) for d in doc_ids))
    per_doc = max(1, max_articles // len(uniq))
    all_ids: List[int] = []
    for did in uniq:
        stmt = (
            select(Article.id)
            .where(Article.document_id == did)
            .order_by(Article.id)
            .limit(per_doc)
        )
        all_ids.extend([r[0] for r in (await db.execute(stmt)).all()])
    if len(all_ids) > max_articles:
        all_ids = all_ids[:max_articles]
    return all_ids


def _synthetic_chunks_from_article_rows(
    rows: List[Any],
    base_rerank: float,
) -> Dict[int, List[Dict]]:
    """Một passage/điều khi chưa có VectorChunk."""
    out: Dict[int, List[Dict]] = {}
    for row in rows:
        aid = int(row.id)
        anum = _normalize_article_number(row.article_number) or str(row.article_number or "").strip()
        title = (row.title or "").strip()
        dn = row.doc_number or ""
        dtitle = (getattr(row, "document_title", None) or "").strip()
        header = f"{dtitle}\nĐiều {anum}"
        if title:
            header += f". {title}"
        text_chunk = f"{header}\n{(row.content or '').strip()}"
        out[aid] = [{
            "id": f"db_{aid}",
            "score": base_rerank,
            "text_chunk": text_chunk,
            "document_id": row.document_id,
            "article_id": aid,
            "clause_id": None,
            "doc_number": dn,
            "document_title": dtitle,
            "article_number": anum,
            "article_title": title,
            "clause_number": None,
            "chapter": "",
            "section": "",
            "rerank_score": base_rerank,
            "_full_article": True,
            "_db_lookup": True,
        }]
    return out


async def _flatten_chunks_for_article_ids(
    db: AsyncSession,
    article_ids: List[int],
    base_rerank: float,
) -> List[Dict]:
    if not article_ids:
        return []
    chunk_map = await _fetch_full_article_chunks(db, article_ids)
    missing = [aid for aid in article_ids if not chunk_map.get(aid)]
    if missing:
        stmt = (
            select(
                Article.id,
                Article.document_id,
                Article.article_number,
                Article.title,
                Article.content,
                Document.doc_number,
                Document.title.label("document_title"),
            )
            .join(Document, Article.document_id == Document.id)
            .where(Article.id.in_(missing))
        )
        synth_rows = (await db.execute(stmt)).all()
        chunk_map.update(_synthetic_chunks_from_article_rows(synth_rows, base_rerank))
    passages: List[Dict] = []
    for aid in article_ids:
        for ch in chunk_map.get(aid) or []:
            ch.setdefault("rerank_score", base_rerank)
            passages.append(ch)
    return passages


async def _fetch_all_passages_for_documents(
    db: AsyncSession,
    doc_ids: List[int],
    max_articles: int,
    base_rerank: float,
) -> List[Dict]:
    article_ids = await _list_article_ids_for_documents(db, doc_ids, max_articles)
    return await _flatten_chunks_for_article_ids(db, article_ids, base_rerank)


def _collect_amendment_expand_document_ids(
    passages: List[Dict],
    query: str,
    scan_limit: int = 40,
) -> List[int]:
    out: List[int] = []
    seen: set = set()
    q_amend = _query_hints_amendment_scope(query)
    for p in passages[:scan_limit]:
        if not (q_amend or _passage_or_doc_title_hints_amendment(p)):
            continue
        did = p.get("document_id")
        if did is None:
            continue
        try:
            i = int(did)
        except (TypeError, ValueError):
            continue
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


async def _merge_amendment_expanded_passages(
    db: AsyncSession,
    snapshot_before_collapse: List[Dict],
    collapsed: List[Dict],
    query: str,
) -> List[Dict]:
    if not RAG_AMENDMENT_FULL_DOC_EXPAND or not snapshot_before_collapse:
        return collapsed
    doc_ids = _collect_amendment_expand_document_ids(snapshot_before_collapse, query)
    if not doc_ids:
        return collapsed
    max_score = max(
        (
            float(p.get("rerank_score", p.get("rrf_score", p.get("score", 0.0))))
            for p in snapshot_before_collapse
        ),
        default=0.55,
    )
    base = max_score * 0.98
    expanded = await _fetch_all_passages_for_documents(
        db, doc_ids, RAG_AMENDMENT_MAX_ARTICLES, base
    )
    if not expanded:
        return collapsed
    doc_set = set(doc_ids)
    tail = [p for p in collapsed if p.get("document_id") not in doc_set]
    seen_article: set = set()
    merged: List[Dict] = []
    for p in expanded:
        aid = p.get("article_id")
        if aid:
            if aid in seen_article:
                continue
            seen_article.add(aid)
        merged.append(p)
    for p in tail:
        aid = p.get("article_id")
        if aid and aid in seen_article:
            continue
        merged.append(p)
        if aid:
            seen_article.add(aid)
    cap = max(RAG_AMENDMENT_MAX_CHUNKS, 1)
    if len(merged) > cap:
        merged = merged[:cap]
    log.info(
        "Amendment expand: doc_ids=%s expanded=%d merged=%d",
        doc_ids,
        len(expanded),
        len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Helpers: doc/article reference extraction
# ---------------------------------------------------------------------------

def _strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics: NĐ→ND, QĐ→QD, etc."""
    text = text.replace("Đ", "D").replace("đ", "d")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_doc_ref(doc_ref: str) -> str:
    """Normalize doc number for comparison: strip diacritics, uppercase."""
    if not doc_ref:
        return ""
    return _strip_diacritics(doc_ref).upper()


def _extract_article_reference(query: str) -> Optional[str]:
    """Extract explicit article number from query (e.g. 'Điều 7')."""
    m = re.search(r"điều\s+(\d+[a-zA-Z]?)", query, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_doc_reference(query: str) -> Optional[str]:
    """Extract legal document number reference like 144/2021/NĐ-CP, 06/CT-UBND."""
    patterns = [
        # num/year/type-issuer: 49/2025/QĐ-UBND
        r"(\d+/\d{4}/[A-ZĐa-zđ]{1,5}[-–][A-Z0-9Đa-zđ]{1,20})",
        # num/type-issuer (no year): 06/CT-UBND
        r"(\d+/[A-ZĐa-zđ]{1,5}[-–][A-Z0-9Đa-zđ]{1,20})",
        # type keyword + num/year
        r"(?:nghị\s*định|thông\s*tư|quyết\s*định|chỉ\s*thị|luật)\s+(?:số\s+)?(\d+/\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, query or "", re.IGNORECASE)
        if m:
            return m.group(1)

    # No-slash variant: 06CT-UBND, 06CT – UBND
    m = re.search(r"(\d{1,4})([A-ZĐ]{1,5})\s*[-–]\s*([A-Z0-9Đ]{1,20})", query or "")
    if m:
        return f"{m.group(1)}/{m.group(2)}-{m.group(3)}"

    return None


def _doc_ref_to_search_variants(raw_ref: str) -> List[str]:
    """Generate search variants from a doc reference.

    '45/2024/QH15' → ['45/2024/QH15', '45_2024_QH15', '45/2024/qh15', ...]
    Handles slash↔underscore and diacritics (NĐ↔ND).
    """
    normalized = _normalize_doc_ref(raw_ref)
    slash_form = normalized.replace("_", "/")
    underscore_form = normalized.replace("/", "_")
    parts = re.split(r"[/_]", normalized)
    return list(dict.fromkeys([
        raw_ref,
        slash_form,
        underscore_form,
        raw_ref.replace("/", "_"),
        raw_ref.replace("_", "/"),
        "_".join(parts) if len(parts) >= 3 else "",
    ]))


async def _resolve_doc_number(db: AsyncSession, raw_ref: str) -> Optional[str]:
    """Resolve a user-typed doc reference to the actual stored doc_number.

    Handles diacritics mismatch (NĐ-CP vs ND-CP), slash↔underscore,
    and trailing numeric IDs in filenames (e.g. 45_2024_QH15_583769).
    """
    if not raw_ref:
        return None

    # 1) Exact match
    stmt = select(Document.doc_number).where(Document.doc_number == raw_ref).limit(1)
    row = (await db.execute(stmt)).scalar()
    if row:
        return row

    # 2) Try all format variants (slash↔underscore, diacritics-stripped)
    for variant in _doc_ref_to_search_variants(raw_ref):
        if not variant:
            continue
        stmt = select(Document.doc_number).where(
            func.upper(Document.doc_number).like(f"%{variant.upper()}%")
        ).limit(1)
        row = (await db.execute(stmt)).scalar()
        if row:
            log.info("[RESOLVE] '%s' → '%s' (variant '%s')", raw_ref, row, variant)
            return row

    # 3) Numeric prefix match (e.g. "45/2024" matches "45/2024/QH15" or "45_2024_QH15_583769")
    m = re.match(r"(\d+)[/_](\d{4})", raw_ref)
    if m:
        num, year = m.group(1), m.group(2)
        for sep in ["/", "_"]:
            prefix = f"{num}{sep}{year}"
            stmt = (
                select(Document.doc_number)
                .where(Document.doc_number.like(f"{prefix}%"))
                .order_by(Document.doc_number)
                .limit(24)
            )
            rows = list((await db.execute(stmt)).scalars().all())
            if not rows:
                continue
            raw_norm = _normalize_doc_ref(raw_ref)
            for cand in rows:
                if _normalize_doc_ref(cand) == raw_norm or raw_ref in cand or cand in raw_ref:
                    log.info("[RESOLVE] '%s' → '%s' (prefix '%s')", raw_ref, cand, prefix)
                    return cand
            # Một số DB có nhiều bản ghi cùng prefix — chọn bản có độ dài ổn định, tên khớp raw_ref hơn
            best = min(
                rows,
                key=lambda d: (
                    0 if raw_norm and raw_norm in _normalize_doc_ref(d) else 1,
                    abs(len(d) - len(raw_ref)),
                    d,
                ),
            )
            log.info("[RESOLVE] '%s' → '%s' (prefix '%s', disambiguated)", raw_ref, best, prefix)
            return best

    log.warning("[RESOLVE] Could not resolve doc_number '%s'", raw_ref)
    return raw_ref


def _normalize_article_number(article_number: Optional[str]) -> Optional[str]:
    if not article_number:
        return None
    m = re.search(r"(\d+[a-zA-Z]?)", str(article_number))
    return m.group(1) if m else str(article_number).strip()


def _normalize_clause_number(clause_number: Optional[str]) -> Optional[str]:
    if not clause_number:
        return None
    m = re.search(r"(\d+[a-zA-Z]?)", str(clause_number))
    return m.group(1) if m else str(clause_number).strip()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", (text or "").lower()))


def _title_similarity(query: str, title: str) -> float:
    q = _tokenize(query)
    t = _tokenize(title)
    if not q or not t:
        return 0.0
    inter = len(q & t)
    denom = max(len(t), 1)
    return inter / denom


def _keyword_overlap(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0
    return len(q & t) / max(len(q), 1)


def _doc_reference_match(doc_ref: Optional[str], item: Dict) -> bool:
    if not doc_ref:
        return False
    ref_norm = _normalize_doc_ref(doc_ref)
    doc_number_norm = _normalize_doc_ref(item.get("doc_number") or "")
    doc_title_norm = _normalize_doc_ref(item.get("document_title") or "")
    return ref_norm in doc_number_norm or ref_norm in doc_title_norm


async def _fetch_full_article_chunks(
    db: AsyncSession,
    article_ids: List[int],
) -> Dict[int, List[Dict]]:
    """Fetch all chunks for selected articles with complete metadata."""
    if not article_ids:
        return {}

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
            Document.effective_date,
            Document.issued_date,
            Article.article_number,
            Article.title.label("article_title"),
            Clause.clause_number,
            Chapter.chapter_number,
            Chapter.title.label("chapter_title"),
            Section.section_number,
            Section.title.label("section_title"),
        )
        .join(Document, VectorChunk.document_id == Document.id)
        .join(Article, VectorChunk.article_id == Article.id)
        .outerjoin(Clause, VectorChunk.clause_id == Clause.id)
        .outerjoin(Chapter, Article.chapter_id == Chapter.id)
        .outerjoin(Section, Article.section_id == Section.id)
        .where(VectorChunk.article_id.in_(article_ids))
        .order_by(VectorChunk.article_id, VectorChunk.id)
    )
    result = await db.execute(stmt)
    rows = result.all()

    def _chapter_label(cnum, ctitle):
        if not cnum:
            return ""
        return f"Chương {cnum} - {ctitle}" if (ctitle and ctitle.strip()) else f"Chương {cnum}"

    def _section_label(snum, stitle):
        if not snum:
            return ""
        return f"Mục {snum} - {stitle}" if (stitle and stitle.strip()) else f"Mục {snum}"

    def _fmt_date(d) -> str:
        if d is None:
            return ""
        if hasattr(d, "strftime"):
            return d.strftime("%d/%m/%Y")
        return str(d)

    article_chunks: Dict[int, List[Dict]] = {}
    for row in rows:
        chunk = {
            "id": str(row.vector_id or row.id),
            "score": 0.0,
            "text_chunk": row.chunk_text,
            "document_id": row.document_id,
            "article_id": row.article_id,
            "clause_id": row.clause_id,
            "doc_number": row.doc_number,
            "document_title": row.document_title or "",
            "article_number": _normalize_article_number(row.article_number),
            "article_title": row.article_title or "",
            "clause_number": _normalize_clause_number(row.clause_number),
            "chapter": _chapter_label(getattr(row, "chapter_number", None), getattr(row, "chapter_title", None) or ""),
            "section": _section_label(getattr(row, "section_number", None), getattr(row, "section_title", None) or ""),
            "_full_article": True,
        }
        if getattr(row, "effective_date", None) is not None:
            chunk["effective_date"] = _fmt_date(row.effective_date)
        if getattr(row, "issued_date", None) is not None:
            chunk["issued_date"] = _fmt_date(row.issued_date)
        article_chunks.setdefault(row.article_id, []).append(chunk)

    return article_chunks


def _item_matches_domain_filter(item: Dict, allowed: set[str]) -> bool:
    """Khớp legal_domain hoặc bất kỳ nhãn nào trong law_intents (sau enrich)."""
    dom = item.get("legal_domain")
    if dom and str(dom) in allowed:
        return True
    raw = item.get("law_intents")
    if isinstance(raw, list):
        labels = {str(x) for x in raw if x and str(x) != "chung"}
        if allowed & labels:
            return True
    return False


def _doc_topic_labels(item: Dict) -> set:
    """Domains associated with a passage (Qdrant payload + DB fallback)."""
    raw = item.get("law_intents")
    if isinstance(raw, list) and raw:
        return {str(x) for x in raw if x and str(x) != "chung"}
    dom = item.get("legal_domain")
    if dom and dom != "chung":
        return {str(dom)}
    return set()


def _apply_topic_mismatch_penalty(query: str, reranked: List[Dict]) -> None:
    """Lower rerank_score when query domains (from classifier) do not overlap document tags."""
    if not query or not query.strip() or not reranked:
        return
    q_hits = classify_query_domain(query.strip(), top_n=4)
    q_set = {
        d["domain"]
        for d in q_hits
        if d.get("domain") and d["domain"] != "chung"
        and float(d.get("confidence") or 0) >= TOPIC_MISMATCH_QUERY_CONF_MIN
    }
    if not q_set:
        return

    n = 0
    for item in reranked:
        doc_set = _doc_topic_labels(item)
        if not doc_set:
            continue
        if q_set & doc_set:
            continue
        prev = float(item.get("rerank_score", 0.0))
        item["rerank_score"] = float(prev - TOPIC_MISMATCH_PENALTY)
        item["topic_mismatch_penalty"] = True
        n += 1
    if n:
        reranked.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
        log.info(
            "Topic mismatch penalty applied to %d/%d passages (query_domains=%s, penalty=%.2f)",
            n, len(reranked), list(q_set), TOPIC_MISMATCH_PENALTY,
        )


async def _enrich_missing_metadata(db: AsyncSession, items: List[Dict]) -> None:
    """Fill article/document metadata for old vectors missing payload fields."""
    article_ids = {item.get("article_id") for item in items if item.get("article_id")}
    if not article_ids:
        return

    def _article_enrich_select(include_law_intents: bool):
        cols = [
            Article.id.label("article_id"),
            Article.article_number,
            Article.title.label("article_title"),
            Document.id.label("document_id"),
            Document.doc_number,
            Document.title.label("document_title"),
            Document.effective_date,
            Document.issued_date,
            Chapter.chapter_number,
            Chapter.title.label("chapter_title"),
            Section.section_number,
            Section.title.label("section_title"),
        ]
        if include_law_intents:
            cols.insert(8, Document.law_intents)
        return (
            select(*cols)
            .join(Document, Article.document_id == Document.id)
            .outerjoin(Chapter, Article.chapter_id == Chapter.id)
            .outerjoin(Section, Article.section_id == Section.id)
            .where(Article.id.in_(article_ids))
        )

    stmt = _article_enrich_select(include_law_intents=True)
    try:
        result = await db.execute(stmt)
    except ProgrammingError as exc:
        if "law_intents" not in str(getattr(exc, "orig", None) or exc):
            raise
        log.warning(
            "DB missing documents.law_intents — run: cd backend && alembic upgrade head. "
            "Continuing without law_intents from DB.",
        )
        # PG aborts the transaction on error; must rollback before any further SQL on this session.
        await db.rollback()
        result = await db.execute(_article_enrich_select(include_law_intents=False))
    mapping = {row.article_id: row for row in result.all()}

    def _fmt_date(d) -> str:
        if d is None:
            return ""
        if hasattr(d, "strftime"):
            return d.strftime("%d/%m/%Y")
        return str(d)

    for item in items:
        aid = item.get("article_id")
        row = mapping.get(aid)
        if not row:
            continue
        item.setdefault("document_id", row.document_id)
        item["article_number"] = _normalize_article_number(
            item.get("article_number") or row.article_number
        )
        item["article_title"] = item.get("article_title") or row.article_title or ""
        item["doc_number"] = item.get("doc_number") or row.doc_number or ""
        item["document_title"] = item.get("document_title") or row.document_title or ""
        if item.get("law_intents") is None and getattr(row, "law_intents", None) is not None:
            item["law_intents"] = row.law_intents
        if getattr(row, "effective_date", None) is not None:
            item["effective_date"] = _fmt_date(row.effective_date)
        if getattr(row, "issued_date", None) is not None:
            item["issued_date"] = _fmt_date(row.issued_date)
        item["clause_number"] = _normalize_clause_number(item.get("clause_number"))
        cnum = getattr(row, "chapter_number", None)
        ctitle = getattr(row, "chapter_title", None) or ""
        if cnum:
            item.setdefault("chapter", f"Chương {cnum} - {ctitle}" if ctitle.strip() else f"Chương {cnum}")
        snum = getattr(row, "section_number", None)
        stitle = getattr(row, "section_title", None) or ""
        if snum:
            item.setdefault("section", f"Mục {snum} - {stitle}" if stitle.strip() else f"Mục {snum}")

    doc_ids = {
        item.get("document_id")
        for item in items
        if item.get("document_id") and item.get("law_intents") is None
    }
    if doc_ids:
        stmt_doc = select(Document.id, Document.law_intents).where(Document.id.in_(doc_ids))
        try:
            by_doc = {r.id: r.law_intents for r in (await db.execute(stmt_doc)).all()}
        except ProgrammingError as exc:
            if "law_intents" not in str(getattr(exc, "orig", None) or exc):
                raise
            await db.rollback()
            by_doc = {}
        for item in items:
            did = item.get("document_id")
            if did and item.get("law_intents") is None:
                li = by_doc.get(did)
                if li:
                    item["law_intents"] = li


def _group_key(item: Dict) -> Optional[Tuple[Optional[int], str]]:
    article_number = _normalize_article_number(item.get("article_number"))
    if not article_number:
        return None
    return item.get("document_id"), article_number


def _score_and_group_articles(
    query: str,
    items: List[Dict],
    explicit_article: Optional[str],
    explicit_doc_ref: Optional[str],
) -> Dict[Tuple[Optional[int], str], float]:
    grouped: Dict[Tuple[Optional[int], str], float] = defaultdict(float)
    for item in items:
        key = _group_key(item)
        if not key:
            continue
        semantic = float(item.get("rerank_score", item.get("rrf_score", item.get("score", 0.0))))
        title_sim = _title_similarity(query, item.get("article_title", ""))
        text_overlap = _keyword_overlap(query, item.get("text_chunk", ""))
        number_bonus = 5.0 if explicit_article and key[1].lower() == explicit_article.lower() else 0.0
        doc_bonus = 5.0 if _doc_reference_match(explicit_doc_ref, item) else 0.0
        # Token sub-chunks get a content-match boost because they are
        # smaller and semantically tighter than full-clause chunks.
        chunk_type = item.get("chunk_type", "clause")
        content_boost = 1.5 * text_overlap if chunk_type == "token_sub" else 0.0
        item["article_match_score"] = (
            semantic + (0.8 * title_sim) + (0.5 * text_overlap)
            + number_bonus + doc_bonus + content_boost
        )
        grouped[key] += item["article_match_score"]
    return grouped


def _select_best_article(group_scores: Dict[Tuple[Optional[int], str], float]) -> Optional[Tuple[Optional[int], str]]:
    if not group_scores:
        return None
    return max(group_scores, key=group_scores.get)


def _select_top_n_articles(
    group_scores: Dict[Tuple[Optional[int], str], float],
    n: int,
) -> List[Tuple[Optional[int], str]]:
    """Return top N article groups by score (desc)."""
    if not group_scores or n <= 0:
        return []
    sorted_groups = sorted(group_scores.keys(), key=lambda g: group_scores[g], reverse=True)
    return sorted_groups[:n]


_GENERAL_QUERY_RE = re.compile(
    r"(?:tóm\s*tắt|nội\s*dung|tổng\s*quan|cho\s*biết|giới\s*thiệu"
    r"|trình\s*bày|liệt\s*kê|bao\s*gồm\s*(?:những|các)\s*gì)",
    re.IGNORECASE,
)

_MULTI_ARTICLE_QUERY_RE = re.compile(
    r"(?:"
    r"\bngành,?\s*nghề\s+đầu\s+tư\s+kinh\s+doanh\s+có\s+điều\s+kiện\b"
    r"|\bdanh\s*mục\b"
    r"|\bliệt\s*kê\b"
    r"|\bbao\s*gồm\b"
    r"|\bcác\s+điều\b"
    r"|\bnhững\s+điều\b"
    r")",
    re.IGNORECASE,
)


def _query_prefers_multi_article(query: str) -> bool:
    return bool(_MULTI_ARTICLE_QUERY_RE.search(query or ""))


def _unique_article_count(items: List[Dict]) -> int:
    return len({i.get("article_id") for i in items if i.get("article_id")})


async def _direct_article_lookup(
    db: AsyncSession,
    resolved_doc_number: str,
    query: str,
    explicit_article: Optional[str] = None,
) -> List[Dict]:
    """Precision retrieval: find the best article inside a known document.

    Used when the user explicitly references a document number. Looks up
    articles by title similarity or explicit article number, then returns
    all chunks from the best-matching article.
    For summary queries (tóm tắt, nội dung), returns all articles.
    Văn bản sửa đổi/bổ sung (theo tiêu đề) → trả về tất cả Điều (giới hạn cấu hình).
    """
    stmt = (
        select(
            Article.id,
            Article.article_number,
            Article.title,
            Document.id.label("document_id"),
            Document.title.label("document_title"),
        )
        .join(Document, Article.document_id == Document.id)
        .where(Document.doc_number == resolved_doc_number)
    )
    rows = (await db.execute(stmt)).all()
    if not rows:
        for variant in _doc_ref_to_search_variants(resolved_doc_number):
            if not variant or len(variant) < 6:
                continue
            stmt_f = (
                select(
                    Article.id,
                    Article.article_number,
                    Article.title,
                    Document.id.label("document_id"),
                    Document.title.label("document_title"),
                )
                .join(Document, Article.document_id == Document.id)
                .where(func.upper(Document.doc_number).like(f"%{variant.upper()}%"))
                .limit(400)
            )
            rows = (await db.execute(stmt_f)).all()
            if rows:
                dids = {getattr(r, "document_id", None) for r in rows}
                if len(dids) == 1:
                    log.info(
                        "[DIRECT] Fuzzy doc_number match (variant=%s) → %d articles",
                        variant,
                        len(rows),
                    )
                    break
                rows = []
        if not rows:
            return []

    doc_title = (getattr(rows[0], "document_title", None) or "") or ""
    if (
        RAG_AMENDMENT_FULL_DOC_EXPAND
        and _document_title_str_hints_amendment(doc_title)
    ):
        lim = min(len(rows), RAG_AMENDMENT_MAX_ARTICLES)
        all_ids = [r.id for r in rows[:lim]]
        log.info(
            "[DIRECT] Văn bản sửa đổi/bổ sung (tiêu đề) → %d điều cho '%s'",
            len(all_ids),
            resolved_doc_number,
        )
        return await _flatten_chunks_for_article_ids(db, all_ids, 0.88)

    if explicit_article:
        for row in rows:
            norm = _normalize_article_number(row.article_number)
            if norm and norm.lower() == explicit_article.lower():
                chunks = await _fetch_full_article_chunks(db, [row.id])
                return chunks.get(row.id, [])

    scored = []
    for row in rows:
        sim = _title_similarity(query, row.title or "")
        scored.append((row, sim))
    scored.sort(key=lambda x: x[1], reverse=True)

    if scored and scored[0][1] > 0.3:
        best = scored[0][0]
        log.info("[DIRECT] Best article match: Điều %s '%s' (sim=%.2f)",
                 best.article_number, best.title, scored[0][1])
        chunks = await _fetch_full_article_chunks(db, [best.id])
        return chunks.get(best.id, [])

    # For summary/general queries, return all articles from the document
    if _GENERAL_QUERY_RE.search(query):
        all_ids = [r.id for r in rows[:30]]
        log.info("[DIRECT] General query → returning all %d articles for '%s'",
                 len(all_ids), resolved_doc_number)
        all_chunks_map = await _fetch_full_article_chunks(db, all_ids)
        result: List[Dict] = []
        for chunks in all_chunks_map.values():
            result.extend(chunks)
        return result

    # Câu hỏi tình huống dài (tiếng ồn, karaoke, chỉ đạo xử lý…) — tiêu đề Điều hiếm khi đạt ngưỡng
    # title_similarity; vẫn cần ngữ cảnh từ đúng văn bản được trích dẫn.
    doc_id = getattr(rows[0], "document_id", None)
    if doc_id is None:
        return []

    stmt_ov = (
        select(Article.id, Article.title, Article.content)
        .where(Article.document_id == doc_id)
    )
    all_arts = (await db.execute(stmt_ov)).all()
    ov_scored: List[tuple[float, int]] = []
    for ar in all_arts:
        blob = f"{(ar.title or '').strip()}\n{(ar.content or '')[:4500]}"
        ov_scored.append((_keyword_overlap(query, blob), ar.id))
    ov_scored.sort(key=lambda x: x[0], reverse=True)

    cap = min(18, max(1, len(ov_scored)))
    if ov_scored and ov_scored[0][0] > 0:
        pick = [aid for _, aid in ov_scored[:cap]]
    else:
        pick = [r.id for r in rows[: min(15, len(rows))]]

    log.info(
        "[DIRECT] Scenario-style query → %d articles by content overlap (best=%.3f) for '%s'",
        len(pick),
        ov_scored[0][0] if ov_scored else 0.0,
        resolved_doc_number,
    )
    return await _flatten_chunks_for_article_ids(db, pick, 0.78)


async def hybrid_search(
    query: str,
    db: AsyncSession,
    top_k: int | None = None,
    retrieval_k: int | None = None,
    doc_number: Optional[str] = None,
    document_id: Optional[int] = None,
    single_article_only: bool = True,
    legal_domains: Optional[List[str]] = None,
    max_articles: Optional[int] = None,
    doc_number_source_query: Optional[str] = None,
) -> List[Dict]:
    """Run article-aware hybrid retrieval and return context-ready chunks.

    Args:
        legal_domains: Pre-classified domain tags to filter Qdrant vectors.
        max_articles: When single_article_only=True, keep top N articles (default from config).
                      1 = one article only; 3–5 = multi-article context for comparison.
        doc_number_source_query: If set, số hiệu văn bản (DB Mode 2 + resolve path) lấy từ
            chuỗi này thay vì từ ``query`` (dùng khi ``query`` là biến thể từ expand_query).
    """
    final_k = top_k or RERANK_TOP_K
    fetch_k = retrieval_k or RETRIEVAL_TOP_K
    explicit_article = _extract_article_reference(query)
    ref_source = (
        doc_number_source_query if doc_number_source_query is not None else query
    )
    explicit_doc_ref = _extract_doc_reference(ref_source)

    # ── 0. Direct DB lookup for specific article/clause queries ──
    db_lookup_results = await lookup_article_from_db(
        db, query, doc_number_source_query=doc_number_source_query
    )
    prefer_multi = _query_prefers_multi_article(query)
    if db_lookup_results:
        db_article_count = _unique_article_count(db_lookup_results)
        if prefer_multi and db_article_count < 2 and not explicit_article:
            log.info(
                "[HYBRID] DB lookup returned %d passages (%d article) for multi-article query; continuing hybrid merge",
                len(db_lookup_results),
                db_article_count,
            )
        else:
            log.info("[HYBRID] DB article lookup returned %d passages — skipping vector search", len(db_lookup_results))
            return db_lookup_results

    # ── 0b. Resolve doc_number against DB (handles NĐ↔ND mismatch) ──
    raw_doc_ref = doc_number or explicit_doc_ref
    resolved_doc_number: Optional[str] = None
    if raw_doc_ref:
        resolved_doc_number = await _resolve_doc_number(db, raw_doc_ref)
        log.info("[HYBRID] raw_ref='%s' → resolved='%s'", raw_doc_ref, resolved_doc_number)

    # ── 0c. Precision path: direct article lookup when doc is known ──
    if resolved_doc_number:
        direct_results = await _direct_article_lookup(
            db, resolved_doc_number, query, explicit_article,
        )
        if direct_results:
            direct_article_count = _unique_article_count(direct_results)
            if prefer_multi and direct_article_count < 2 and not explicit_article:
                log.info(
                    "[HYBRID] Direct lookup returned %d chunks (%d article) for multi-article query; continuing hybrid merge",
                    len(direct_results),
                    direct_article_count,
                )
            else:
                log.info("[HYBRID] Direct article lookup returned %d chunks", len(direct_results))
                return direct_results

    # ── 1. Vector search (with domain pre-filter) ────────
    vector_results = vector_search(
        query=query,
        top_k=fetch_k,
        doc_number=resolved_doc_number,
        document_id=document_id,
        legal_domains=legal_domains,
    )

    # ── 2. Keyword search ────────────────────────────────
    keyword_results = await keyword_search(
        query=query,
        db=db,
        top_k=fetch_k,
        doc_number=resolved_doc_number,
    )

    # ── 1b. Vector lọc domain trả về 0 nhưng keyword (ILIKE) vẫn có — bổ sung vector không lọc
    # để không chỉ dựa vào từ khóa rời (dễ kéo nhầm văn bản khác lĩnh vực).
    vector_broad: List[Dict] = []
    if (
        legal_domains
        and not resolved_doc_number
        and len(vector_results) == 0
        and len(keyword_results) >= 3
    ):
        log.info(
            "Domain-filtered vector returned 0 hits; supplementing unfiltered vector (domains=%s)",
            legal_domains,
        )
        vector_broad = vector_search(
            query=query,
            top_k=fetch_k,
            doc_number=None,
            document_id=document_id,
            legal_domains=None,
        )

    # ── 2b. Fallback: if filtered search returned too few results, retry without filters
    # Không bỏ lọc số hiệu khi người dùng nêu rõ văn bản — tránh kéo nhầm NĐ/luật khác (vd. 137 vs 138).
    if (resolved_doc_number or legal_domains) and (len(vector_results) + len(keyword_results)) < 3:
        if explicit_doc_ref:
            log.info(
                "Filtered search few results (%d+%d) but explicit doc ref present — "
                "keeping doc/domain filters (doc=%s)",
                len(vector_results),
                len(keyword_results),
                resolved_doc_number,
            )
        else:
            log.info(
                "Filtered search too few results (%d+%d), retrying unfiltered (domains=%s, doc=%s)",
                len(vector_results),
                len(keyword_results),
                legal_domains,
                resolved_doc_number,
            )
            vector_results = vector_search(query=query, top_k=fetch_k, document_id=document_id)
            keyword_results = await keyword_search(query=query, db=db, top_k=fetch_k)

    # ── 3. Merge with RRF ────────────────────────────────
    if vector_broad:
        merged = _reciprocal_rank_fusion(vector_results, vector_broad, keyword_results)
    else:
        merged = _reciprocal_rank_fusion(vector_results, keyword_results)

    if not merged:
        log.warning("No results from hybrid search for: '%.60s...'", query)
        return []

    # ── 4. Rerank (keep more candidates for article grouping) ──
    reranked = rerank(query=query, candidates=merged, top_k=max(final_k * 4, 12))
    await _enrich_missing_metadata(db, reranked)
    for item in reranked:
        item["article_number"] = _normalize_article_number(item.get("article_number"))
        item["clause_number"] = _normalize_clause_number(item.get("clause_number"))

    # ── 3.5. Penalize passages whose law_intents / legal_domain disagree with query domains ──
    if not resolved_doc_number:
        _apply_topic_mismatch_penalty(query, reranked)

    # ── 4a. Boost điểm rerank khi chunk khớp domain đã phân loại (ưu tiên đúng lĩnh vực) ──
    if legal_domains and reranked:
        allowed_dom = set(legal_domains)
        boost = 0.12
        for item in reranked:
            if item.get("legal_domain") in allowed_dom:
                item["rerank_score"] = float(item.get("rerank_score", 0.0)) + boost
        reranked.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)

    # ── 4b. Post-filter by domain when we had domain filter (incl. after fallback) ──
    # Only keep passages that explicitly match allowed domains; drop empty/other to avoid wrong-law citations.
    if legal_domains and reranked:
        allowed = set(legal_domains)
        filtered = [item for item in reranked if _item_matches_domain_filter(item, allowed)]
        if filtered:
            reranked = filtered
            log.info("Domain post-filter: %d → %d passages (allowed=%s)", len(merged), len(reranked), list(allowed))
        else:
            # Metadata legal_domain trên chunk có thể thiếu/sai lệch với classifier — không được trả rỗng.
            log.warning(
                "Domain post-filter would drop all passages (allowed=%s); keeping boosted rerank list",
                list(allowed),
            )

    # ── 4c. V3: đa dạng hóa theo document + quyết định số article động (sau rerank)
    if reranked:
        reranked = diversify_by_article(reranked, min_docs=3)
        dyn_article_cap = dynamic_max_articles(reranked, query)
    else:
        dyn_article_cap = 1

    reranked_before_article_collapse = list(reranked) if reranked else []

    # ── 5. Article selection mode ───────────────────────────
    # Winner-takes-all tĩnh đã thay bằng diversify + dynamic_max_articles (n_keep).
    if not single_article_only:
        reranked = reranked[:final_k]
    else:
        n_keep = dyn_article_cap
        if max_articles is not None:
            n_keep = min(n_keep, max_articles)
        n_keep = max(1, min(n_keep, MULTI_ARTICLE_MAX_ARTICLES))
        group_scores = _score_and_group_articles(query, reranked, explicit_article, explicit_doc_ref)
        if n_keep > 1:
            top_groups = _select_top_n_articles(group_scores, n_keep)
            if top_groups:
                all_expanded: List[Dict] = []
                for grp in top_groups:
                    selected = [r for r in reranked if _group_key(r) == grp]
                    selected.sort(
                        key=lambda x: float(x.get("article_match_score", x.get("rerank_score", 0.0))),
                        reverse=True,
                    )
                    article_ids = [r.get("article_id") for r in selected if r.get("article_id")]
                    if article_ids:
                        expanded_map = await _fetch_full_article_chunks(db, [article_ids[0]])
                        expanded = expanded_map.get(article_ids[0], [])
                        best_score = selected[0].get("rerank_score", selected[0].get("score", 0.0))
                        for chunk in expanded:
                            chunk["rerank_score"] = best_score
                        all_expanded.extend(expanded)
                if all_expanded:
                    reranked = all_expanded
                else:
                    reranked = reranked[: min(len(reranked), final_k * n_keep)]
            else:
                reranked = reranked[:final_k]
        else:
            best_group = _select_best_article(group_scores)
            if best_group:
                selected = [r for r in reranked if _group_key(r) == best_group]
                selected.sort(
                    key=lambda x: float(x.get("article_match_score", x.get("rerank_score", 0.0))),
                    reverse=True,
                )
                article_ids = [r.get("article_id") for r in selected if r.get("article_id")]
                chosen_article_id = article_ids[0] if article_ids else None
                if chosen_article_id:
                    expanded_map = await _fetch_full_article_chunks(db, [chosen_article_id])
                    expanded = expanded_map.get(chosen_article_id, [])
                    for chunk in expanded:
                        chunk["rerank_score"] = selected[0].get("rerank_score", selected[0].get("score", 0.0))
                    reranked = expanded or selected[:final_k]
                else:
                    reranked = selected[:final_k]
            else:
                reranked = reranked[:final_k]

    if reranked_before_article_collapse:
        reranked = await _merge_amendment_expanded_passages(
            db, reranked_before_article_collapse, reranked, query
        )

    log.info(
        "Hybrid search: vector=%d keyword=%d merged=%d explicit_article=%s doc_ref=%s final=%d",
        len(vector_results),
        len(keyword_results),
        len(merged),
        explicit_article or "-",
        resolved_doc_number or "-",
        len(reranked),
    )
    return reranked


def _reciprocal_rank_fusion(
    *result_lists: List[Dict],
    k: int = 60,
) -> List[Dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for a document = Σ 1 / (k + rank_i)
    where k is a constant (default 60) and rank_i is the rank in list i.
    """
    scores: Dict[str, float] = {}
    items: Dict[str, Dict] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            # Use a composite key: document_id + text hash for dedup
            key = f"{item.get('document_id', '')}:{hash(item.get('text_chunk', ''))}"
            rrf_score = 1.0 / (k + rank + 1)
            scores[key] = scores.get(key, 0.0) + rrf_score

            # Keep the item with the best individual score
            if key not in items or item.get("score", 0) > items[key].get("score", 0):
                items[key] = item

    # Sort by RRF score
    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    merged = []
    for key in ranked_keys:
        item = items[key].copy()
        item["rrf_score"] = scores[key]
        merged.append(item)

    return merged
