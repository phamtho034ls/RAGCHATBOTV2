"""Hybrid retriever with article-aware legal retrieval."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import MULTI_ARTICLE_MAX_ARTICLES, RERANK_TOP_K, RETRIEVAL_TOP_K
from app.database.models import VectorChunk, Document, Article, Clause, Chapter, Section
from app.retrieval.article_lookup import lookup_article_from_db
from app.retrieval.keyword_retriever import keyword_search
from app.retrieval.reranker import rerank
from app.retrieval.vector_retriever import vector_search
from app.retrieval.article_selection import diversify_by_article, dynamic_max_articles

log = logging.getLogger(__name__)


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
            stmt = select(Document.doc_number).where(
                Document.doc_number.like(f"{prefix}%")
            ).limit(1)
            row = (await db.execute(stmt)).scalar()
            if row:
                log.info("[RESOLVE] '%s' → '%s' (prefix '%s')", raw_ref, row, prefix)
                return row

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


async def _enrich_missing_metadata(db: AsyncSession, items: List[Dict]) -> None:
    """Fill article/document metadata for old vectors missing payload fields."""
    article_ids = {item.get("article_id") for item in items if item.get("article_id")}
    if not article_ids:
        return

    stmt = (
        select(
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
        )
        .join(Document, Article.document_id == Document.id)
        .outerjoin(Chapter, Article.chapter_id == Chapter.id)
        .outerjoin(Section, Article.section_id == Section.id)
        .where(Article.id.in_(article_ids))
    )
    result = await db.execute(stmt)
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
    """
    stmt = (
        select(Article.id, Article.article_number, Article.title, Document.id.label("document_id"))
        .join(Document, Article.document_id == Document.id)
        .where(Document.doc_number == resolved_doc_number)
    )
    rows = (await db.execute(stmt)).all()
    if not rows:
        return []

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

    return []


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

    # ── 2b. Fallback: if filtered search returned too few results, retry without filters
    if (resolved_doc_number or legal_domains) and (len(vector_results) + len(keyword_results)) < 3:
        log.info(
            "Filtered search too few results (%d+%d), retrying unfiltered (domains=%s, doc=%s)",
            len(vector_results), len(keyword_results), legal_domains, resolved_doc_number,
        )
        vector_results = vector_search(query=query, top_k=fetch_k, document_id=document_id)
        keyword_results = await keyword_search(query=query, db=db, top_k=fetch_k)

    # ── 3. Merge with RRF ────────────────────────────────
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

    # ── 4b. Post-filter by domain when we had domain filter (incl. after fallback) ──
    # Only keep passages that explicitly match allowed domains; drop empty/other to avoid wrong-law citations.
    if legal_domains and reranked:
        allowed = set(legal_domains)
        filtered = [item for item in reranked if item.get("legal_domain") in allowed]
        if filtered:
            reranked = filtered
            log.info("Domain post-filter: %d → %d passages (allowed=%s)", len(merged), len(reranked), list(allowed))
        else:
            log.warning(
                "Domain post-filter removed all passages (allowed=%s); keeping none to avoid wrong citations",
                list(allowed),
            )
            reranked = []

    # ── 4c. V3: đa dạng hóa theo document + quyết định số article động (sau rerank)
    if reranked:
        reranked = diversify_by_article(reranked, min_docs=3)
        dyn_article_cap = dynamic_max_articles(reranked)
    else:
        dyn_article_cap = 1

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
