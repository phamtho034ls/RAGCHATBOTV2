"""Production RAG chain with legal QA prompt engineering.

Enforces:
- Only answer using retrieved documents
- Always cite legal source (Article X of Document Y)
- Full article retrieval – never return partial articles
- Internal metadata never leaks into answer text
- Structured legal response format (Câu trả lời / Căn cứ pháp lý)
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import (
    ANSWER_VALIDATION_THRESHOLD,
    DEFAULT_TEMPERATURE,
    MULTI_ARTICLE_MAX_ARTICLES,
    OPENAI_MODEL,
    SYSTEM_PROMPT_V2,
    RAG_PROMPT_TEMPLATE_V2,
    COMMUNE_OFFICER_SYSTEM_PROMPT,
    COMMUNE_OFFICER_RAG_TEMPLATE,
    NO_INFO_MESSAGE,
    USE_MULTI_ARTICLE_FOR_CONDITIONS,
)
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.vector_retriever import vector_search
from app.cache.redis_cache import get_cached_answer, cache_answer
from app.monitoring.chat_logger import log_interaction
from app.services.llm_client import generate, generate_with_messages
from app.services.answer_validator import validate_article_completeness
from app.services.query_understanding import analyze_query
from app.services.query_expansion import needs_expansion, expand_query, should_expand_query_v2
from app.services.intent_detector import get_rag_intents
from app.services.article_grouper import (
    dedup_chunks,
    group_chunks_by_article,
    format_grouped_context,
)
from app.services.domain_classifier import classify_query_domain, get_domain_filter_values
from app.tools import draft_tool

log = logging.getLogger(__name__)

def _get_rag_intent_flags(query: str) -> Dict[str, bool]:
    """Cờ RAG từ intent_detector (structural + semantic embedding), không dùng regex scenario/multi-article."""
    return get_rag_intents(query)


async def _multi_query_retrieve(
    query: str,
    db: AsyncSession,
    top_k: int = 20,
    doc_number: Optional[str] = None,
    legal_domains: Optional[List[str]] = None,
    force_expansion: bool = False,
) -> List[Dict]:
    """Expand query → retrieve per sub-query → merge + deduplicate.

    When force_expansion=True (e.g. từ intent classifier), luôn mở rộng truy vấn.
    Khi False, chỉ mở rộng nếu needs_expansion(query) (regex).
    """
    if not force_expansion and not needs_expansion(query):
        return await hybrid_search(
            query=query, db=db, top_k=top_k,
            doc_number=doc_number, single_article_only=False,
            legal_domains=legal_domains,
        )

    variants = await expand_query(query, max_variants=3)
    all_passages: List[Dict] = []

    for variant in variants:
        passages = await hybrid_search(
            query=variant,
            db=db,
            top_k=top_k,
            doc_number=doc_number,
            single_article_only=False,
            legal_domains=legal_domains,
            doc_number_source_query=query,
        )
        all_passages.extend(passages)

    deduped = dedup_chunks(all_passages)
    deduped.sort(
        key=lambda p: float(
            p.get("rerank_score", p.get("rrf_score", p.get("score", 0.0)))
        ),
        reverse=True,
    )

    final_k = top_k * 2
    log.info(
        "Multi-query retrieval: %d variants → %d raw → %d deduped → keeping %d",
        len(variants), len(all_passages), len(deduped), min(len(deduped), final_k),
    )
    return deduped[:final_k]


async def _fallback_full_article_retrieval(
    db: AsyncSession,
    passages: List[Dict],
    query: str,
) -> List[Dict]:
    """Fallback: If initial retrieval missed article chunks, re-retrieve
    from PostgreSQL using article_id with boosted coverage.

    Steps:
    1. Identify article_ids from existing passages
    2. Fetch ALL chunks for those articles from PostgreSQL
    3. Merge with existing passages, preserving order
    """
    from app.retrieval.hybrid_retriever import _fetch_full_article_chunks

    article_ids = list({p.get("article_id") for p in passages if p.get("article_id")})
    if not article_ids:
        return passages

    full_chunks = await _fetch_full_article_chunks(db, article_ids)
    if not full_chunks:
        return passages

    # Merge: keep all existing passages + add missing chunks from full fetch
    existing_texts = {p.get("text_chunk", "")[:100] for p in passages}
    expanded = list(passages)

    for aid, chunks in full_chunks.items():
        for chunk in chunks:
            chunk_key = chunk["text_chunk"][:100]
            if chunk_key not in existing_texts:
                existing_texts.add(chunk_key)
                expanded.append(chunk)

    log.info("Fallback article retrieval: %d → %d passages", len(passages), len(expanded))
    return expanded


def _extract_article_reference(text: str) -> Optional[str]:
    m = re.search(r"điều\s+(\d+[a-zA-Z]?)", text or "", re.IGNORECASE)
    return m.group(1) if m else None


def _normalize_article_number(article_number: Optional[str]) -> Optional[str]:
    if not article_number:
        return None
    m = re.search(r"(\d+[a-zA-Z]?)", str(article_number))
    return m.group(1) if m else str(article_number).strip()


def _is_document_lookup_query(query: str) -> bool:
    """Detect queries asking 'which legal documents' (not article-content queries).

    Excludes patterns like 'điều luật nào', 'điều khoản nào', 'khoản nào'
    where the user is asking for the specific article/clause content,
    not a document list.
    """
    q = (query or "").lower()

    # "điều luật nào" / "điều khoản nào" / "khoản nào" / "điều nào"
    # → user asks WHICH article/clause, wants content, not a doc list
    article_content_patterns = [
        "điều luật nào",
        "điều khoản nào",
        "khoản nào",
        "điều nào",
        "nằm trong điều",
        "quy định tại điều",
        "nằm ở điều",
        "thuộc điều",
    ]
    if any(p in q for p in article_content_patterns):
        return False

    patterns = [
        "văn bản nào",
        "nghị định nào",
        "luật nào",
        "thông tư nào",
        "chỉ thị nào",
        "theo các văn bản pháp luật",
        "trong các tài liệu đã cung cấp",
    ]
    return any(p in q for p in patterns)


def _build_document_lookup_answer(sources: List[Dict]) -> str:
    """Return document-list answer for queries asking 'which legal documents'."""
    if not sources:
        return NO_INFO_MESSAGE

    docs = []
    seen = set()
    for s in sources:
        label = _format_doc_label(s)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        docs.append(label)

    if not docs:
        return NO_INFO_MESSAGE

    lines = ["Câu trả lời:", "", "Các văn bản pháp luật liên quan trong cơ sở dữ liệu hiện có:"]
    for i, d in enumerate(docs, 1):
        lines.append(f"    {d}")

    lines.append("")
    citation_block = _format_legal_citation(sources)
    if citation_block:
        lines.append(citation_block)
    return "\n".join(lines)


def _build_multi_source_answer(query: str, sources: List[Dict]) -> str:
    """Deterministic multi-source answer to avoid collapsing into one article."""
    if not sources:
        return NO_INFO_MESSAGE

    docs: List[str] = []
    seen_docs = set()
    for s in sources:
        label = _format_doc_label(s)
        if not label:
            continue
        key = label.lower()
        if key in seen_docs:
            continue
        seen_docs.add(key)
        docs.append(label)

    citations: List[str] = []
    seen_cites = set()
    for s in sources:
        cite = (s.get("citation") or "").strip()
        if not cite:
            label = _format_doc_label(s)
            article = _normalize_article_number(s.get("article_number"))
            cite = f"{label} – Điều {article}" if label and article else label
        if not cite:
            continue
        k = cite.lower()
        if k in seen_cites:
            continue
        seen_cites.add(k)
        citations.append(cite)

    lines = [
        "Câu trả lời:",
        "",
        f"Câu hỏi \"{query}\" được tổng hợp từ nhiều nguồn pháp lý trong cơ sở dữ liệu hiện có.",
    ]
    if docs:
        lines.extend(["", "Các văn bản pháp luật liên quan:"])
        lines.extend([f"- {d}" for d in docs[:12]])
    if citations:
        lines.extend(["", "Các điều khoản tiêu biểu liên quan:"])
        lines.extend([f"- {c}" for c in citations[:15]])
    lines.extend([
        "",
        "Lưu ý: Danh mục ngành, nghề đầu tư kinh doanh có điều kiện cần đối chiếu theo phụ lục/danh mục ban hành kèm văn bản còn hiệu lực tại thời điểm áp dụng.",
    ])
    return "\n".join(lines)


def _should_use_multi_source_summary(
    query: str,
    sources: List[Dict],
    *,
    use_multi: bool,
    multi_article_conditions: bool,
) -> bool:
    """Decide format by retrieval diversity instead of manual query regex."""
    if not sources:
        return False
    if _is_document_lookup_query(query) or _query_demands_specific_article(query):
        return False

    unique_articles = len({s.get("article_id") for s in sources if s.get("article_id")})
    unique_docs = len({(s.get("doc_number") or "").strip() for s in sources if (s.get("doc_number") or "").strip()})

    # Strong evidence from retrieval diversity.
    if unique_articles >= 3 and unique_docs >= 2:
        return True
    # Weaker diversity but query already routed to multi-article path.
    if (use_multi or multi_article_conditions) and unique_articles >= 3:
        return True
    return False


def _select_single_article_passages(passages: List[Dict], query: str) -> List[Dict]:
    """Keep passages from only one best-matching article to prevent article mixing."""
    if not passages:
        return []

    query_article = _extract_article_reference(query)
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for p in passages:
        art_no = _normalize_article_number(p.get("article_number"))
        if art_no:
            p["article_number"] = art_no
            grouped[art_no].append(p)

    if not grouped:
        return passages

    if query_article and query_article in grouped:
        return grouped[query_article]

    # For "mức phạt" queries, deprioritize "thẩm quyền xử phạt" articles when
    # non-authority alternatives are available – they likely have the actual fine amounts.
    effective_grouped = grouped
    if _query_asks_fine_amount(query):
        non_authority = {
            art: chunks
            for art, chunks in grouped.items()
            if not any(
                _THAM_QUYEN_TITLE_RE.search(c.get("article_title", "") or "")
                for c in chunks
            )
        }
        if non_authority:
            log.info(
                "Filtered thẩm quyền articles for mức phạt query: %d → %d candidate articles",
                len(grouped),
                len(non_authority),
            )
            effective_grouped = non_authority

    best_article = max(
        effective_grouped.keys(),
        key=lambda art: max(
            [
                float(x.get("article_match_score", x.get("rerank_score", x.get("score", 0.0))))
                for x in effective_grouped[art]
            ]
            or [0.0]
        ),
    )
    return effective_grouped[best_article]


def _is_no_info_answer(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    return normalized == NO_INFO_MESSAGE.lower() or "không tìm thấy nội dung phù hợp" in normalized


def _query_demands_specific_article(query: str) -> bool:
    q = (query or "").lower()
    return bool(re.search(r"\bđiều\s+\d+[a-zA-Z]?\b", q))


# ── Fine-amount vs enforcement-authority mismatch detection ──────────────

_MUC_PHAT_QUERY_RE = re.compile(
    r"mức\s+phạt|mức\s+xử\s+phạt|phạt\s+bao\s+nhiêu"
    r"|bị\s+phạt\s*(bao\s+nhiêu|như\s+thế\s+nào|thế\s+nào)"
    r"|tiền\s+phạt|mức\s+tiền\s+phạt|hình\s+thức\s+xử\s+phạt",
    re.IGNORECASE,
)

_THAM_QUYEN_TITLE_RE = re.compile(r"thẩm\s+quyền", re.IGNORECASE)

_THAM_QUYEN_CONTEXT_RE = re.compile(
    r"thẩm\s+quyền\s+xử\s+phạt"
    r"|có\s+quyền\s*[:\-]?\s*(?:[a-z]\)\s*(?:phạt|cảnh\s+cáo)|xử\s+phạt)"
    r"|được\s+quyền\s+phạt\s+tiền"
    r"|có\s+quyền\s+(?:phạt\s+tiền|cảnh\s+cáo|tịch\s+thu)",
    re.IGNORECASE,
)


def _query_asks_fine_amount(query: str) -> bool:
    """Detect if the query is specifically asking about fine/penalty amounts."""
    return bool(_MUC_PHAT_QUERY_RE.search(query or ""))


def _article_is_tham_quyen(passages: List[Dict]) -> bool:
    """Check if passages belong to a 'thẩm quyền xử phạt' (enforcement authority) article."""
    for p in passages:
        title = p.get("article_title", "") or ""
        if _THAM_QUYEN_TITLE_RE.search(title):
            return True
    return False


def _context_describes_authority(context: str) -> bool:
    """Return True if retrieved context is about enforcement authority, not specific fine amounts."""
    return bool(_THAM_QUYEN_CONTEXT_RE.search(context or ""))


def _build_related_documents_fallback(sources: List[Dict]) -> str:
    """Case-2 fallback: context has legal docs but no direct detailed answer."""
    docs: List[str] = []
    seen = set()
    for s in sources:
        label = _format_doc_label(s)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        docs.append(label)

    if not docs:
        return NO_INFO_MESSAGE

    lines = [
        "Câu trả lời:",
        "",
        "Trong các tài liệu hiện có chưa tìm thấy nội dung chi tiết trả lời trực tiếp câu hỏi.",
        "",
        "Tuy nhiên, hệ thống đã tìm thấy các văn bản pháp luật liên quan:",
        "",
        "Danh sách văn bản liên quan:",
    ]
    lines.extend([f"- {d}" for d in docs[:12]])
    return "\n".join(lines)


async def _answer_with_authority_summary(
    query: str,
    context: str,
    sources: List[Dict],
    temperature: float,
) -> str:
    """Re-generate answer when the query asks about a fine amount but retrieved context
    describes enforcement authority (thẩm quyền xử phạt).

    Strategy:
    - Explain that the found article is about who CAN impose fines, not the specific
      fine amount for the violation behaviour in question.
    - Summarize ALL enforcement entities and their maximum fine limits from the context.
    - Instruct the user where to look for the specific fine amount.
    """
    prompt = f"""NGỮ CẢNH PHÁP LÝ:
{context}

CÂU HỎI: {query}

PHÂN TÍCH TÌNH HUỐNG:
- Câu hỏi hỏi về MỨC PHẠT cụ thể cho một hành vi vi phạm.
- Nội dung văn bản tìm được là điều khoản về THẨM QUYỀN XỬ PHẠT (quy định ai có thẩm quyền \
xử phạt và mức tiền phạt tối đa mà họ được áp dụng), không phải điều khoản về hành vi vi phạm \
và mức phạt tương ứng.

YÊU CẦU TRẢ LỜI:
1. Nêu rõ: văn bản tìm được quy định về THẨM QUYỀN XỬ PHẠT, không phải mức phạt cụ thể cho hành vi.
2. Liệt kê ĐẦY ĐỦ các CHỦ THỂ có thẩm quyền xử phạt trong lĩnh vực liên quan (từ NGỮ CẢNH), \
kèm theo MỨC TIỀN PHẠT TỐI ĐA mà mỗi chủ thể được áp dụng cho từng lĩnh vực (văn hóa, quảng cáo, ...).
3. Kết luận: để biết mức phạt cụ thể cho hành vi vi phạm đang hỏi, cần tham khảo điều khoản \
quy định hành vi vi phạm cụ thể (thường là các điều trước phần thẩm quyền trong cùng văn bản).
4. Ghi rõ SỐ HIỆU văn bản và số điều (ví dụ: 144/2020/NĐ-CP – Điều 68).

ĐỊNH DẠNG BẮT BUỘC:
Câu trả lời:

[Giải thích ngắn: văn bản này quy định thẩm quyền, không phải mức phạt hành vi]

Các chủ thể có thẩm quyền xử phạt và mức tiền phạt tối đa:
- [Chủ thể 1]: phạt tiền đến [X] đồng (văn hóa); đến [Y] đồng (quảng cáo/lĩnh vực liên quan)
- [Chủ thể 2]: ...
...

Lưu ý: Để biết mức phạt cụ thể cho hành vi [hành vi trong câu hỏi], cần tra cứu điều khoản \
về hành vi vi phạm trong cùng văn bản.

Chỉ sử dụng thông tin từ NGỮ CẢNH. KHÔNG bịa số liệu."""

    answer = await generate(
        prompt=prompt,
        system=SYSTEM_PROMPT_V2,
        temperature=min(temperature, 0.1),
    )
    return _sanitize_output(answer)


def _force_no_info_if_needed(answer: str) -> str:
    """Normalize no-info variants to canonical response."""
    return NO_INFO_MESSAGE if _is_no_info_answer(answer) else answer


def _format_doc_label(source: Dict) -> str:
    """Build human-readable label that always prioritizes legal document number."""
    doc_number = (source.get("doc_number") or "").strip()
    doc_title = (source.get("document_title") or "").strip()
    if doc_number and doc_title:
        if doc_number.lower() in doc_title.lower():
            return doc_title
        return f"{doc_number} ({doc_title})"
    return doc_number or doc_title


def _answer_has_explicit_doc_number(answer: str) -> bool:
    return bool(re.search(r"\b\d+/\d{4}/[A-ZĐa-zđ0-9\-]+\b", answer or ""))


def _ensure_explicit_document_numbers(answer: str, sources: List[Dict]) -> str:
    """Ensure answer lists exact document numbers for legal document questions."""
    if not sources:
        return answer
    if _answer_has_explicit_doc_number(answer):
        return answer

    labels: List[str] = []
    seen = set()
    for s in sources:
        label = _format_doc_label(s)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    if not labels:
        return answer

    appendix = ["", "Văn bản xác định rõ số hiệu:", *[f"- {x}" for x in labels[:12]]]
    return (answer or "").rstrip() + "\n" + "\n".join(appendix)


def _convert_legacy_sources_to_v2(legacy_sources: List[Dict]) -> List[Dict]:
    """Convert legacy source format from tools to SourceInfo-compatible dict."""
    converted: List[Dict] = []
    for s in legacy_sources:
        meta = s.get("metadata", {}) or {}
        doc_title = meta.get("law_name") or meta.get("title") or ""
        doc_number = meta.get("doc_number", "")
        article_number = _normalize_article_number(meta.get("article_number"))
        citation = ""
        label = _format_doc_label({"doc_number": doc_number, "document_title": doc_title})
        if label and article_number:
            citation = f"{label} – Điều {article_number}"
        elif label:
            citation = label
        converted.append(
            {
                "citation": citation,
                "document_title": doc_title,
                "article_number": article_number,
                "article_title": meta.get("article_title") or "",
                "snippet": (s.get("content", "") or "")[:500],
                "document_id": meta.get("document_id"),
                "article_id": meta.get("article_id"),
                "clause_id": meta.get("clause_id"),
                "doc_number": doc_number,
                "score": None,
            }
        )
    return converted


async def _answer_checklist_query(
    query: str,
    db: AsyncSession,
    temperature: float,
    doc_number: Optional[str],
    legal_domains: Optional[List[str]] = None,
) -> Dict:
    """Answer synthesis/checklist queries by aggregating multiple legal documents."""
    passages = await hybrid_search(
        query=query,
        db=db,
        top_k=20,
        retrieval_k=40,
        doc_number=doc_number,
        single_article_only=False,
        legal_domains=legal_domains,
    )
    if not passages:
        return {"answer": NO_INFO_MESSAGE, "sources": [], "confidence_score": 0.0}

    passages = passages[:20]
    context = _build_context(passages)
    sources = _extract_sources(passages)

    prompt = f"""
NGỮ CẢNH PHÁP LÝ:
{context}

YÊU CẦU:
{query}

Hãy trả lời theo dạng CHECKLIST PHÁP LÝ TỔNG HỢP:
I. Danh mục văn bản pháp lý liên quan.
II. Checklist chức năng, nhiệm vụ, thẩm quyền quản lý.
III. Danh mục việc cần triển khai trong quản lý thực tế.

NGUYÊN TẮC BẮT BUỘC:
- CHỈ ĐƯỢC sử dụng văn bản có SỐ HIỆU xuất hiện trong NGỮ CẢNH PHÁP LÝ ở trên.
- TUYỆT ĐỐI KHÔNG ĐƯỢC bịa đặt hoặc thêm văn bản từ kiến thức bên ngoài.
- Nếu một văn bản không có trong NGỮ CẢNH → KHÔNG ĐƯỢC đề cập.
- Ghi rõ SỐ HIỆU văn bản (ví dụ: 38/2021/NĐ-CP), không ghi chung chung.
"""
    answer = await generate(
        prompt=prompt,
        system=SYSTEM_PROMPT_V2,
        temperature=min(temperature, 0.3),
    )
    answer = _sanitize_output(answer)
    answer = _force_no_info_if_needed(answer)

    context_nums = _collect_context_doc_numbers(passages)
    answer = _strip_hallucinated_doc_numbers(answer, context_nums)

    if _is_no_info_answer(answer) and sources:
        answer = _build_related_documents_fallback(sources)
    answer = _ensure_explicit_document_numbers(answer, sources)
    answer = _ensure_legal_citations(answer, sources)
    answer = _ensure_response_format(answer)
    confidence = _compute_confidence(passages, answer)
    return {"answer": answer, "sources": sources, "confidence_score": confidence}


async def _answer_commune_officer_query(
    query: str,
    db: AsyncSession,
    temperature: float,
    doc_number: Optional[str],
    legal_domains: Optional[List[str]] = None,
) -> Dict:
    """Handle commune-level administrative queries using the VHXH officer pipeline.

    Uses COMMUNE_OFFICER_SYSTEM_PROMPT with the 5-section mandatory response format:
    1. Nhận định tình huống
    2. Căn cứ pháp lý
    3. Quy trình xử lý
    4. Phối hợp liên ngành
    5. Giải pháp lâu dài
    """
    from app.services.query_understanding import analyze_commune_situation

    situation = analyze_commune_situation(query)

    if needs_expansion(query):
        passages = await _multi_query_retrieve(
            query=query, db=db, top_k=20, doc_number=doc_number,
            legal_domains=legal_domains,
        )
    else:
        passages = await hybrid_search(
            query=query,
            db=db,
            top_k=20,
            retrieval_k=40,
            doc_number=doc_number,
            single_article_only=False,
            legal_domains=legal_domains,
        )

    if not passages:
        # No retrieved context – still provide general administrative guidance
        prompt = COMMUNE_OFFICER_RAG_TEMPLATE.format(
            context="Không tìm thấy văn bản pháp luật cụ thể trong cơ sở dữ liệu.",
            question=query,
            field=situation.get("subject", "không xác định"),
            subject=situation.get("subject", "không xác định"),
            violation=situation.get("violation", "không có"),
            severity=situation.get("severity", "chưa xác định"),
        )
        answer = await generate(
            prompt=prompt,
            system=COMMUNE_OFFICER_SYSTEM_PROMPT,
            temperature=min(temperature, 0.3),
        )
        answer = _sanitize_output(answer)
        return {"answer": answer, "sources": [], "confidence_score": 0.4}

    passages = passages[:20]
    context = _build_context(passages)
    sources = _extract_sources(passages)

    prompt = COMMUNE_OFFICER_RAG_TEMPLATE.format(
        context=context,
        question=query,
        field=situation.get("subject", "không xác định"),
        subject=situation.get("subject", "không xác định"),
        violation=situation.get("violation", "không có"),
        severity=situation.get("severity", "chưa xác định"),
    )

    answer = await generate(
        prompt=prompt,
        system=COMMUNE_OFFICER_SYSTEM_PROMPT,
        temperature=min(temperature, 0.3),
    )
    answer = _sanitize_output(answer)

    context_nums = _collect_context_doc_numbers(passages)
    answer = _strip_hallucinated_doc_numbers(answer, context_nums)

    if _is_no_info_answer(answer) and sources:
        answer = _build_related_documents_fallback(sources)

    answer = _ensure_legal_citations(answer, sources)
    confidence = _compute_confidence(passages, answer)

    return {"answer": answer, "sources": sources, "confidence_score": confidence}


async def _answer_drafting_query(query: str, temperature: float) -> Dict:
    """Answer administrative/legal drafting queries using dedicated drafting tool."""
    tool_result = await draft_tool.run(content=query, temperature=min(temperature, 0.35))
    answer = _sanitize_output(tool_result.get("result", ""))
    answer = _force_no_info_if_needed(answer)
    if not answer:
        answer = NO_INFO_MESSAGE
    sources = _convert_legacy_sources_to_v2(tool_result.get("sources", []))
    confidence = 0.75 if sources else 0.5
    return {"answer": answer, "sources": sources, "confidence_score": confidence}


def _with_conv_meta(result: Dict, conversation_id: str, retried: bool = False) -> Dict:
    out = dict(result)
    out["conversation_id"] = conversation_id
    out["retried"] = retried
    return out


async def rag_query(
    query: str,
    db: AsyncSession,
    temperature: float = DEFAULT_TEMPERATURE,
    doc_number: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Dict:
    """Run the full RAG pipeline: retrieve → build context → generate answer.

    Includes fallback safety: if the answer references an article but misses
    clauses, re-retrieves the full article and regenerates.

    Returns dict with: answer, sources, confidence_score
    """
    start_time = time.time()

    from app.memory.conversation_store import conversation_store

    if conversation_id and conversation_store.get(conversation_id):
        conv_id = conversation_id
    else:
        conv_id = conversation_store.create()["id"]

    # ── 0. Ninh Bình tool: câu hỏi phi pháp lý về Ninh Bình (địa lý, huyện, du lịch...)
    from app.services.ninh_binh_router import should_use_ninh_binh_tool, route_to_ninh_binh
    if should_use_ninh_binh_tool(query):
        nb_result = await route_to_ninh_binh(query)
        if nb_result:
            latency = (time.time() - start_time) * 1000
            await log_interaction(db, query, nb_result.get("answer", ""), [], 0.85, latency)
            return _with_conv_meta(
                {
                    "answer": nb_result.get("answer", ""),
                    "sources": nb_result.get("sources", []),
                    "confidence_score": 0.85,
                },
                conv_id,
            )

    analysis = analyze_query(query)
    intent = analysis.get("intent", "")

    # ── 1a. Legal domain for retrieval (all paths: commune, checklist, default) ──
    domain_filter = get_domain_filter_values(query)
    if domain_filter:
        domain_info = classify_query_domain(query, top_n=2)
        log.info(
            "Domain classification: %s → filter=%s",
            [f"{d['domain']}({d['confidence']:.2f})" for d in domain_info],
            domain_filter,
        )

    # ── 1b. RAG intent flags — intent_detector (structural + semantic embedding) ──
    rag_intents = _get_rag_intent_flags(query)
    log.info(
        "RAG intents (intent_detector): scenario=%s legal_lookup=%s multi_article=%s needs_expansion=%s",
        rag_intents.get("is_scenario"),
        rag_intents.get("is_legal_lookup"),
        rag_intents.get("use_multi_article"),
        rag_intents.get("needs_expansion"),
    )

    # ── 1. Check cache ───────────────────────────────────
    skip_cache = intent in {"checklist_documents", "document_drafting", "document_summary"}
    if not skip_cache:
        cached = await get_cached_answer(query)
        if cached:
            cached_answer = cached.get("answer", "") or ""
            cached_sources = cached.get("sources", []) or []
            # Guard against stale malformed cached answers (e.g. "Căn cứ pháp lý: string")
            if "căn cứ pháp lý" in cached_answer.lower() and "string" in cached_answer.lower():
                log.warning("Ignoring malformed cached answer for query: '%.50s...'", query)
            elif _is_no_info_answer(cached_answer) and cached_sources:
                log.warning("Ignoring stale no-info cache with existing sources: '%.50s...'", query)
            else:
                log.info("Cache hit for query: '%.50s...'", query)
                return _with_conv_meta(dict(cached), conv_id, retried=cached.get("retried", False))

    # ── Commune-level intent routing ─────────────────────
    from app.services.intent_detector import COMMUNE_LEVEL_INTENTS
    commune_situation = analysis.get("commune_situation")
    is_commune_query = (
        intent in COMMUNE_LEVEL_INTENTS
        or (commune_situation and commune_situation.get("violation", "không có") != "không có")
        or rag_intents.get("is_scenario", False)
    )

    if is_commune_query:
        result = await _answer_commune_officer_query(
            query=query,
            db=db,
            temperature=temperature,
            doc_number=doc_number,
            legal_domains=domain_filter,
        )
        await cache_answer(query, result)
        latency = (time.time() - start_time) * 1000
        doc_numbers = list({s.get("doc_number", "") for s in result.get("sources", [])})
        await log_interaction(
            db, query, result.get("answer", ""),
            doc_numbers, result.get("confidence_score", 0.0), latency,
        )
        return _with_conv_meta(result, conv_id)

    # Intent-specific routes
    if intent == "checklist_documents":
        result = await _answer_checklist_query(
            query=query,
            db=db,
            temperature=temperature,
            doc_number=doc_number,
            legal_domains=domain_filter,
        )
        await cache_answer(query, result)
        latency = (time.time() - start_time) * 1000
        doc_numbers = list({s.get("doc_number", "") for s in result.get("sources", [])})
        await log_interaction(
            db,
            query,
            result.get("answer", ""),
            doc_numbers,
            result.get("confidence_score", 0.0),
            latency,
        )
        return _with_conv_meta(result, conv_id)

    if intent == "document_drafting":
        result = await _answer_drafting_query(query=query, temperature=temperature)
        await cache_answer(query, result)
        latency = (time.time() - start_time) * 1000
        doc_numbers = list({s.get("doc_number", "") for s in result.get("sources", [])})
        await log_interaction(
            db,
            query,
            result.get("answer", ""),
            doc_numbers,
            result.get("confidence_score", 0.0),
            latency,
        )
        return _with_conv_meta(result, conv_id)

    if intent == "document_summary":
        from app.services.document_summarizer import list_document_articles
        summary_result = await list_document_articles(query)
        result = {
            "answer": summary_result.get("summary", ""),
            "sources": summary_result.get("sources", []),
            "confidence_score": summary_result.get("confidence_score", 0.0),
        }
        await cache_answer(query, result)
        latency = (time.time() - start_time) * 1000
        doc_numbers = list({s.get("doc_number", "") for s in result.get("sources", [])})
        await log_interaction(
            db, query, result["answer"], doc_numbers,
            result["confidence_score"], latency,
        )
        return _with_conv_meta(result, conv_id)

    # ── 2. Hybrid retrieval (with multi-query expansion) ──
    initial_for_expand = vector_search(
        query=query,
        top_k=8,
        doc_number=doc_number,
        legal_domains=domain_filter,
    )
    use_multi = rag_intents.get("needs_expansion", False) or should_expand_query_v2(
        query, initial_for_expand
    )
    multi_article_conditions = (
        rag_intents.get("use_multi_article", False) and USE_MULTI_ARTICLE_FOR_CONDITIONS
    )
    if use_multi:
        passages = await _multi_query_retrieve(
            query=query, db=db, doc_number=doc_number,
            legal_domains=domain_filter,
            force_expansion=use_multi,
        )
    else:
        passages = await hybrid_search(
            query=query,
            db=db,
            doc_number=doc_number,
            legal_domains=domain_filter,
            single_article_only=not multi_article_conditions,
            max_articles=MULTI_ARTICLE_MAX_ARTICLES if multi_article_conditions else None,
        )

    # ── 3. Build context ─────────────────────────────────
    if not passages:
        answer = NO_INFO_MESSAGE
        result = {
            "answer": answer,
            "sources": [],
            "confidence_score": 0.0,
        }
        latency = (time.time() - start_time) * 1000
        await log_interaction(db, query, answer, [], 0.0, latency)
        return _with_conv_meta(result, conv_id)

    # For multi-article / condition queries, group by article for structured context
    if use_multi or multi_article_conditions or any(p.get("_db_lookup") for p in passages):
        groups = group_chunks_by_article(dedup_chunks(passages))
        context = format_grouped_context(groups)
    else:
        if not any(p.get("_db_lookup") for p in passages):
            passages = _select_single_article_passages(passages, query)
        context = _build_context(passages)
    sources = _extract_sources(passages)

    prefer_multi_source_summary = _should_use_multi_source_summary(
        query,
        sources,
        use_multi=bool(use_multi),
        multi_article_conditions=bool(multi_article_conditions),
    )

    # Query type: user asks "which legal document(s)".
    # If retrieval evidence is strongly multi-source, prefer synthesis mode below.
    if _is_document_lookup_query(query) and not prefer_multi_source_summary:
        answer = _build_document_lookup_answer(sources)
        confidence = _compute_confidence(passages, answer)
        result = {
            "answer": answer,
            "sources": sources,
            "confidence_score": confidence,
        }
        await cache_answer(query, result)
        latency = (time.time() - start_time) * 1000
        doc_numbers = list({s.get("doc_number", "") for s in sources})
        await log_interaction(db, query, answer, doc_numbers, confidence, latency)
        return _with_conv_meta(result, conv_id)

    if prefer_multi_source_summary:
        answer = _build_multi_source_answer(query, sources)
        answer = _ensure_legal_citations(answer, sources)
        answer = _ensure_response_format(answer)
        confidence = _compute_confidence(passages, answer)
        result = {
            "answer": answer,
            "sources": sources,
            "confidence_score": confidence,
        }
        await cache_answer(query, result)
        latency = (time.time() - start_time) * 1000
        doc_numbers = list({s.get("doc_number", "") for s in sources})
        await log_interaction(db, query, answer, doc_numbers, confidence, latency)
        return _with_conv_meta(result, conv_id)

    # ── 4. Generate answer ───────────────────────────────
    prompt = RAG_PROMPT_TEMPLATE_V2.format(context=context, question=query)

    hist = conversation_store.get_history(conv_id, limit=20)
    conv_messages = [{"role": m["role"], "content": m["content"]} for m in hist if m.get("role") in ("user", "assistant")]
    if conv_messages:
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT_V2}]
            + conv_messages
            + [{"role": "user", "content": prompt}]
        )
        answer = await generate_with_messages(messages, temperature=temperature)
    else:
        answer = await generate(
            prompt=prompt,
            system=SYSTEM_PROMPT_V2,
            temperature=temperature,
        )

    # ── 5. Sanitize output (strip leaked metadata) ───────
    answer = _sanitize_output(answer)
    answer = _force_no_info_if_needed(answer)

    # If LLM says NO_INFO while context exists, retry with extractive instruction.
    if passages and _is_no_info_answer(answer):
        retry_prompt = (
            f"{prompt}\n\n"
            "BẮT BUỘC: Nếu NGỮ CẢNH có Điều/Khoản liên quan, hãy trích xuất trực tiếp từ NGỮ CẢNH.\n"
            "Không được trả lời 'Không tìm thấy...' trừ khi NGỮ CẢNH hoàn toàn không chứa nội dung pháp luật liên quan."
        )
        answer = await generate(
            prompt=retry_prompt,
            system=SYSTEM_PROMPT_V2,
            temperature=0.0,
        )
        answer = _sanitize_output(answer)
        answer = _force_no_info_if_needed(answer)

    # ── 5b. Context-query alignment: mức phạt vs thẩm quyền ─
    # When user asks about fine amounts but retrieved context describes enforcement authority,
    # regenerate with an authority-summary prompt so the answer clearly lists:
    # (a) that this is the authority article, (b) each entity's max fine limit,
    # (c) a pointer to the specific violation article for exact fine amounts.
    if (
        not _is_no_info_answer(answer)
        and _query_asks_fine_amount(query)
        and _context_describes_authority(context)
    ):
        log.info(
            "Context-query mismatch: query asks mức phạt but context is thẩm quyền – "
            "regenerating with authority summary."
        )
        answer = await _answer_with_authority_summary(query, context, sources, temperature)
        answer = _sanitize_output(answer)
        answer = _force_no_info_if_needed(answer)

    # ── 6. Article completeness check + fallback ─────────
    completeness = validate_article_completeness(
        answer, [{"text": p.get("text_chunk", "")} for p in passages]
    )
    if not completeness["is_complete"]:
        log.warning(
            "Incomplete article detected: %s. Re-retrieving full articles.",
            completeness["incomplete_articles"],
        )
        # Fallback: fetch complete article content and regenerate
        passages = await _fallback_full_article_retrieval(db, passages, query)
        context = _build_context(passages)
        sources = _extract_sources(passages)
        prompt = RAG_PROMPT_TEMPLATE_V2.format(context=context, question=query)
        answer = await generate(
            prompt=prompt,
            system=SYSTEM_PROMPT_V2,
            temperature=temperature,
        )
        answer = _sanitize_output(answer)

    # ── 7. Anti-hallucination: article mismatch guard ────
    expected_article = _normalize_article_number(passages[0].get("article_number")) if passages else None
    answer_article = _extract_article_reference(answer)
    if (
        expected_article
        and answer_article
        and answer_article != expected_article
        and _query_demands_specific_article(query)
    ):
        log.warning(
            "Answer article mismatch detected (expected=%s, got=%s). Returning NO_INFO.",
            expected_article,
            answer_article,
        )
        answer = NO_INFO_MESSAGE

    # ── 7b. Anti-hallucination: strip doc numbers not in context ─
    context_nums = _collect_context_doc_numbers(passages)
    answer = _strip_hallucinated_doc_numbers(answer, context_nums)

    # If answer still says no-info while we do have legal sources, downgrade to case-2 related-docs answer.
    if _is_no_info_answer(answer) and sources:
        answer = _build_related_documents_fallback(sources)

    # ── 8. Ensure legal citations and document list ─────
    answer = _ensure_legal_citations(answer, sources)
    answer = _ensure_document_list(answer, sources)
    answer = _ensure_response_format(answer)

    # ── 9. Compute confidence + V3 retry (multi-article) / fallback ──
    confidence = _compute_confidence(passages, answer)
    retried = False

    if (
        intent != "hoi_dap_chung"
        and confidence >= 0.20
        and confidence < ANSWER_VALIDATION_THRESHOLD
        and not use_multi
        and not multi_article_conditions
    ):
        from app.retrieval.article_selection import dynamic_max_articles, diversify_by_article

        passages2 = await hybrid_search(
            query=query,
            db=db,
            doc_number=doc_number,
            legal_domains=domain_filter,
            single_article_only=False,
            max_articles=MULTI_ARTICLE_MAX_ARTICLES,
        )
        if passages2:
            passages2 = diversify_by_article(passages2, min_docs=3)
            _ = dynamic_max_articles(passages2)
            groups2 = group_chunks_by_article(dedup_chunks(passages2))
            context2 = format_grouped_context(groups2)
            sources2 = _extract_sources(passages2)
            prompt2 = RAG_PROMPT_TEMPLATE_V2.format(context=context2, question=query)
            answer2 = await generate(
                prompt=prompt2,
                system=SYSTEM_PROMPT_V2,
                temperature=temperature,
            )
            answer2 = _sanitize_output(answer2)
            answer2 = _force_no_info_if_needed(answer2)
            context_nums2 = _collect_context_doc_numbers(passages2)
            answer2 = _strip_hallucinated_doc_numbers(answer2, context_nums2)
            if _is_no_info_answer(answer2) and sources2:
                answer2 = _build_related_documents_fallback(sources2)
            answer2 = _ensure_legal_citations(answer2, sources2)
            answer2 = _ensure_document_list(answer2, sources2)
            answer2 = _ensure_response_format(answer2)
            conf2 = _compute_confidence(passages2, answer2)
            prev_conf = confidence
            if conf2 >= ANSWER_VALIDATION_THRESHOLD or conf2 > confidence:
                passages, context, sources, answer, confidence = (
                    passages2,
                    context2,
                    sources2,
                    answer2,
                    conf2,
                )
                retried = True
                log.info(
                    "RAG validation retry: adopted multi-article path (confidence %.3f → %.3f)",
                    prev_conf,
                    conf2,
                )

    if (
        intent != "hoi_dap_chung"
        and confidence >= 0.20
        and confidence < ANSWER_VALIDATION_THRESHOLD
    ):
        try:
            from app.services.rag_chain import _fallback_reasoning

            fb = await _fallback_reasoning(query, intent)
            if fb and len(fb.strip()) > 20:
                answer = fb
                sources = []
                confidence = min(confidence, 0.35)
                log.info("RAG fallback_reasoning applied (confidence still below %.2f)", ANSWER_VALIDATION_THRESHOLD)
        except Exception as exc:
            log.error("fallback_reasoning failed: %s", exc)

    # ── 10. Cache result ─────────────────────────────────
    result = {
        "answer": answer,
        "sources": sources,
        "confidence_score": confidence,
    }
    await cache_answer(query, result)

    # ── 11. Log interaction + conversation memory ────────
    latency = (time.time() - start_time) * 1000
    doc_numbers = list({s.get("doc_number", "") for s in sources})
    await log_interaction(db, query, answer, doc_numbers, confidence, latency)
    try:
        conversation_store.add_message(conv_id, "user", query)
        conversation_store.add_message(conv_id, "assistant", answer)
    except Exception as exc:
        log.error("conversation_store add_message failed: %s", exc)

    return _with_conv_meta(result, conv_id, retried=retried)


async def rag_query_stream(
    query: str,
    db: AsyncSession,
    temperature: float = DEFAULT_TEMPERATURE,
    doc_number: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streaming wrapper that guarantees the same post-processed output as rag_query."""
    result = await rag_query(
        query=query,
        db=db,
        temperature=temperature,
        doc_number=doc_number,
        conversation_id=conversation_id,
    )

    meta_event = json.dumps(
        {
            "type": "meta",
            "conversation_id": result.get("conversation_id"),
            "retried": result.get("retried", False),
            "confidence_score": result.get("confidence_score", 0.0),
        },
        ensure_ascii=False,
    )
    yield meta_event + "\n"

    sources_event = json.dumps(
        {"type": "sources", "data": result.get("sources", [])},
        ensure_ascii=False,
    )
    yield sources_event + "\n"

    answer = result.get("answer", "") or NO_INFO_MESSAGE
    for i in range(0, len(answer), 80):
        yield answer[i : i + 80]


# ── Helpers ───────────────────────────────────────────────

# Patterns matching internal metadata JSON that should never appear in answers
_METADATA_JSON_PATTERN = re.compile(
    r'\{[^{}]*"(?:sources|confidence_score|document_id|vector_score|similarity|embedding)'
    r'[^{}]*\}',
    re.DOTALL,
)
_METADATA_FIELD_PATTERN = re.compile(
    r'"(?:sources|confidence_score|document_id|score|vector_score|similarity|embedding(?:_metadata)?)"'
    r'\s*:\s*(?:\[.*?\]|"[^"]*"|\d+(?:\.\d+)?|null|true|false)',
    re.DOTALL,
)


def _sanitize_output(text: str) -> str:
    """Remove any leaked internal metadata from the answer text.

    Strips JSON fragments containing internal fields like sources,
    confidence_score, document_id, score, vector_score, similarity.
    """
    if not text:
        return text

    # Remove full JSON objects containing metadata keys
    cleaned = _METADATA_JSON_PATTERN.sub("", text)

    # Remove standalone JSON-like fragments with metadata fields
    cleaned = _METADATA_FIELD_PATTERN.sub("", cleaned)

    # Clean up leftover empty braces/brackets and excessive whitespace
    cleaned = re.sub(r'\{\s*,?\s*\}', '', cleaned)
    cleaned = re.sub(r'\[\s*,?\s*\]', '', cleaned)
    cleaned = re.sub(r"```(?:\w+)?\n", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


_DOC_NUMBER_RE = re.compile(r"\b(\d+[/_]\d{4}[/_][A-ZĐa-zđ\-]+)\b")


def _extract_doc_numbers_from_text(text: str) -> set[str]:
    """Extract all document numbers (e.g. 01/2021/TT-BVHTTDL) from text."""
    return {m.group(1).replace("_", "/") for m in _DOC_NUMBER_RE.finditer(text or "")}


def _collect_context_doc_numbers(passages: List[Dict]) -> set[str]:
    """Collect all document numbers present in retrieved passages (context)."""
    nums: set[str] = set()
    for p in passages:
        nums |= _extract_doc_numbers_from_text(p.get("text_chunk", ""))
        nums |= _extract_doc_numbers_from_text(p.get("document_title", ""))
        nums |= _extract_doc_numbers_from_text(p.get("doc_number", ""))
    return {n.replace("_", "/") for n in nums}


def _normalize_for_comparison(doc_num: str) -> str:
    """Normalize doc number for comparison: NĐ→ND, QĐ→QD, lowercase."""
    s = doc_num.replace("_", "/").replace("Đ", "D").replace("đ", "d").lower()
    return s


def _strip_hallucinated_doc_numbers(answer: str, context_doc_numbers: set[str]) -> str:
    """Remove document references from the answer that don't exist in context."""
    if not context_doc_numbers or not answer:
        return answer

    context_normalized = {_normalize_for_comparison(n) for n in context_doc_numbers}
    answer_doc_numbers = _extract_doc_numbers_from_text(answer)

    hallucinated = {
        n for n in answer_doc_numbers
        if _normalize_for_comparison(n) not in context_normalized
    }

    if not hallucinated:
        return answer

    log.warning(
        "[ANTI-HALLUCINATION] Stripping doc numbers not in context: %s",
        hallucinated,
    )

    cleaned = answer
    for bad_num in hallucinated:
        cleaned = re.sub(
            rf"[^\n]*{re.escape(bad_num)}[^\n]*\n?",
            "",
            cleaned,
        )

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or NO_INFO_MESSAGE


def _format_legal_citation(sources: List[Dict]) -> str:
    """Convert source metadata into a readable legal citation block.

    Format:
        Căn cứ pháp lý:
        - <doc_number> – Điều <article_number>
    """
    if not sources:
        return ""

    lines = ["Căn cứ pháp lý:"]
    seen = set()
    for s in sources:
        citation = (s.get("citation", "") or "").strip()
        if not citation:
            label = _format_doc_label(s)
            article = _normalize_article_number(s.get("article_number"))
            citation = f"{label} – Điều {article}" if label and article else label
        if not citation or citation.lower() == "string":
            continue
        if citation and citation not in seen:
            seen.add(citation)
            lines.append(f"- {citation}")

    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def _ensure_legal_citations(answer: str, sources: List[Dict]) -> str:
    """Ensure the answer has a proper legal citation section.

    If the answer already contains 'Căn cứ pháp lý:' or 'Nguồn:', skip.
    Otherwise, append formatted citations.
    """
    if not sources:
        return answer

    if _is_no_info_answer(answer):
        return answer

    lower_answer = answer.lower()
    if "căn cứ pháp lý" in lower_answer or "nguồn:" in lower_answer:
        return answer

    citation_block = _format_legal_citation(sources)
    if citation_block:
        return f"{answer}\n\n{citation_block}"
    return answer


def _ensure_document_list(answer: str, sources: List[Dict]) -> str:
    """Ensure the answer includes a list of related legal documents from the DB."""
    if not sources or _is_no_info_answer(answer):
        return answer

    lower = answer.lower()
    if "các văn bản pháp luật liên quan" in lower or "danh sách văn bản" in lower:
        return answer

    docs = []
    seen = set()
    for s in sources:
        label = _format_doc_label(s)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        docs.append(label)

    if not docs:
        return answer

    doc_block = "Các văn bản pháp luật liên quan trong cơ sở dữ liệu hiện có:\n"
    for d in docs[:12]:
        doc_block += f"    {d}\n"

    # Insert after "Câu trả lời:" header if present
    if answer.lower().startswith("câu trả lời"):
        header_end = answer.find("\n")
        if header_end >= 0:
            return answer[:header_end + 1] + "\n" + doc_block + answer[header_end + 1:]

    return doc_block + "\n" + answer


def _ensure_response_format(answer: str) -> str:
    """Normalize answer to required format header."""
    if not answer or answer.strip() == NO_INFO_MESSAGE:
        return answer
    normalized = answer.strip()
    if not normalized.lower().startswith("câu trả lời"):
        normalized = f"Câu trả lời:\n\n{normalized}"
    return normalized


def _build_context(passages: List[Dict]) -> str:
    """Build context string from retrieved passages.

    Includes article metadata labels (chương, mục, điều, khoản) for better LLM grounding.
    """
    parts = []
    for i, p in enumerate(passages, 1):
        doc_title = p.get("document_title") or p.get("doc_number", "Văn bản pháp luật")
        chapter = (p.get("chapter") or "").strip()
        section = (p.get("section") or "").strip()
        article_number = _normalize_article_number(p.get("article_number"))
        article_title = p.get("article_title", "")
        clause_number = p.get("clause_number")
        text = p.get("text_chunk", "")

        label_parts = [f"Nguồn {i}: {doc_title}"]
        if chapter:
            label_parts.append(chapter)
        if section:
            label_parts.append(section)
        if article_number:
            if article_title:
                label_parts.append(f"Điều {article_number}. {article_title}")
            else:
                label_parts.append(f"Điều {article_number}")
        if clause_number:
            label_parts.append(f"Khoản {clause_number}")

        header = "[" + " | ".join(label_parts) + "]"
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(passages: List[Dict]) -> List[Dict]:
    """Convert raw retrieval metadata to human-readable legal citations."""
    sources = []
    seen = set()
    for p in passages:
        article_number = _normalize_article_number(p.get("article_number"))
        key = (p.get("doc_number"), article_number)
        if key in seen:
            continue
        seen.add(key)
        doc_title = p.get("document_title") or p.get("doc_number", "")
        doc_number = p.get("doc_number", "")
        article_title = p.get("article_title", "")
        doc_label = _format_doc_label({"doc_number": doc_number, "document_title": doc_title})
        citation = ""
        if doc_label and article_number:
            citation = f"{doc_label} – Điều {article_number}"
        elif doc_label:
            citation = doc_label

        sources.append({
            "citation": citation,
            "document_title": doc_title,
            "article_number": article_number,
            "article_title": article_title,
            "snippet": (p.get("text_chunk", "") or "")[:500],
            "document_id": p.get("document_id"),
            "article_id": p.get("article_id"),
            "clause_id": p.get("clause_id"),
            "doc_number": p.get("doc_number", ""),
            "score": None,
        })
    return sources


def _compute_confidence(passages: List[Dict], answer: str) -> float:
    """Estimate confidence based on retrieval scores and answer content."""
    if not passages:
        return 0.0

    # Average rerank score (or vector score fallback)
    scores = [p.get("rerank_score", p.get("score", 0.0)) for p in passages]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Boost if answer contains citations
    citation_boost = 0.0
    citation_markers = ["Điều", "Khoản", "Nghị định", "Thông tư", "Luật", "Quyết định"]
    for marker in citation_markers:
        if marker in answer:
            citation_boost += 0.05

    confidence = min(avg_score + citation_boost, 1.0)
    return round(confidence, 3)
