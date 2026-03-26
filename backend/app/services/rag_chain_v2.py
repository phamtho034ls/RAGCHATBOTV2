"""Production RAG chain with legal QA prompt engineering.

Enforces:
- Only answer using retrieved documents
- Always cite legal source (Article X of Document Y)
- Full article retrieval – never return partial articles
- Internal metadata never leaks into answer text
- Structured legal response format (Câu trả lời / Căn cứ pháp lý)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import (
    ANSWER_VALIDATION_THRESHOLD,
    DEFAULT_TEMPERATURE,
    MULTI_ARTICLE_MAX_ARTICLES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OUT_OF_SCOPE_USER_MESSAGE,
    QUERY_UTTERANCE_CLASSIFIER_ENABLED,
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
from app.services import conversation_repository as conv_repo
from app.services.llm_client import (
    generate,
    generate_stream,
    generate_with_messages,
    generate_with_messages_stream,
)
from app.services.answer_validator import (
    validate_article_completeness,
    validate_answer,
    get_fallback_answer,
)
from app.services.query_rewriter import rewrite_query
from app.services.query_features import extract_query_features
from app.services.strategy_router import (
    compute_strategy_scores,
    select_strategies,
    STRATEGY_LOOKUP,
    STRATEGY_MULTI_QUERY,
    STRATEGY_SEMANTIC,
)
from app.services.query_understanding import analyze_query
from app.services.query_text_patterns import (
    answer_contains_explicit_doc_number,
    article_sort_key_tuple,
    context_describes_authority,
    extract_article_numbers_mentioned_in_answer,
    extract_article_reference_from_text,
    extract_doc_numbers_from_text,
    normalize_article_number_canonical,
    normalize_doc_number_for_compare,
    query_asks_fine_amount,
    query_demands_specific_article,
    query_expects_llm_synthesis_from_context,
    query_asks_comprehensive_statutory_coverage,
    query_asks_structured_registration_conditions,
    query_looks_procedural,
    query_requests_prohibited_acts_list,
    sanitize_rag_llm_output,
    shorten_title_long_parenthetical,
    strip_answer_lines_with_hallucinated_doc_numbers,
    title_contains_tham_quyen,
)
from app.services.query_expansion import needs_expansion, expand_query, should_expand_query_v2
from app.services.intent_detector import get_rag_intents
from app.services.query_intent import query_requires_multi_document_synthesis
from app.services.article_grouper import (
    dedup_chunks,
    group_chunks_by_article,
    format_grouped_context,
)
from app.services.domain_classifier import classify_query_domain, get_domain_filter_values
from app.tools import draft_tool

log = logging.getLogger(__name__)

_STREAM_SENTINEL = object()

# Ngưỡng độ tin cậy để chọn câu hỏi gợi ý cuối câu trả lời (thấp vs cao).
FOLLOWUP_LOW_CONFIDENCE_THRESHOLD = 0.5


async def _persist_conv_turn(db: AsyncSession, conv_id: str, query: str, answer: str) -> None:
    try:
        if await conv_repo.add_message(db, conv_id, "user", query):
            await conv_repo.add_message(db, conv_id, "assistant", answer or "")
    except Exception as exc:
        log.error("persist conv turn failed: %s", exc)


async def _stream_emit_complete(
    stream_queue: Optional[asyncio.Queue],
    conv_id: str,
    result: Dict,
    *,
    retried: bool = False,
) -> None:
    """Luồng không LLM hoặc trả lời tức thì: meta + sources + text_finalize."""
    if stream_queue is None:
        return
    meta = {
        "type": "meta",
        "conversation_id": conv_id,
        "retried": retried,
        "confidence_score": float(result.get("confidence_score", 0.0) or 0.0),
    }
    await stream_queue.put(json.dumps(meta, ensure_ascii=False))
    await stream_queue.put(
        json.dumps({"type": "sources", "data": result.get("sources", [])}, ensure_ascii=False)
    )
    fin = {
        "type": "text_finalize",
        "text": result.get("answer", "") or "",
        "confidence_score": float(result.get("confidence_score", 0.0) or 0.0),
        "retried": retried,
    }
    await stream_queue.put(json.dumps(fin, ensure_ascii=False))


async def _stream_emit_hybrid_prelude(
    stream_queue: Optional[asyncio.Queue],
    conv_id: str,
    sources: List[Dict],
) -> None:
    if stream_queue is None:
        return
    await stream_queue.put(
        json.dumps(
            {
                "type": "meta",
                "conversation_id": conv_id,
                "retried": False,
                "confidence_score": 0.0,
            },
            ensure_ascii=False,
        )
    )
    await stream_queue.put(json.dumps({"type": "sources", "data": sources}, ensure_ascii=False))


async def _stream_emit_finalize(
    stream_queue: Optional[asyncio.Queue],
    result: Dict,
    retried: bool,
) -> None:
    if stream_queue is None:
        return
    fin = {
        "type": "text_finalize",
        "text": result.get("answer", "") or "",
        "confidence_score": float(result.get("confidence_score", 0.0) or 0.0),
        "retried": retried,
    }
    await stream_queue.put(json.dumps(fin, ensure_ascii=False))


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


async def _parallel_retrieve_all(
    query: str,
    db: AsyncSession,
    strategies: List[str],
    doc_number: Optional[str] = None,
    legal_domains: Optional[List[str]] = None,
    top_k: int = 20,
) -> List[Dict]:
    """Run multiple retrieval strategies in parallel and merge results.

    Each strategy runs concurrently via ``asyncio.gather``.  Results are
    deduplicated and re-sorted by the best available score.

    Args:
        query:        (Rewritten) user query for retrieval.
        db:           Async database session.
        strategies:   List of strategy names from ``select_strategies()``.
        doc_number:   Optional explicit document number filter.
        legal_domains:Optional domain filter values.
        top_k:        Passages to fetch per strategy.

    Returns:
        Merged, deduplicated list of passages sorted by score descending.
    """
    tasks = []

    for strategy in strategies:
        if strategy == STRATEGY_LOOKUP:
            # Lookup: hybrid search without multi-article, tight single-article focus
            tasks.append(
                hybrid_search(
                    query=query,
                    db=db,
                    top_k=top_k,
                    doc_number=doc_number,
                    single_article_only=True,
                    legal_domains=legal_domains,
                )
            )
        elif strategy == STRATEGY_MULTI_QUERY:
            # Multi-query: expand + gather sub-queries
            tasks.append(
                _multi_query_retrieve(
                    query=query,
                    db=db,
                    top_k=top_k,
                    doc_number=doc_number,
                    legal_domains=legal_domains,
                    force_expansion=True,
                )
            )
        else:  # STRATEGY_SEMANTIC / default
            tasks.append(
                hybrid_search(
                    query=query,
                    db=db,
                    top_k=top_k,
                    doc_number=doc_number,
                    single_article_only=False,
                    legal_domains=legal_domains,
                )
            )

    if not tasks:
        return []

    # Run all strategies in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged: List[Dict] = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            log.warning(
                "Parallel retrieval strategy %s failed: %s",
                strategies[i] if i < len(strategies) else "unknown",
                res,
            )
            continue
        merged.extend(res)  # type: ignore[arg-type]

    # Deduplicate and sort
    deduped = dedup_chunks(merged)
    deduped.sort(
        key=lambda p: float(
            p.get("rerank_score", p.get("rrf_score", p.get("score", 0.0)))
        ),
        reverse=True,
    )

    final_k = top_k * len(strategies)
    log.info(
        "Parallel retrieval: strategies=%s → %d raw → %d deduped → keeping %d",
        strategies,
        len(merged),
        len(deduped),
        min(len(deduped), final_k),
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


def _query_requests_comparison(query: str) -> bool:
    """Câu hỏi so sánh / đối chiếu — cần LLM tổng hợp, không dùng bản liệt kê đa nguồn."""
    q = (query or "").lower()
    return any(
        p in q
        for p in (
            "so sánh",
            "so sanh",
            "đối chiếu",
            "doi chieu",
            "khác nhau",
            "khac nhau",
            "khác gì",
            "khac gi",
            "điểm khác",
            "diem khac",
            "so với",
            "so voi",
        )
    )


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


def _select_single_article_passages(passages: List[Dict], query: str) -> List[Dict]:
    """Keep passages from only one best-matching article to prevent article mixing."""
    if not passages:
        return []

    query_article = extract_article_reference_from_text(query)
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for p in passages:
        art_no = normalize_article_number_canonical(p.get("article_number"))
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
    if query_asks_fine_amount(query):
        non_authority = {
            art: chunks
            for art, chunks in grouped.items()
            if not any(
                title_contains_tham_quyen(c.get("article_title", "") or "")
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
    return sanitize_rag_llm_output(answer)


def _force_no_info_if_needed(answer: str) -> str:
    """Normalize no-info variants to canonical response."""
    return NO_INFO_MESSAGE if _is_no_info_answer(answer) else answer


def _extract_doc_reference_from_query(query: str) -> Optional[str]:
    """Extract explicit legal document reference from user query."""
    q = (query or "").strip()
    if not q:
        return None
    m = re.search(r"(\d+/\d{4}/[A-ZĐa-zđ0-9\-]+)", q)
    if m:
        return m.group(1)
    m2 = re.search(
        r"(?:nghị\s*định|thông\s*tư|quyết\s*định|chỉ\s*thị|luật)\s+(?:số\s+)?(\d+/\d{4})",
        q,
        re.IGNORECASE,
    )
    return m2.group(1) if m2 else None


def _query_requires_direct_legal_lookup(query: str) -> bool:
    """True when query asks direct legal extraction, not procedural commune flow."""
    q = (query or "").lower()
    if not q:
        return False
    if _extract_doc_reference_from_query(query):
        return True
    legal_lookup_markers = (
        "trích xuất",
        "điều ",
        "khoản ",
        "theo nghị định",
        "theo quyết định",
        "theo luật",
        "thẩm quyền",
        "căn cứ pháp lý",
    )
    return any(k in q for k in legal_lookup_markers)


def _query_subject_anchor_phrases(query: str) -> List[str]:
    """Cụm từ đặc trưng chủ đề — dùng để tránh khớp nhầm văn bản cùng từ chung (chính sách…)."""
    q = (query or "").lower()
    anchors: List[str] = []
    if "khuyết tật" in q or "khuyet tat" in q:
        anchors.append("khuyết tật")
    if "thư viện" in q:
        anchors.append("thư viện")
    if "đầu tư công" in q:
        anchors.append("đầu tư công")
    if "trọng điểm quốc gia" in q or "dự án trọng điểm" in q:
        anchors.append("trọng điểm")
    if "phân loại dự án" in q or ("tiêu chí" in q and "dự án" in q):
        anchors.append("dự án")
    return anchors


def _passages_match_subject_anchors(passages: List[Dict], anchors: List[str]) -> bool:
    if not anchors or not passages:
        return True
    blob = " ".join(
        f"{p.get('document_title', '')} {p.get('text_chunk', '')}".lower()
        for p in passages[:10]
    )
    return any(a in blob for a in anchors)


def _query_topic_terms(query: str, *, max_terms: int = 6) -> List[str]:
    """Extract topic anchors from user query for lightweight mismatch guard."""
    q = (query or "").lower()
    tokens = re.findall(r"[0-9a-zà-ỹđ]+", q, re.IGNORECASE)
    stop = {
        "là",
        "về",
        "theo",
        "của",
        "cho",
        "và",
        "các",
        "những",
        "như",
        "thế",
        "nào",
        "quy",
        "định",
        "điều",
        "khoản",
        "tra",
        "cứu",
        "thẩm",
        "quyền",
        "quyết",
        "định",
        "văn",
        "bản",
        "pháp",
        "luật",
    }
    out: List[str] = []
    seen = set()
    for t in tokens:
        if len(t) < 4 or t in stop or t.isdigit():
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _has_topic_overlap(passages: List[Dict], topic_terms: List[str]) -> bool:
    if not passages or not topic_terms:
        return True
    text_blob = " ".join(
        f"{p.get('document_title', '')} {p.get('text_chunk', '')}".lower()
        for p in passages[:8]
    )
    # Một từ khóa chung ("chính", "quy định"…) dễ khớp nhầm văn bản khác lĩnh vực → cần ≥2 hit khi có đủ mốc.
    weak_singletons = frozenset(
        {"chính", "sách", "quy", "định", "nhà", "nước", "công", "dự", "án", "pháp"}
    )
    strong_hits = sum(1 for t in topic_terms if t in text_blob and t not in weak_singletons)
    weak_hits = sum(1 for t in topic_terms if t in text_blob and t in weak_singletons)
    if len(topic_terms) >= 3:
        return strong_hits >= 2 or (strong_hits >= 1 and weak_hits >= 2)
    if len(topic_terms) >= 2:
        return strong_hits >= 1 and (strong_hits + max(0, weak_hits - 1)) >= 2
    return strong_hits + weak_hits >= 1


def _passages_match_explicit_doc_ref(passages: List[Dict], query: str) -> List[Dict]:
    """Keep only passages that match explicit doc reference in query."""
    doc_ref = _extract_doc_reference_from_query(query)
    if not doc_ref:
        return passages
    ref_norm = normalize_doc_number_for_compare(doc_ref)
    matched: List[Dict] = []
    for p in passages:
        dn = normalize_doc_number_for_compare((p.get("doc_number") or ""))
        dt = normalize_doc_number_for_compare((p.get("document_title") or ""))
        if ref_norm in dn or ref_norm in dt:
            matched.append(p)
    return matched


def _format_doc_label(source: Dict) -> str:
    """Build human-readable label that always prioritizes legal document number."""
    doc_number = (source.get("doc_number") or "").strip()
    doc_title = (source.get("document_title") or "").strip()
    # Tiêu đề quá dài, ít khoảng trắng → thường là chuỗi ghép lỗi khi ingest
    if doc_number and doc_title and len(doc_title) > 90 and doc_title.count(" ") < 5:
        return doc_number
    if doc_number and doc_title:
        if doc_number.lower() in doc_title.lower():
            return doc_title
        return f"{doc_number} ({doc_title})"
    return doc_number or doc_title


def _ensure_explicit_document_numbers(answer: str, sources: List[Dict]) -> str:
    """Ensure answer lists exact document numbers for legal document questions."""
    if not sources:
        return answer
    if answer_contains_explicit_doc_number(answer):
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
        article_number = normalize_article_number_canonical(meta.get("article_number"))
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
    retrieval_query: Optional[str] = None,
) -> Dict:
    """Answer synthesis/checklist queries by aggregating multiple legal documents."""
    _rq = (retrieval_query or query).strip() or query
    passages = await hybrid_search(
        query=_rq,
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
I. Danh mục văn bản pháp lý liên quan — với mỗi văn bản ưu tiên một dòng dạng:
   `- Số hiệu (vd. 38/2021/NĐ-CP): Điều 36, 56, 57` (gom các Điều cùng văn bản, không lặp tên văn bản cho từng Điều).
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
    answer = sanitize_rag_llm_output(answer)
    answer = _force_no_info_if_needed(answer)

    context_nums = _collect_context_doc_numbers(passages)
    answer = strip_answer_lines_with_hallucinated_doc_numbers(answer, context_nums)

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
    retrieval_query: Optional[str] = None,
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
    _rq = (retrieval_query or query).strip() or query

    if needs_expansion(_rq):
        passages = await _multi_query_retrieve(
            query=_rq, db=db, top_k=20, doc_number=doc_number,
            legal_domains=legal_domains,
        )
    else:
        passages = await hybrid_search(
            query=_rq,
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
        answer = sanitize_rag_llm_output(answer)
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
    answer = sanitize_rag_llm_output(answer)

    context_nums = _collect_context_doc_numbers(passages)
    answer = strip_answer_lines_with_hallucinated_doc_numbers(answer, context_nums)

    if _is_no_info_answer(answer) and sources:
        answer = _build_related_documents_fallback(sources)

    answer = _ensure_legal_citations(answer, sources)
    confidence = _compute_confidence(passages, answer)

    return {"answer": answer, "sources": sources, "confidence_score": confidence}


async def _answer_drafting_query(query: str, temperature: float) -> Dict:
    """Answer administrative/legal drafting queries using dedicated drafting tool."""
    tool_result = await draft_tool.run(content=query, temperature=min(temperature, 0.35))
    answer = sanitize_rag_llm_output(tool_result.get("result", ""))
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
    stream_queue: Optional[asyncio.Queue] = None,
    utterance_labels: Optional[Any] = None,
) -> Dict:
    """Run the full RAG pipeline: retrieve → build context → generate answer.

    Includes fallback safety: if the answer references an article but misses
    clauses, re-retrieves the full article and regenerates.

    Returns dict with: answer, sources, confidence_score
    """
    start_time = time.time()

    if conversation_id and await conv_repo.conversation_exists(db, conversation_id):
        conv_id = conversation_id
    else:
        created = await conv_repo.create_conversation(db, title=None)
        conv_id = created["id"]

    # ── 0. Ninh Bình tool: câu hỏi phi pháp lý về Ninh Bình (địa lý, huyện, du lịch...)
    from app.services.ninh_binh_router import should_use_ninh_binh_tool, route_to_ninh_binh
    if should_use_ninh_binh_tool(query):
        nb_result = await route_to_ninh_binh(query)
        if nb_result:
            latency = (time.time() - start_time) * 1000
            result = {
                "answer": _append_followup_prompts(
                    nb_result.get("answer", "") or "",
                    0.85,
                    has_sources=bool(nb_result.get("sources")),
                ),
                "sources": nb_result.get("sources", []),
                "confidence_score": 0.85,
            }
            await log_interaction(db, query, result.get("answer", "") or "", [], 0.85, latency)
            await _persist_conv_turn(db, conv_id, query, result.get("answer", "") or "")
            await _stream_emit_complete(stream_queue, conv_id, result)
            return _with_conv_meta(result, conv_id)

    analysis = analyze_query(query)
    if utterance_labels is not None:
        from app.services.query_route_classifier import merge_utterance_labels_into_analysis

        analysis = merge_utterance_labels_into_analysis(
            analysis, utterance_labels, query=query
        )
    elif QUERY_UTTERANCE_CLASSIFIER_ENABLED and OPENAI_API_KEY:
        from app.services.query_route_classifier import (
            classify_user_utterance,
            merge_utterance_labels_into_analysis,
        )

        _ul = await classify_user_utterance(query, has_conversation=bool(conv_id))
        if _ul is not None:
            analysis = merge_utterance_labels_into_analysis(analysis, _ul, query=query)

    intent = analysis.get("intent", "")
    detector_intent = str(analysis.get("detector_intent", "") or "")
    det_for_scope = detector_intent
    if det_for_scope == "nan" or intent == "out_of_scope":
        latency = (time.time() - start_time) * 1000
        oos = {
            "answer": OUT_OF_SCOPE_USER_MESSAGE,
            "sources": [],
            "confidence_score": 0.0,
            "query_analysis": analysis,
        }
        await log_interaction(db, query, oos["answer"], [], 0.0, latency)
        await _persist_conv_turn(db, conv_id, query, oos["answer"])
        await _stream_emit_complete(stream_queue, conv_id, oos)
        return _with_conv_meta(oos, conv_id)

    rag_intents = analysis.get("rag_flags") or _get_rag_intent_flags(query)

    # ── 1a. Legal domain for retrieval (all paths: commune, checklist, default) ──
    domain_filter = get_domain_filter_values(query)
    if domain_filter:
        domain_info = classify_query_domain(query, top_n=2)
        log.info(
            "Domain classification: %s → filter=%s",
            [f"{d['domain']}({d['confidence']:.2f})" for d in domain_info],
            domain_filter,
        )

    # ── 1b. RAG intent flags — cùng nguồn với analyze_query (query_intent bundle) ──
    log.info(
        "RAG intents (query_intent / analysis): scenario=%s legal_lookup=%s multi_article=%s needs_expansion=%s",
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
                out = dict(cached)
                await _persist_conv_turn(db, conv_id, query, out.get("answer", "") or "")
                await _stream_emit_complete(
                    stream_queue, conv_id, out, retried=out.get("retried", False)
                )
                return _with_conv_meta(out, conv_id, retried=out.get("retried", False))

    # ── 0.5–0.6. Rewrite + strategy scoring (sau cache miss; tránh LLM khi cache hit) ──
    # Soạn thảo / tóm tắt văn bản không đi retrieval hybrid → bỏ rewrite tiết kiệm gọi LLM.
    if intent in {"document_drafting", "document_summary"}:
        rewritten_query = query
    else:
        rewritten_query = await rewrite_query(query)
        if rewritten_query != query:
            log.info(
                "Query rewrite applied | original='%s' → retrieval='%s'",
                query[:80],
                rewritten_query[:100],
            )
    _q_features = extract_query_features(rewritten_query)
    _strategy_scores = compute_strategy_scores(_q_features)
    _selected_strategies = select_strategies(_strategy_scores, top_k=2)
    log.info(
        "Strategy routing | features=%s → scores=%s → selected=%s",
        {k: v for k, v in _q_features.items() if v},
        {k: round(v, 2) for k, v in _strategy_scores.items()},
        _selected_strategies,
    )

    # ── Commune officer pipeline: semantic margin + LLM khi mơ hồ ──
    from app.services.intent_detector import COMMUNE_LEVEL_INTENTS
    from app.services.commune_route_arbiter import resolve_use_commune_officer_pipeline

    commune_situation = analysis.get("commune_situation")
    legacy_commune_hint = (
        intent in COMMUNE_LEVEL_INTENTS
        or (commune_situation and commune_situation.get("violation", "không có") != "không có")
        or rag_intents.get("is_scenario", False)
    )
    if query_requires_multi_document_synthesis(query) or _query_requires_direct_legal_lookup(query):
        is_commune_query = False
    else:
        is_commune_query = await resolve_use_commune_officer_pipeline(
            query, legacy_commune_hint=legacy_commune_hint
        )

    if is_commune_query:
        result = await _answer_commune_officer_query(
            query=query,
            db=db,
            temperature=temperature,
            doc_number=doc_number,
            legal_domains=domain_filter,
            retrieval_query=rewritten_query,
        )
        await cache_answer(query, result)
        latency = (time.time() - start_time) * 1000
        doc_numbers = list({s.get("doc_number", "") for s in result.get("sources", [])})
        result["answer"] = _append_followup_prompts(
            result.get("answer", "") or "",
            float(result.get("confidence_score", 0.0) or 0.0),
            has_sources=bool(result.get("sources")),
        )
        await log_interaction(
            db, query, result.get("answer", ""),
            doc_numbers, result.get("confidence_score", 0.0), latency,
        )
        await _persist_conv_turn(db, conv_id, query, result.get("answer", "") or "")
        await _stream_emit_complete(stream_queue, conv_id, result)
        return _with_conv_meta(result, conv_id)

    # Intent-specific routes
    if intent == "checklist_documents":
        result = await _answer_checklist_query(
            query=query,
            db=db,
            temperature=temperature,
            doc_number=doc_number,
            legal_domains=domain_filter,
            retrieval_query=rewritten_query,
        )
        result["answer"] = _append_followup_prompts(
            result.get("answer", "") or "",
            float(result.get("confidence_score", 0.0) or 0.0),
            has_sources=bool(result.get("sources")),
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
        await _persist_conv_turn(db, conv_id, query, result.get("answer", "") or "")
        await _stream_emit_complete(stream_queue, conv_id, result)
        return _with_conv_meta(result, conv_id)

    if intent == "document_drafting":
        result = await _answer_drafting_query(query=query, temperature=temperature)
        result["answer"] = _append_followup_prompts(
            result.get("answer", "") or "",
            float(result.get("confidence_score", 0.0) or 0.0),
            has_sources=bool(result.get("sources")),
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
        await _persist_conv_turn(db, conv_id, query, result.get("answer", "") or "")
        await _stream_emit_complete(stream_queue, conv_id, result)
        return _with_conv_meta(result, conv_id)

    if intent == "document_summary":
        from app.services.document_summarizer import summarize_matched_document
        summary_result = await summarize_matched_document(
            query=query,
            temperature=min(temperature, 0.25),
        )
        result = {
            "answer": _append_followup_prompts(
                summary_result.get("summary", "") or "",
                float(summary_result.get("confidence_score", 0.0) or 0.0),
                has_sources=bool(summary_result.get("sources")),
            ),
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
        await _persist_conv_turn(db, conv_id, query, result.get("answer", "") or "")
        await _stream_emit_complete(stream_queue, conv_id, result)
        return _with_conv_meta(result, conv_id)

    # ── 2. Hybrid retrieval (feature-based strategy + multi-query expansion) ──
    # Use the rewritten query for retrieval; fall back to original if unchanged.
    _retrieval_query = rewritten_query

    initial_for_expand = vector_search(
        query=_retrieval_query,
        top_k=8,
        doc_number=doc_number,
        legal_domains=domain_filter,
    )
    use_multi = (
        rag_intents.get("needs_expansion", False)
        or should_expand_query_v2(_retrieval_query, initial_for_expand)
        or STRATEGY_MULTI_QUERY in _selected_strategies
    )
    multi_article_conditions = (
        rag_intents.get("use_multi_article", False) and USE_MULTI_ARTICLE_FOR_CONDITIONS
    )
    if query_requests_prohibited_acts_list(query):
        multi_article_conditions = True
        use_multi = True

    # When multiple strategies selected and not already handled by intents,
    # run them in parallel for richer context coverage.
    _use_parallel = (
        len(_selected_strategies) >= 2
        and not use_multi
        and not multi_article_conditions
        and STRATEGY_LOOKUP not in _selected_strategies  # lookup still uses single-path
    )

    if use_multi:
        passages = await _multi_query_retrieve(
            query=_retrieval_query, db=db, doc_number=doc_number,
            legal_domains=domain_filter,
            force_expansion=use_multi,
        )
    elif _use_parallel:
        passages = await _parallel_retrieve_all(
            query=_retrieval_query,
            db=db,
            strategies=_selected_strategies,
            doc_number=doc_number,
            legal_domains=domain_filter,
        )
    else:
        passages = await hybrid_search(
            query=_retrieval_query,
            db=db,
            doc_number=doc_number,
            legal_domains=domain_filter,
            single_article_only=not multi_article_conditions,
            max_articles=MULTI_ARTICLE_MAX_ARTICLES if multi_article_conditions else None,
        )

    # If user cites explicit document number, do not answer from other documents.
    strict_doc_passages = _passages_match_explicit_doc_ref(passages, query)
    if _extract_doc_reference_from_query(query):
        passages = strict_doc_passages

    # Lightweight topic mismatch guard: avoid answering from unrelated legal domain.
    # Use original query for topic guard (topic terms reflect user intent, not rewrite).
    topic_terms = _query_topic_terms(query)
    anchors = _query_subject_anchor_phrases(query)
    if (
        passages
        and anchors
        and not _passages_match_subject_anchors(passages, anchors)
        and not _extract_doc_reference_from_query(query)
    ):
        log.warning(
            "Subject anchor mismatch (anchors=%s), retrying retrieval with anchored query",
            anchors,
        )
        topic_query = " ".join(anchors[:3]) + " " + _retrieval_query
        passages_retry = await hybrid_search(
            query=topic_query.strip(),
            db=db,
            doc_number=doc_number,
            legal_domains=domain_filter,
            single_article_only=not multi_article_conditions,
            max_articles=MULTI_ARTICLE_MAX_ARTICLES if multi_article_conditions else None,
        )
        if _passages_match_subject_anchors(passages_retry, anchors) or _has_topic_overlap(
            passages_retry, topic_terms
        ):
            passages = passages_retry
    if passages and not _has_topic_overlap(passages, topic_terms) and not _extract_doc_reference_from_query(query):
        log.warning("Topic mismatch suspected, retrying retrieval with topic anchors: %s", topic_terms)
        topic_query = " ".join(topic_terms[:4]) + " " + _retrieval_query
        passages_retry = await hybrid_search(
            query=topic_query.strip(),
            db=db,
            doc_number=doc_number,
            legal_domains=domain_filter,
            single_article_only=not multi_article_conditions,
            max_articles=MULTI_ARTICLE_MAX_ARTICLES if multi_article_conditions else None,
        )
        if _has_topic_overlap(passages_retry, topic_terms):
            passages = passages_retry

    # ── 3. Build context ─────────────────────────────────
    if not passages and query_looks_procedural(query):
        # Procedural queries are brittle with strict domain filters/doc constraints:
        # retry once with relaxed filters before returning NO_INFO.
        log.info("Procedural query with empty retrieval — retrying relaxed search.")
        relaxed_doc = None
        if doc_number:
            relaxed_doc = doc_number
        passages = await hybrid_search(
            query=query,
            db=db,
            doc_number=relaxed_doc,
            legal_domains=None,
            single_article_only=False,
            max_articles=MULTI_ARTICLE_MAX_ARTICLES,
        )

    if not passages:
        answer = _append_followup_prompts(NO_INFO_MESSAGE, 0.0, has_sources=False)
        result = {
            "answer": answer,
            "sources": [],
            "confidence_score": 0.0,
        }
        latency = (time.time() - start_time) * 1000
        await log_interaction(db, query, answer, [], 0.0, latency)
        await _persist_conv_turn(db, conv_id, query, answer)
        await _stream_emit_complete(stream_queue, conv_id, result)
        return _with_conv_meta(result, conv_id)

    # Lấy đủ khoản trong cùng Điều (tránh chỉ đoạn rút gọn khi hỏi điều kiện / chính sách / tiêu chí).
    if passages and (
        query_asks_structured_registration_conditions(query)
        or query_asks_comprehensive_statutory_coverage(query)
    ):
        passages = await _fallback_full_article_retrieval(db, passages, query)

    # For multi-article / condition queries, group by article for structured context
    if use_multi or multi_article_conditions or any(p.get("_db_lookup") for p in passages):
        groups = group_chunks_by_article(dedup_chunks(passages))
        context = format_grouped_context(groups)
    else:
        if not any(p.get("_db_lookup") for p in passages):
            passages = _select_single_article_passages(passages, query)
        context = _build_context(passages)
    sources = _extract_sources(passages)

    # Query type: user asks "which legal document(s)".
    if _is_document_lookup_query(query):
        answer = _build_document_lookup_answer(sources)
        confidence = _compute_confidence(passages, answer)
        answer = _append_followup_prompts(answer, confidence, has_sources=bool(sources))
        result = {
            "answer": answer,
            "sources": sources,
            "confidence_score": confidence,
        }
        await cache_answer(query, result)
        latency = (time.time() - start_time) * 1000
        doc_numbers = list({s.get("doc_number", "") for s in sources})
        await log_interaction(db, query, answer, doc_numbers, confidence, latency)
        await _persist_conv_turn(db, conv_id, query, answer)
        await _stream_emit_complete(stream_queue, conv_id, result)
        return _with_conv_meta(result, conv_id)

    # ── 4. Generate answer (luôn qua LLM khi có passages — tránh stub chỉ liệt kê văn bản) ──
    prompt = _build_rag_user_prompt(query, context)

    hist = await conv_repo.get_history(db, conv_id, limit=20)
    conv_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in hist
        if m.get("role") in ("user", "assistant")
    ]
    if conv_messages:
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT_V2}]
            + conv_messages
            + [{"role": "user", "content": prompt}]
        )
        if stream_queue is not None:
            await _stream_emit_hybrid_prelude(stream_queue, conv_id, sources)
            answer = ""
            async for delta in generate_with_messages_stream(messages, temperature=temperature):
                answer += delta
                await stream_queue.put(delta)
        else:
            answer = await generate_with_messages(messages, temperature=temperature)
    else:
        if stream_queue is not None:
            await _stream_emit_hybrid_prelude(stream_queue, conv_id, sources)
            answer = ""
            async for delta in generate_stream(
                prompt=prompt,
                system=SYSTEM_PROMPT_V2,
                temperature=temperature,
            ):
                answer += delta
                await stream_queue.put(delta)
        else:
            answer = await generate(
                prompt=prompt,
                system=SYSTEM_PROMPT_V2,
                temperature=temperature,
            )

    # ── 5. Sanitize output (strip leaked metadata) ───────
    answer = sanitize_rag_llm_output(answer)
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
        answer = sanitize_rag_llm_output(answer)
        answer = _force_no_info_if_needed(answer)

    # ── 5b. Context-query alignment: mức phạt vs thẩm quyền ─
    # When user asks about fine amounts but retrieved context describes enforcement authority,
    # regenerate with an authority-summary prompt so the answer clearly lists:
    # (a) that this is the authority article, (b) each entity's max fine limit,
    # (c) a pointer to the specific violation article for exact fine amounts.
    if (
        not _is_no_info_answer(answer)
        and query_asks_fine_amount(query)
        and context_describes_authority(context)
    ):
        log.info(
            "Context-query mismatch: query asks mức phạt but context is thẩm quyền – "
            "regenerating with authority summary."
        )
        answer = await _answer_with_authority_summary(query, context, sources, temperature)
        answer = sanitize_rag_llm_output(answer)
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
        prompt = _build_rag_user_prompt(query, context)
        answer = await generate(
            prompt=prompt,
            system=SYSTEM_PROMPT_V2,
            temperature=temperature,
        )
        answer = sanitize_rag_llm_output(answer)

    # ── 7. Anti-hallucination: article mismatch guard ────
    expected_article = normalize_article_number_canonical(passages[0].get("article_number")) if passages else None
    answer_article = extract_article_reference_from_text(answer)
    if (
        expected_article
        and answer_article
        and answer_article != expected_article
        and query_demands_specific_article(query)
    ):
        log.warning(
            "Answer article mismatch detected (expected=%s, got=%s). Returning NO_INFO.",
            expected_article,
            answer_article,
        )
        answer = NO_INFO_MESSAGE

    # ── 7b. Anti-hallucination: strip doc numbers not in context ─
    context_nums = _collect_context_doc_numbers(passages)
    answer = strip_answer_lines_with_hallucinated_doc_numbers(answer, context_nums)

    # If answer still says no-info while we do have legal sources, downgrade to case-2 related-docs answer.
    if _is_no_info_answer(answer) and sources:
        answer = _build_related_documents_fallback(sources)

    # ── 7c. LLM-based answer validation (groundedness + hallucination check) ──
    # Only run when there is a real answer and real context (skip for no-info responses).
    if not _is_no_info_answer(answer) and passages and OPENAI_API_KEY:
        _validation = await validate_answer(
            query=query,
            context=context,
            answer=answer,
        )

        if not _validation["is_valid"]:
            # Attempt regeneration with stricter, extractive prompt
            _strict_prompt = (
                f"{_build_rag_user_prompt(query, context)}\n\n"
                "━━ KIỂM TRA NGHIÊM NGẶT ━━\n"
                "Câu trả lời phải BÁM SÁT HOÀN TOÀN vào NGỮ CẢNH bên trên.\n"
                "KHÔNG được đề cập bất kỳ văn bản, điều luật, số hiệu nào\n"
                "KHÔNG có trong NGỮ CẢNH.\n"
                "Nếu ngữ cảnh không đủ thông tin, trả lời:\n"
                f'"{get_fallback_answer()}"\n'
            )
            _retry_answer = await generate(
                prompt=_strict_prompt,
                system=SYSTEM_PROMPT_V2,
                temperature=0.0,
            )
            _retry_answer = sanitize_rag_llm_output(_retry_answer)
            _retry_answer = _force_no_info_if_needed(_retry_answer)

            if _retry_answer and not _is_no_info_answer(_retry_answer):
                log.info(
                    "Validation retry succeeded — replacing answer after failed validation."
                )
                answer = _retry_answer
            else:
                # Regeneration also failed → use structured fallback
                log.warning(
                    "Validation retry produced no-info — using fallback answer. "
                    "Issues: %s",
                    _validation.get("issues", []),
                )
                answer = get_fallback_answer()

    # ── 8. Ensure legal citations (no auto-inserted DB document list in body) ─────
    answer = _ensure_legal_citations(answer, sources)
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
            _ = dynamic_max_articles(passages2, query)
            groups2 = group_chunks_by_article(dedup_chunks(passages2))
            context2 = format_grouped_context(groups2)
            sources2 = _extract_sources(passages2)
            prompt2 = _build_rag_user_prompt(query, context2)
            answer2 = await generate(
                prompt=prompt2,
                system=SYSTEM_PROMPT_V2,
                temperature=temperature,
            )
            answer2 = sanitize_rag_llm_output(answer2)
            answer2 = _force_no_info_if_needed(answer2)
            context_nums2 = _collect_context_doc_numbers(passages2)
            answer2 = strip_answer_lines_with_hallucinated_doc_numbers(answer2, context_nums2)
            if _is_no_info_answer(answer2) and sources2:
                answer2 = _build_related_documents_fallback(sources2)
            answer2 = _ensure_legal_citations(answer2, sources2)
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

    answer = _append_followup_prompts(answer, confidence, has_sources=bool(sources))

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
    await _stream_emit_finalize(stream_queue, result, retried)
    await _persist_conv_turn(db, conv_id, query, result.get("answer", "") or "")

    return _with_conv_meta(result, conv_id, retried=retried)


async def rag_query_stream(
    query: str,
    db: AsyncSession,
    temperature: float = DEFAULT_TEMPERATURE,
    doc_number: Optional[str] = None,
    conversation_id: Optional[str] = None,
    utterance_labels: Optional[Any] = None,
) -> AsyncGenerator[str, None]:
    """SSE payload strings: JSON events (meta, sources, text_finalize) và chunk text thô từ LLM."""
    queue: asyncio.Queue = asyncio.Queue()

    async def runner() -> None:
        try:
            await rag_query(
                query=query,
                db=db,
                temperature=temperature,
                doc_number=doc_number,
                conversation_id=conversation_id,
                stream_queue=queue,
                utterance_labels=utterance_labels,
            )
        finally:
            await queue.put(_STREAM_SENTINEL)

    task = asyncio.create_task(runner())
    try:
        while True:
            item = await queue.get()
            if item is _STREAM_SENTINEL:
                break
            yield item
    finally:
        await task


# ── Helpers ───────────────────────────────────────────────


def _collect_context_doc_numbers(passages: List[Dict]) -> set[str]:
    """Collect all document numbers present in retrieved passages (context)."""
    nums: set[str] = set()
    for p in passages:
        nums |= extract_doc_numbers_from_text(p.get("text_chunk", ""))
        nums |= extract_doc_numbers_from_text(p.get("document_title", ""))
        nums |= extract_doc_numbers_from_text(p.get("doc_number", ""))
    return {n.replace("_", "/") for n in nums}


def _citation_group_key(source: Dict) -> str:
    dn = (source.get("doc_number") or "").strip().replace("_", "/")
    if dn:
        return normalize_doc_number_for_compare(dn)
    t = (source.get("document_title") or "").strip()[:120]
    return t.lower() or "__unknown__"


def _short_citation_label(source: Dict) -> str:
    """Nhãn ngắn cho căn cứ: ưu tiên số hiệu; tránh lặp mô tả dài trong ngoặc."""
    dn = (source.get("doc_number") or "").strip().replace("_", "/")
    if dn:
        return dn
    title = (source.get("document_title") or "").strip()
    if not title:
        return "Văn bản"
    title = shorten_title_long_parenthetical(title)
    return title if len(title) <= 220 else title[:217] + "…"


def _strip_trailing_legal_basis(answer: str) -> tuple[str, bool]:
    """Gỡ khối 'Căn cứ pháp lý' ở CUỐI bài. Trả (text, đã_gỡ)."""
    if not answer:
        return answer, False
    lower = answer.lower()
    key = "căn cứ pháp lý"
    idx = lower.rfind(key)
    if idx < 0:
        return answer, False
    # Chỉ gỡ khi mục căn cứ nằm ở nửa sau bài (tránh cắt nhầm mục giữa template cán bộ xã)
    if len(answer) > 100 and idx < int(len(answer) * 0.35):
        return answer, False
    return answer[:idx].rstrip(), True


def _format_legal_citation(sources: List[Dict], answer: str = "") -> str:
    """Căn cứ pháp lý gom theo văn bản: `- Số hiệu: Điều 1, 2, 3`.

    Chỉ gom các Điều thực sự xuất hiện trong phần trả lời (tránh liệt kê mọi Điều đã retrieve).
    """
    if not sources:
        return ""

    mentioned = extract_article_numbers_mentioned_in_answer(answer) if answer else set()
    cite_pool = list(sources)
    if mentioned:
        filtered: List[Dict] = []
        for s in sources:
            art = normalize_article_number_canonical(s.get("article_number"))
            if not art:
                filtered.append(s)
            elif art in mentioned:
                filtered.append(s)
        if filtered:
            cite_pool = filtered
        else:
            cite_pool = list(sources)[:10]
    else:
        cite_pool = list(sources)[:12]

    groups: Dict[str, Dict[str, object]] = {}
    order_keys: List[str] = []

    for s in cite_pool:
        key = _citation_group_key(s)
        if key not in groups:
            groups[key] = {"label": _short_citation_label(s), "articles": []}
            order_keys.append(key)
        art = normalize_article_number_canonical(s.get("article_number"))
        if art:
            groups[key]["articles"].append(art)  # type: ignore[index]

    lines = ["Căn cứ pháp lý:"]
    for key in order_keys:
        g = groups[key]
        label = str(g["label"])
        if label.lower() == "string":
            continue
        arts_list: List[str] = g["articles"]  # type: ignore[assignment]
        seen_a: set = set()
        ordered: List[str] = []
        for a in arts_list:
            if a and a not in seen_a:
                seen_a.add(a)
                ordered.append(a)
        ordered.sort(key=article_sort_key_tuple)
        if ordered:
            lines.append(f"- {label}: Điều {', '.join(ordered)}")
        else:
            lines.append(f"- {label}")

    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def _ensure_legal_citations(answer: str, sources: List[Dict]) -> str:
    """Gắn khối căn cứ gom Điều từ metadata; thay khối căn cứ cuối LLM nếu gỡ được."""
    if not sources:
        return answer

    if _is_no_info_answer(answer):
        return answer

    trimmed, stripped = _strip_trailing_legal_basis(answer)
    trimmed = trimmed.rstrip()
    citation_block = _format_legal_citation(sources, answer=trimmed)
    if not citation_block:
        return answer
    if not stripped and "căn cứ pháp lý" in answer.lower():
        # Đã có mục căn cứ ở giữa bài (vd. pipeline cán bộ xã) — không thêm khối trùng
        return answer
    return f"{trimmed}\n\n{citation_block}"


def _append_followup_prompts(
    answer: str,
    confidence: float,
    *,
    has_sources: bool = True,
) -> str:
    """Câu hỏi dẫn dắt cuối phản hồi — phụ thuộc độ tin cậy ước lượng."""
    a = (answer or "").rstrip()
    if not a:
        return a

    leader = "Để tiếp tục hỗ trợ phù hợp hơn, xin ghi nhận ý kiến của anh/chị:"

    if _is_no_info_answer(a):
        return (
            f"{a}\n\n---\n{leader} Anh/chị có thể nêu thêm bối cảnh, số hiệu văn bản hoặc "
            "diễn đạt lại câu hỏi để hệ thống tra cứu chính xác hơn không?"
        )

    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0

    if not has_sources:
        tail = (
            f"{leader} Nội dung trên mang tính định hướng chung. Anh/chị cần tra cứu thêm "
            "văn bản cụ thể hay chi tiết nào nữa không?"
        )
    elif conf < FOLLOWUP_LOW_CONFIDENCE_THRESHOLD:
        tail = (
            f"{leader} Anh/chị xem phần trả lời và căn cứ pháp lý đã giải đúng thắc mắc chưa? "
            "Nếu chưa, vui lòng cho biết phần còn thiếu hoặc cần làm rõ thêm."
        )
    else:
        tail = (
            f"{leader} Anh/chị cần hỗ trợ thêm nội dung nào nữa "
            "(điều khoản chi tiết, thủ tục, thẩm quyền, mức phạt, so sánh văn bản…) không?"
        )

    return f"{a}\n\n---\n{tail}"


def _build_rag_user_prompt(query: str, context: str) -> str:
    base = RAG_PROMPT_TEMPLATE_V2.format(context=context, question=query)
    if _query_requests_comparison(query):
        base += (
            "\n\n━━ YÊU CẦU SO SÁNH/ĐỐI CHIẾU ━━\n"
            "Câu hỏi yêu cầu so sánh hoặc nêu điểm khác giữa các văn bản. Bắt buộc: "
            "(1) nêu quy định chính theo TỪNG số hiệu văn bản; "
            "(2) sau đó nêu rõ điểm giống và khác nếu ngữ cảnh cho phép; "
            "(3) không chỉ liệt kê điều khoản mà thiếu phân tích đối chiếu.\n"
        )
    elif query_asks_structured_registration_conditions(query):
        base += (
            "\n\n━━ ĐIỀU KIỆN / YÊU CẦU — TRÍCH NGUYÊN VĂN TRƯỚC, TÓM NHÓM SAU ━━\n"
            "Câu hỏi về điều kiện đăng ký, thành lập, cấp phép hoặc tổ chức hoạt động. "
            "TUYỆT ĐỐI KHÔNG chỉ trích một câu kiểu \"đáp ứng điều kiện về cơ sở vật chất và nhân lực theo quy định\" "
            "rồi kết thúc.\n"
            "**Bước 1 — Trích nguyên văn:** Với Điều luật trong NGỮ CẢNH trực tiếp quy định điều kiện/yêu cầu "
            "(ví dụ Điều 18 Luật Thư viện…), phải chép **đầy đủ mọi Khoản và Điểm** có trong NGỮ CẢNH, "
            "giữ nguyên đánh số; **KHÔNG** thay thế bằng bullet tóm tắt trước bước 2.\n"
            "**Bước 2 — Tóm tắt theo nhóm** (sau bước 1), tối thiểu:\n"
            "1) **Cơ sở vật chất / trang thiết bị / địa điểm** — theo từng khoản/điểm đã trích.\n"
            "2) **Hoạt động / phạm vi / nội dung hoạt động**.\n"
            "3) **Nhân lực** — số lượng, trình độ, chứng chỉ, chức danh…\n"
            "Nếu NGỮ CẢNH không có chi tiết cho một nhóm → ghi rõ phần đó không có trong đoạn trích; KHÔNG bịa.\n"
            "Với DẠNG A: thứ tự = **Trích nguyên văn Điều…** → **Tóm tắt điều kiện theo nhóm**. "
            "Với DẠNG B: trong **## 2. CĂN CỨ PHÁP LÝ** có **Trích nguyên văn** rồi **Điều kiện cụ thể** (3 nhóm).\n"
        )
    elif query_asks_comprehensive_statutory_coverage(query):
        base += (
            "\n\n━━ QUÉT ĐỦ ĐIỀU TRONG NGỮ CẢNH (CÙNG CHỦ ĐỀ) ━━\n"
            "Câu hỏi về chính sách nhà nước, tiêu chí phân loại hoặc quy định chung theo một luật/lĩnh vực.\n"
            "- PHẢI rà và trích **mọi Điều/Khoản trong NGỮ CẢNH** thuộc **cùng văn bản** trả lời trực tiếp câu hỏi; "
            "không chỉ một Điều đầu tiên.\n"
            "- Với **Luật Người khuyết tật** và câu về **chính sách**: nếu NGỮ CẢNH có điều khoản về chính sách/quyền "
            "(thường gồm Điều 18 hoặc tương đương), **bắt buộc** đưa vào câu trả lời kèm trích nguyên văn các khoản có trong ngữ cảnh.\n"
            "- Với **Luật Đầu tư công** / **dự án trọng điểm quốc gia**: ưu tiên Điều quy định **tiêu chí phân loại**; "
            "nếu NGỮ CẢNH chỉ có Điều về **điều chỉnh** tiêu chí mà câu hỏi hỏi **tiêu chí phân loại**, phải nêu rõ "
            "và trích thêm mọi Điều khác trong ngữ cảnh có nội dung tiêu chí (ví dụ các Điều về phân nhóm dự án).\n"
            "- **Không** lấy điều khoản từ văn bản **lệch chủ đề** làm phần chính khi ngữ cảnh đã có văn bản đúng lĩnh vực.\n"
        )
    elif query_expects_llm_synthesis_from_context(query):
        base += (
            "\n\n━━ TRẢ LỜI TRỰC TIẾP CÂU HỎI ━━\n"
            "Câu hỏi yêu cầu thông tin CỤ THỂ (mức phạt/số tiền, điều kiện, yêu cầu pháp lý, "
            "thủ tục, văn bản hoặc điều khoản căn cứ…). "
            "Phần **Câu trả lời:** phải trả lời đúng trọng tâm ngay từ đầu, trích từ NGỮ CẢNH "
            "(số điều, khung tiền, nội dung chính); "
            "KHÔNG chỉ nói hệ thống đã truy xuất văn bản hoặc chỉ liệt kê tên văn bản mà không trả lời.\n"
        )
    base += (
        "\n\n━━ KHÔNG ĐƯỢC ━━\n"
        "Không mở bài bằng câu 'Hệ thống đã truy xuất nhiều đoạn…' rồi chỉ liệt kê tên văn bản. "
        "Bắt buộc trích và/hoặc hệ thống hóa nội dung các Điều/Khoản trong NGỮ CẢNH để trả lời đúng câu hỏi.\n"
    )
    return base


def _ensure_response_format(answer: str) -> str:
    """Normalize answer to required format header."""
    if not answer or answer.strip() == NO_INFO_MESSAGE:
        return answer
    normalized = answer.strip()
    # Gộp tiêu đề "Câu trả lời:" lặp liên tiếp ở đầu bài
    normalized = re.sub(
        r"(?is)^(?:\s*câu trả lời\s*:\s*\n*){2,}",
        "Câu trả lời:\n\n",
        normalized,
    )
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
        article_number = normalize_article_number_canonical(p.get("article_number"))
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
        article_number = normalize_article_number_canonical(p.get("article_number"))
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
