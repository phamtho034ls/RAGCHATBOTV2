"""RAG pipeline with strict grounding and post-generation validation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.config import (
    CHECKLIST_PROMPT_TEMPLATE,
    CHECKLIST_SYSTEM_PROMPT,
    CONTEXT_RELEVANCE_THRESHOLD,
    COPILOT_SYSTEM_PROMPT,
    FALLBACK_REASONING_PROMPT,
    CAN_CU_PHAP_LY_PROMPT,
    GIAI_THICH_QUY_DINH_PROMPT,
    LEGAL_DOCUMENT_FORMAT_PROMPT,
    NO_INFO_MESSAGE,
    QUERY_REWRITE_PROMPT,
    RAG_PROMPT_TEMPLATE,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    SYSTEM_PROMPT,
    TOP_K,
)
from app.services.answer_validator import validate_answer_grounding
from app.services.llm_client import generate
from app.services.query_understanding import analyze_query
from app.services.retrieval import search_all, search_with_fallback
from app.services.retrieval import search_by_metadata as _search_metadata

log = logging.getLogger(__name__)


# ── Context validation & answer extraction helpers ─────────────────


_DOC_NUMBER_RE = re.compile(r"\b(\d+[/_]\d{4}[/_][A-ZĐa-zđ\-]+)\b")


def _extract_doc_numbers_from_text(text: str) -> set:
    return {m.group(1).replace("_", "/") for m in _DOC_NUMBER_RE.finditer(text or "")}


def _collect_context_doc_numbers(context_docs: List[dict]) -> set:
    nums: set = set()
    for doc in context_docs:
        nums |= _extract_doc_numbers_from_text(doc.get("text", ""))
        meta = doc.get("metadata", {}) or {}
        nums |= _extract_doc_numbers_from_text(meta.get("law_name", ""))
        raw_doc_num = (meta.get("doc_number") or "").strip()
        if raw_doc_num:
            nums.add(raw_doc_num)
        nums |= _extract_doc_numbers_from_text(raw_doc_num)
    return nums


def _normalize_for_comparison(doc_num: str) -> str:
    """Normalize doc number for comparison: NĐ→ND, QĐ→QD, lowercase."""
    return doc_num.replace("_", "/").replace("Đ", "D").replace("đ", "d").lower()


def _strip_hallucinated_doc_numbers(answer: str, context_doc_numbers: set) -> str:
    if not context_doc_numbers or not answer:
        return answer
    context_normalized = {_normalize_for_comparison(n) for n in context_doc_numbers}
    answer_doc_numbers = _extract_doc_numbers_from_text(answer)
    hallucinated = {n for n in answer_doc_numbers if _normalize_for_comparison(n) not in context_normalized}
    if not hallucinated:
        return answer
    log.warning("[ANTI-HALLUCINATION] Stripping doc numbers not in context: %s", hallucinated)
    cleaned = answer
    for bad_num in hallucinated:
        cleaned = re.sub(rf"[^\n]*{re.escape(bad_num)}[^\n]*\n?", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or NO_INFO_MESSAGE


def strip_hallucinated_references(answer: str, sources: list) -> str:
    """Public API: strip doc numbers from *answer* that don't exist in *sources*.

    Accepts both RAG context_docs format (text+metadata) and query_router
    source format (content+metadata).  Safe to call from any handler.
    """
    doc_nums: set = set()
    for s in sources:
        doc_nums |= _extract_doc_numbers_from_text(s.get("text", ""))
        doc_nums |= _extract_doc_numbers_from_text(s.get("content", ""))
        meta = s.get("metadata", {}) or {}
        doc_nums |= _extract_doc_numbers_from_text(meta.get("law_name", ""))
        raw_doc_num = (meta.get("doc_number") or "").strip()
        if raw_doc_num:
            doc_nums.add(raw_doc_num)
        doc_nums |= _extract_doc_numbers_from_text(raw_doc_num)
    return _strip_hallucinated_doc_numbers(answer, doc_nums)


def _extract_keywords(text: str) -> List[str]:
    """Trích xuất keyword quan trọng từ câu hỏi (tên văn bản, điều luật, từ pháp lý)."""
    keywords = []
    # Tên văn bản pháp luật
    law_patterns = [
        re.compile(r"(luật\s+[^\.,\?]{3,}?)(?:\s*,|\s*\?|\s+điều|\s+quy|\s*$)", re.IGNORECASE),
        re.compile(r"(nghị định\s+[\d/\w\-]+)", re.IGNORECASE),
        re.compile(r"(thông tư\s+[\d/\w\-]+)", re.IGNORECASE),
        re.compile(r"(quyết định\s+[\d/\w\-]+)", re.IGNORECASE),
    ]
    for pat in law_patterns:
        for m in pat.finditer(text):
            keywords.append(m.group(1).strip().lower())
    # Điều luật
    article_matches = re.findall(r"điều\s+(\d+[a-zA-Z]?)", text, re.IGNORECASE)
    for a in article_matches:
        keywords.append(f"điều {a}".lower())
    # Từ khóa pháp lý
    legal_terms = re.findall(
        r"(quy định|xử phạt|chế tài|thẩm quyền|trách nhiệm|nghĩa vụ|quyền hạn|"
        r"cấm|cho phép|điều kiện|thủ tục|hồ sơ|chức năng|nhiệm vụ)",
        text, re.IGNORECASE,
    )
    keywords.extend([t.lower() for t in legal_terms])
    return list(dict.fromkeys(keywords))  # dedupe, keep order


def _chunk_has_legal_reference(text: str) -> bool:
    """Kiểm tra chunk có chứa tham chiếu pháp luật (tên văn bản hoặc điều luật)."""
    patterns = [
        r"điều\s+\d+",
        r"khoản\s+\d+",
        r"luật\s+\w+",
        r"nghị định\s+\d+",
        r"thông tư\s+\d+",
        r"quyết định\s+\d+",
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _context_has_legal_content(context_docs: List[dict]) -> bool:
    """Kiểm tra xem context có chứa nội dung pháp luật không."""
    for doc in context_docs:
        text = doc.get("text", "")
        meta = doc.get("metadata", {}) or {}
        if meta.get("law_name") or meta.get("article_number"):
            return True
        if _chunk_has_legal_reference(text):
            return True
    return False


def _filter_relevant_chunks(
    chunks: List[dict],
    question: str,
    question_keywords: List[str],
) -> List[dict]:
    """Answer extraction: Lọc và giữ top chunks chứa keyword từ câu hỏi.

    Ưu tiên chunks có:
    - Số hiệu văn bản
    - Điều luật
    - Keyword overlap với câu hỏi
    """
    if not question_keywords:
        return chunks

    scored: List[tuple] = []
    for chunk in chunks:
        text_lower = chunk.get("text", "").lower()
        meta = chunk.get("metadata", {}) or {}
        score = 0

        # Keyword overlap
        for kw in question_keywords:
            if kw in text_lower:
                score += 2

        # Có tên văn bản trong metadata
        if meta.get("law_name"):
            score += 1
        # Có số điều trong metadata
        if meta.get("article_number"):
            score += 1
        # Có tham chiếu pháp luật trong text
        if _chunk_has_legal_reference(text_lower):
            score += 1

        # Rerank score nếu có
        rerank_score = chunk.get("rerank_score", chunk.get("hybrid_score", chunk.get("score", 0)))
        scored.append((chunk, score, rerank_score))

    # Sort by keyword overlap score (desc), then rerank score (desc)
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Giữ tất cả chunks có overlap > 0, hoặc tối thiểu top RERANK_TOP_K
    filtered = [c for c, s, _ in scored if s > 0]
    if len(filtered) < RERANK_TOP_K:
        # Bổ sung thêm chunks theo rerank score
        remaining = [c for c, s, _ in scored if s == 0]
        filtered.extend(remaining[: RERANK_TOP_K - len(filtered)])

    return filtered


def _validate_context_relevance(context_docs: List[dict]) -> dict:
    """Validate context trước khi gửi vào LLM.

    Returns dict with:
    - has_context: bool - có chunk không
    - has_legal_content: bool - context có chứa nội dung pháp luật
    - should_answer: bool - hệ thống có nên trả lời từ context hay không
    - chunk_count: int
    - top_score: float
    """
    if not context_docs:
        return {
            "has_context": False,
            "has_legal_content": False,
            "should_answer": False,
            "chunk_count": 0,
            "top_score": 0.0,
        }

    chunk_count = len(context_docs)
    top_score = max(
        doc.get("rerank_score", doc.get("hybrid_score", doc.get("score", 0)))
        for doc in context_docs
    )
    has_legal = _context_has_legal_content(context_docs)

    # Nếu có chunk được trả về → hệ thống PHẢI cố gắng trả lời
    should_answer = chunk_count > 0

    return {
        "has_context": True,
        "has_legal_content": has_legal,
        "should_answer": should_answer,
        "chunk_count": chunk_count,
        "top_score": round(top_score, 4),
    }


def _source_label(doc: dict, idx: int) -> str:
    meta = doc.get("metadata", {}) or {}
    law_name = meta.get("law_name") or meta.get("title") or f"Nguồn {idx}"
    article = meta.get("article_number")
    article_title = meta.get("article_title")
    section = meta.get("section")
    ref_parts = [law_name]
    if article:
        article_ref = f"Điều {article}"
        if article_title:
            article_ref += f". {article_title}"
        ref_parts.append(article_ref)
    elif section:
        ref_parts.append(str(section))
    return ", ".join(ref_parts)


def _extract_related_doc_names(context_docs: List[dict]) -> List[str]:
    """Trích xuất danh sách tên văn bản pháp luật duy nhất từ context docs."""
    names: List[str] = []
    seen: set = set()
    for doc in context_docs:
        meta = doc.get("metadata", {}) or {}
        law_name = meta.get("law_name") or meta.get("title")
        if law_name:
            key = law_name.strip().lower()
            if key not in seen:
                seen.add(key)
                names.append(law_name.strip())
        # Cũng trích xuất từ text nếu metadata không có
        text = doc.get("text", "")
        for pat in [
            re.compile(r"(Luật\s+[^\.,\?\n]{5,60})", re.IGNORECASE),
            re.compile(r"(Nghị định\s+\d+/\d{4}/[A-ZĐa-zđ\-]+[^\.,\?\n]{0,60})", re.IGNORECASE),
            re.compile(r"(Thông tư\s+\d+/\d{4}/[A-ZĐa-zđ\-]+[^\.,\?\n]{0,60})", re.IGNORECASE),
            re.compile(r"(Quyết định\s+\d+/\d{4}/[A-ZĐa-zđ\-]+[^\.,\?\n]{0,60})", re.IGNORECASE),
        ]:
            for m in pat.finditer(text):
                found = m.group(1).strip()
                fkey = found.lower()
                if fkey not in seen:
                    seen.add(fkey)
                    names.append(found)
    return names


def _build_related_docs_response(context_docs: List[dict]) -> str:
    """Trường hợp 2: Có văn bản liên quan nhưng không có nội dung chi tiết.

    Xây dựng câu trả lời liệt kê các văn bản pháp luật liên quan."""
    doc_names = _extract_related_doc_names(context_docs)
    if not doc_names:
        return NO_INFO_MESSAGE

    lines = [
        "Trong các tài liệu hiện có chưa tìm thấy nội dung chi tiết trả lời trực tiếp câu hỏi.",
        "",
        "Tuy nhiên, hệ thống đã tìm thấy các văn bản pháp luật liên quan:",
        "",
        "Danh sách văn bản liên quan:",
    ]
    for name in doc_names:
        lines.append(f"- {name}")
    lines.append("")
    lines.append("Các văn bản trên có thể chứa quy định liên quan đến vấn đề bạn đang hỏi.")
    return "\n".join(lines)


def _is_no_info_answer(answer: str) -> bool:
    """Kiểm tra xem LLM có trả lời dạng 'Không tìm thấy thông tin' không."""
    if not answer:
        return True
    lower = answer.strip().lower()
    no_info_patterns = [
        "không tìm thấy thông tin",
        "không có thông tin",
        "không tìm thấy nội dung",
        "không có nội dung",
        "không chứa thông tin",
        "không có dữ liệu",
    ]
    for pat in no_info_patterns:
        if pat in lower:
            return True
    return False


def _build_prompt(question: str, context_docs: List[dict]) -> str:
    blocks = []
    for i, doc in enumerate(context_docs, start=1):
        blocks.append(f"[{i}] {_source_label(doc, i)}\n{doc['text']}")
    context = "\n\n---\n\n".join(blocks)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def _format_sources(docs: List[dict]) -> List[dict]:
    return [
        {
            "content": doc["text"],
            "score": round(doc.get("rerank_score", doc.get("hybrid_score", doc.get("score", 0))), 4),
            "dataset_id": doc.get("dataset_id", ""),
            "metadata": doc.get("metadata", {}),
        }
        for doc in docs
    ]


def _append_citations(answer: str, docs: List[dict]) -> str:
    if not docs:
        return answer
    citations = []
    for i, doc in enumerate(docs[:5], start=1):
        citations.append(f"- [{i}] {_source_label(doc, i)}")
    # Use structured format: Answer section + Source section
    if "Nguồn" not in answer and "nguồn" not in answer.lower():
        return f"{answer}\n\nNguồn:\n" + "\n".join(citations)
    return answer


async def rewrite_query(question: str) -> str:
    try:
        prompt = QUERY_REWRITE_PROMPT.format(question=question)
        rewritten = (await generate(prompt, temperature=0.0)).strip()
        if rewritten and len(rewritten) > 5:
            return rewritten
    except Exception:
        pass
    return question


async def expand_queries(question: str, rewritten_query: str,
                         analysis: Optional[Dict[str, Any]] = None) -> List[str]:
    """Sinh thêm truy vấn phụ để tăng recall retrieval."""
    intent = analysis.get("intent", "") if analysis else ""

    if intent == "checklist_documents":
        # For checklist, generate targeted sub-queries from analysis
        sub_queries = _generate_checklist_sub_queries(question, analysis)
        if sub_queries:
            return sub_queries

    prompt = f"""Tạo 3 truy vấn tìm kiếm tương đương cho câu hỏi pháp lý sau.
Chỉ trả về JSON array gồm các string, không giải thích.

Câu hỏi gốc: {question}
Câu hỏi đã rewrite: {rewritten_query}
"""
    try:
        raw = await generate(prompt, temperature=0.0)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            expanded = [str(x).strip() for x in parsed if str(x).strip()]
            return expanded[:3]
    except Exception:
        pass
    return []


def _generate_checklist_sub_queries(question: str,
                                     analysis: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate multiple targeted sub-queries from checklist analysis."""
    if not analysis:
        return []

    filters = analysis.get("filters", {})
    keywords = analysis.get("keywords", [])

    sub_queries = []

    # Sub-query by field (domain)
    field_map = {
        "van_hoa_the_thao": "văn bản quy định quản lý nhà nước về văn hóa thể thao",
        "giao_duc": "văn bản quy định quản lý nhà nước về giáo dục đào tạo",
        "y_te": "văn bản quy định quản lý nhà nước về y tế sức khỏe",
        "dat_dai": "văn bản quy định quản lý đất đai nhà ở quy hoạch",
        "tai_chinh": "văn bản quy định quản lý tài chính ngân sách",
        "lao_dong": "văn bản quy định quản lý lao động việc làm bảo hiểm",
        "hanh_chinh": "văn bản quy định quản lý hành chính công chức viên chức",
        "moi_truong": "văn bản quy định quản lý môi trường tài nguyên",
        "an_ninh_quoc_phong": "văn bản quy định an ninh quốc phòng quân sự",
        "nong_nghiep": "văn bản quy định quản lý nông nghiệp nông thôn",
    }
    field = filters.get("field")
    if field and field in field_map:
        sub_queries.append(field_map[field])

    # Sub-query by government level
    gov_level = filters.get("government_level")
    if gov_level:
        level_map = {
            "xa": "nhiệm vụ quyền hạn UBND cấp xã phường thị trấn",
            "huyen": "nhiệm vụ quyền hạn UBND cấp huyện quận",
            "tinh": "nhiệm vụ quyền hạn UBND cấp tỉnh thành phố",
            "trung_uong": "nhiệm vụ quyền hạn chính quyền trung ương",
        }
        if gov_level in level_map:
            sub_queries.append(level_map[gov_level])

    # Sub-query by position
    position = filters.get("position")
    if position:
        pos_map = {
            "pho_chu_tich_ubnd": "chức năng nhiệm vụ quyền hạn phó chủ tịch UBND",
            "chu_tich_ubnd": "chức năng nhiệm vụ quyền hạn chủ tịch UBND",
            "bi_thu": "chức năng nhiệm vụ bí thư đảng ủy",
            "truong_cong_an": "chức năng nhiệm vụ trưởng công an xã",
        }
        if position in pos_map:
            sub_queries.append(pos_map[position])

    # Combine keywords as a sub-query
    if keywords:
        kw_query = " ".join(keywords[:6])
        sub_queries.append(kw_query)

    return sub_queries[:5]


def _dedupe_docs(docs: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for doc in docs:
        key = (doc.get("dataset_id"), doc.get("id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


async def retrieve_context(
    question: str,
    top_k: int,
    filters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    analysis = analyze_query(question)
    merged_filters = {**analysis.get("filters", {}), **(filters or {})}
    keywords = analysis.get("keywords", [])

    # Rewrite + expand song song để giảm latency
    rewritten, expanded = await asyncio.gather(
        rewrite_query(question),
        expand_queries(question, question, analysis=analysis),
    )
    # Expand lại với rewritten query nếu khác question gốc
    if rewritten != question and not expanded:
        expanded = await expand_queries(question, rewritten, analysis=analysis)

    variants = [rewritten, *expanded]

    # === Sử dụng RETRIEVAL_TOP_K (20) để lấy nhiều candidates hơn ===
    retrieval_k = max(top_k, RETRIEVAL_TOP_K)

    # Use fallback search for checklist intent or when keywords are available
    intent = analysis.get("intent", "")
    if intent == "checklist_documents" or keywords:
        results = await search_with_fallback(
            question,
            keywords=keywords,
            filters=merged_filters or None,
            top_k=retrieval_k,
            query_variants=variants,
            min_results=3,
        )
    else:
        results = await search_all(
            question,
            filters=merged_filters or None,
            top_k=retrieval_k,
            query_variants=variants,
        )

    # If not enough results, try metadata search in PostgreSQL as additional source
    if len(results) < 3 and keywords:
        log.info("[RETRIEVAL] Supplementing with metadata search (%d results so far)", len(results))
        field_filter = merged_filters.get("field", "")
        # Map field to Vietnamese for search
        field_vi = {
            "van_hoa_the_thao": "văn hóa",
            "giao_duc": "giáo dục",
            "y_te": "y tế",
            "dat_dai": "đất đai",
        }.get(field_filter, field_filter)

        db_results = await _search_metadata(
            linh_vuc=field_vi or None,
            keywords=keywords,
            only_effective=True,
            limit=retrieval_k,
        )
        existing_keys = {(r.get("dataset_id", ""), r.get("id", "")) for r in results}
        for row in db_results:
            key = (row.get("dataset_id", ""), row.get("id"))
            if key not in existing_keys:
                results.append({
                    "id": row.get("id"),
                    "text": row.get("chunk_text", ""),
                    "metadata": {
                        "law_name": row.get("law_name"),
                        "document_type": row.get("document_type"),
                        "doc_number": row.get("doc_number"),
                        "source_file": row.get("source_file"),
                    },
                    "score": 0.0,
                    "dataset_id": row.get("dataset_id", ""),
                    "retrieval": "metadata_db",
                })
                existing_keys.add(key)

    results = _dedupe_docs(results)

    # === Answer Extraction: Lọc chunks liên quan trước khi gửi LLM ===
    question_keywords = _extract_keywords(question)
    if results and question_keywords:
        results = _filter_relevant_chunks(results, question, question_keywords)

    # === Giới hạn top RERANK_TOP_K (5) chunks gửi vào LLM ===
    results = results[:RERANK_TOP_K]

    # === DEBUG LOG ===
    log.info("[DEBUG] Query: %s", question[:120])
    log.info("[DEBUG] Rewritten query: %s", rewritten[:120] if rewritten else "N/A")
    log.info("[DEBUG] Retrieved chunks: %d", len(results))
    if results:
        top_doc = results[0]
        top_meta = top_doc.get("metadata", {}) or {}
        top_source = top_meta.get("law_name", "Unknown")
        top_article = top_meta.get("article_number", "")
        top_score = top_doc.get("rerank_score", top_doc.get("hybrid_score", top_doc.get("score", 0)))
        log.info("[DEBUG] Top source: %s Điều %s (score=%.4f)", top_source, top_article, top_score)
    log.info("[DEBUG] Sending %d chunks to LLM", len(results))
    log.info("[DEBUG] Keywords extracted: %s", question_keywords[:8] if question_keywords else [])
    log.info("[RETRIEVAL] intent=%s, retrieved_docs=%d, keywords=%s",
             intent, len(results), keywords[:5] if keywords else [])

    return {
        "results": results,
        "analysis": analysis,
        "rewritten_query": rewritten,
        "expanded_queries": expanded,
    }


async def _generate_grounded_answer(question: str, context_docs: List[dict], temperature: float,
                                    intent: str = "") -> str:
    # === CONTEXT VALIDATION ===
    ctx_validation = _validate_context_relevance(context_docs)
    has_legal = ctx_validation["has_legal_content"]
    log.info("[DEBUG] Context validation: chunks=%d, has_legal=%s, should_answer=%s, top_score=%.4f",
             ctx_validation["chunk_count"], has_legal,
             ctx_validation["should_answer"], ctx_validation["top_score"])

    # ━━━ TRƯỜNG HỢP 3: Không có chunk nào → fallback ━━━
    if not context_docs or not ctx_validation["has_context"]:
        log.info("[DEBUG] No context docs → TRƯỜNG HỢP 3 → fallback reasoning")
        return await _fallback_reasoning(question, intent)

    # Checklist intent uses a different prompt and less strict validation
    if intent == "checklist_documents":
        return await _generate_checklist_answer(question, context_docs, temperature)

    # Document metadata intent → query SQLite documents table directly
    if intent == "document_metadata":
        return await _generate_metadata_answer(question, context_docs, temperature)

    # Document relation intent
    if intent == "document_relation":
        return await _generate_relation_answer(question, context_docs, temperature)

    # Program goal intent
    if intent == "program_goal":
        return await _generate_program_goal_answer(question, context_docs, temperature)

    # Căn cứ pháp lý intent
    if intent == "can_cu_phap_ly":
        return await _generate_can_cu_phap_ly_answer(question, context_docs, temperature)

    # Giải thích quy định intent
    if intent == "giai_thich_quy_dinh":
        return await _generate_giai_thich_answer(question, context_docs, temperature)

    # === Gửi vào LLM với prompt 3-tier ===
    prompt = _build_prompt(question, context_docs)
    answer = await generate(prompt, system=SYSTEM_PROMPT, temperature=temperature)
    answer_stripped = answer.strip()

    # === Phát hiện LLM trả "Không tìm thấy" sai ===
    if _is_no_info_answer(answer_stripped) and has_legal:
        log.warning("[SAFETY] LLM returned NO_INFO but context has legal content. Attempting re-generation.")
        # Lần 2: Re-generate với prompt mạnh hơn
        force_prompt = (
            f"{prompt}\n\n"
            "LƯU Ý QUAN TRỌNG: NGỮ CẢNH ở trên CÓ CHỨA thông tin pháp luật liên quan.\n"
            "Bạn PHẢI xử lý theo một trong hai cách:\n"
            "1. Nếu có nội dung chi tiết → trích xuất và trả lời (TRƯỜNG HỢP 1).\n"
            "2. Nếu chỉ có tên văn bản → liệt kê danh sách văn bản liên quan (TRƯỜNG HỢP 2).\n"
            "TUYỆT ĐỐI KHÔNG ĐƯỢC trả lời 'Không tìm thấy thông tin'."
        )
        answer = await generate(force_prompt, system=SYSTEM_PROMPT, temperature=temperature)
        answer_stripped = answer.strip()

    # === Nếu LLM vẫn trả NO_INFO sau 2 lần → xây dựng TRƯỜNG HỢP 2 thủ công ===
    if _is_no_info_answer(answer_stripped) and has_legal:
        log.warning("[SAFETY] LLM still returned NO_INFO after retry. Building related docs response (Case 2).")
        return _build_related_docs_response(context_docs)

    # === Nếu LLM trả NO_INFO và context không có legal content → TRƯỜNG HỢP 3 ===
    if _is_no_info_answer(answer_stripped) and not has_legal:
        log.info("[DEBUG] No legal content + LLM NO_INFO → TRƯỜNG HỢP 3")
        return await _fallback_reasoning(question, intent, context_docs)

    # === Anti-hallucination: strip doc numbers not in context ===
    ctx_nums = _collect_context_doc_numbers(context_docs)
    answer_stripped = _strip_hallucinated_doc_numbers(answer_stripped, ctx_nums)

    # === Validation grounding ===
    if answer_stripped and answer_stripped != NO_INFO_MESSAGE:
        validation = validate_answer_grounding(answer_stripped, context_docs)
        if not validation["is_grounded"] and not has_legal:
            log.warning("Answer not grounded (sim=%.2f) and no legal content, trying fallback",
                        validation["similarity"])
            return await _fallback_reasoning(question, intent, context_docs)
        elif not validation["is_grounded"] and has_legal:
            log.warning("Answer grounding low (sim=%.2f) but context has legal content, keeping answer",
                        validation["similarity"])
        log.info("[DEBUG] Final answer: case=1, length=%d, grounding_sim=%.4f",
                 len(answer_stripped), validation["similarity"])
        return _append_citations(answer_stripped, context_docs)

    # Nếu answer rỗng
    if has_legal:
        return _build_related_docs_response(context_docs)
    return await _fallback_reasoning(question, intent, context_docs)


async def _generate_metadata_answer(
    question: str,
    context_docs: List[dict],
    temperature: float,
) -> str:
    """Answer DOCUMENT_METADATA queries using PostgreSQL documents table."""
    import re as _re
    from sqlalchemy import select, or_
    from app.database.session import _session_factory
    from app.database.models import Document

    doc_info = None
    # Extract doc number patterns from question
    doc_num_patterns = [
        _re.compile(r"(\d+/\d{4}/[A-ZĐa-zđ\-]+)", _re.IGNORECASE),
        _re.compile(r"(\d+/[A-ZĐa-zđ\-]+-[A-ZĐa-zđ]+)", _re.IGNORECASE),
        _re.compile(r"(?:quyết định|nghị định|thông tư|luật)\s+(?:số\s+)?(\d+)", _re.IGNORECASE),
    ]
    async with _session_factory() as session:
        for pat in doc_num_patterns:
            m = pat.search(question)
            if m:
                term = m.group(1)
                stmt = select(Document).where(
                    or_(
                        Document.doc_number.ilike(f"%{term}%"),
                        Document.title.ilike(f"%{term}%"),
                    )
                ).limit(1)
                result = await session.execute(stmt)
                row = result.scalar_one_or_none()
                if row:
                    doc_info = row
                    break
        if doc_info is None:
            keywords = _re.findall(r"[\w]{3,}", question)
            for kw in keywords:
                stmt = select(Document).where(
                    or_(
                        Document.title.ilike(f"%{kw}%"),
                        Document.doc_number.ilike(f"%{kw}%"),
                    )
                ).limit(1)
                result = await session.execute(stmt)
                row = result.scalar_one_or_none()
                if row:
                    doc_info = row
                    break

    if doc_info:
        parts = ["**Thông tin văn bản:**\n"]
        if doc_info.title:
            parts.append(f"- **Tên văn bản:** {doc_info.title}")
        if doc_info.doc_number:
            parts.append(f"- **Số hiệu:** {doc_info.doc_number}")
        if doc_info.issuer:
            parts.append(f"- **Cơ quan ban hành:** {doc_info.issuer}")
        if doc_info.issued_date:
            parts.append(f"- **Ngày ban hành:** {doc_info.issued_date}")
        if doc_info.document_type:
            parts.append(f"- **Loại văn bản:** {doc_info.document_type}")
        if doc_info.file_path:
            from pathlib import Path as _Path
            parts.append(f"- **Tên file:** {_Path(doc_info.file_path).name}")

        metadata_text = "\n".join(parts)

        if context_docs:
            prompt = f"""Dựa trên thông tin metadata sau và ngữ cảnh, hãy trả lời câu hỏi.

THÔNG TIN VĂN BẢN:
{metadata_text}

NGỮ CẢNH BỔ SUNG:
{_build_prompt(question, context_docs[:3])}
"""
            answer = await generate(prompt, system=SYSTEM_PROMPT, temperature=temperature)
            if answer.strip():
                answer = strip_hallucinated_references(answer, context_docs)
                return f"{answer.strip()}\n\n{metadata_text}\n\n" + _format_citation_footer(context_docs)
        return metadata_text

    # Fallback to regular RAG if no metadata found
    if context_docs:
        prompt = _build_prompt(question, context_docs)
        answer = await generate(prompt, system=SYSTEM_PROMPT, temperature=temperature)
        answer = strip_hallucinated_references(answer, context_docs)
        return _append_citations(answer, context_docs)
    return NO_INFO_MESSAGE


async def _generate_relation_answer(
    question: str,
    context_docs: List[dict],
    temperature: float,
) -> str:
    """Answer DOCUMENT_RELATION queries about which documents modify/replace others."""
    if not context_docs:
        return await _fallback_reasoning(question, "document_relation")

    blocks = []
    for i, doc in enumerate(context_docs, start=1):
        blocks.append(f"[{i}] {_source_label(doc, i)}\n{doc['text']}")
    context = "\n\n---\n\n".join(blocks)

    prompt = f"""NGỮ CẢNH:
{context}

CÂU HỎI: {question}

HƯỚNG DẪN:
1. Xác định các văn bản pháp luật được sửa đổi, bổ sung hoặc thay thế.
2. Liệt kê rõ ràng từng văn bản bị ảnh hưởng.
3. Trích dẫn Điều, Khoản cụ thể quy định về sửa đổi.
4. Chỉ sử dụng thông tin trong ngữ cảnh. Không bịa đặt.

ĐỊNH DẠNG:
Câu trả lời:
<nội dung>

Nguồn:
<Tên văn bản> – Điều X
"""
    answer = await generate(prompt, system=SYSTEM_PROMPT, temperature=temperature)
    answer = strip_hallucinated_references(answer, context_docs)
    return _append_citations(answer, context_docs)


async def _generate_program_goal_answer(
    question: str,
    context_docs: List[dict],
    temperature: float,
) -> str:
    """Answer PROGRAM_GOAL queries about objectives of programs/plans."""
    if not context_docs:
        return await _fallback_reasoning(question, "program_goal")

    blocks = []
    for i, doc in enumerate(context_docs, start=1):
        blocks.append(f"[{i}] {_source_label(doc, i)}\n{doc['text']}")
    context = "\n\n---\n\n".join(blocks)

    prompt = f"""NGỮ CẢNH:
{context}

CÂU HỎI: {question}

HƯỚNG DẪN:
1. Xác định mục tiêu, nhiệm vụ chính của chương trình/kế hoạch/đề án.
2. Liệt kê các mục tiêu cụ thể nếu có.
3. Trích dẫn Điều, Khoản cụ thể.
4. Chỉ sử dụng thông tin trong ngữ cảnh. Không bịa đặt.

ĐỊNH DẠNG:
Câu trả lời:
<nội dung>

Nguồn:
<Tên văn bản> – Điều X
"""
    answer = await generate(prompt, system=SYSTEM_PROMPT, temperature=temperature)
    answer = strip_hallucinated_references(answer, context_docs)
    return _append_citations(answer, context_docs)


def _format_citation_footer(docs: List[dict]) -> str:
    """Format citation footer for the response."""
    if not docs:
        return ""
    citations = []
    for i, doc in enumerate(docs[:5], start=1):
        citations.append(f"- [{i}] {_source_label(doc, i)}")
    return "Nguồn:\n" + "\n".join(citations)


async def _fallback_reasoning(
    question: str,
    intent: str = "",
    context_docs: Optional[List[dict]] = None,
) -> str:
    """Fallback: khi retrieval không tìm đủ tài liệu, suy luận văn bản liên quan.

    Thay vì trả lời 'Không tìm thấy', mở rộng tìm kiếm qua metadata DB
    và suy luận các văn bản pháp luật có thể liên quan.
    """
    log.info("[FALLBACK] Starting fallback reasoning for: %s", question[:80])

    analysis = analyze_query(question)
    filters = analysis.get("filters", {})
    keywords = analysis.get("keywords", [])

    # Step 1: Try metadata-based search in PostgreSQL
    linh_vuc = filters.get("field", "")
    doc_type = filters.get("document_type", "")

    # Map field values to Vietnamese text for search
    field_vi_map = {
        "van_hoa_the_thao": "văn hóa thể thao",
        "giao_duc": "giáo dục đào tạo",
        "y_te": "y tế",
        "dat_dai": "đất đai",
        "tai_chinh": "tài chính",
        "lao_dong": "lao động",
        "hanh_chinh": "hành chính",
        "moi_truong": "môi trường",
        "an_ninh_quoc_phong": "an ninh quốc phòng",
        "nong_nghiep": "nông nghiệp",
    }
    linh_vuc_text = field_vi_map.get(linh_vuc, linh_vuc)

    db_results = await _search_metadata(
        linh_vuc=linh_vuc_text or None,
        keywords=keywords or None,
        only_effective=True,
        limit=10,
    )

    if db_results:
        log.info("[FALLBACK] Found %d results via metadata search", len(db_results))
        # Convert DB results to context docs format
        fallback_docs = []
        for row in db_results:
            fallback_docs.append({
                "text": row.get("chunk_text", ""),
                "metadata": {
                    "law_name": row.get("law_name"),
                    "article_number": row.get("article_number"),
                    "article_title": row.get("article_title"),
                    "document_type": row.get("document_type"),
                    "year": row.get("year") or row.get("nam_ban_hanh"),
                    "source_file": row.get("document_name"),
                    "linh_vuc": row.get("linh_vuc"),
                },
                "score": row.get("match_score", 0),
                "dataset_id": row.get("dataset_id", ""),
            })

        prompt = _build_prompt(question, fallback_docs)
        system = SYSTEM_PROMPT + "\n\n" + LEGAL_DOCUMENT_FORMAT_PROMPT
        answer = await generate(prompt, system=system, temperature=0.3)
        answer = answer.strip()
        if answer and answer != NO_INFO_MESSAGE:
            answer = strip_hallucinated_references(answer, fallback_docs)
            return _append_citations(answer, fallback_docs)
    else:
        fallback_docs = []

    # Step 2: LLM-based reasoning as final fallback
    log.info("[FALLBACK] Using LLM reasoning fallback")
    prompt = FALLBACK_REASONING_PROMPT.format(
        question=question,
        linh_vuc=linh_vuc_text or "không xác định",
        loai_van_ban=doc_type or "không xác định",
        keywords=", ".join(keywords) if keywords else "không xác định",
    )
    answer = await generate(prompt, temperature=0.3)
    answer = answer.strip()

    all_docs = (context_docs or []) + fallback_docs
    if all_docs:
        answer = strip_hallucinated_references(answer, all_docs)
        answer = _append_citations(answer, context_docs or fallback_docs)

    return answer


async def _generate_can_cu_phap_ly_answer(
    question: str,
    context_docs: List[dict],
    temperature: float,
    document_info: str = "",
) -> str:
    """Generate answer for CAN_CU_PHAP_LY intent."""
    blocks = []
    for i, doc in enumerate(context_docs, start=1):
        blocks.append(f"[{i}] {_source_label(doc, i)}\n{doc['text']}")
    context = "\n\n---\n\n".join(blocks)

    prompt = CAN_CU_PHAP_LY_PROMPT.format(
        context=context,
        document_info=document_info or "Không có thông tin cụ thể",
        question=question,
    )
    system = SYSTEM_PROMPT + "\n\n" + LEGAL_DOCUMENT_FORMAT_PROMPT
    answer = await generate(prompt, system=system, temperature=temperature)
    answer = answer.strip()
    if not answer:
        return await _fallback_reasoning(question, "can_cu_phap_ly", context_docs)
    answer = strip_hallucinated_references(answer, context_docs)
    return _append_citations(answer, context_docs)


async def _generate_giai_thich_answer(
    question: str,
    context_docs: List[dict],
    temperature: float,
) -> str:
    """Generate answer for GIAI_THICH_QUY_DINH intent."""
    blocks = []
    for i, doc in enumerate(context_docs, start=1):
        blocks.append(f"[{i}] {_source_label(doc, i)}\n{doc['text']}")
    context = "\n\n---\n\n".join(blocks)

    prompt = GIAI_THICH_QUY_DINH_PROMPT.format(
        context=context,
        question=question,
    )
    answer = await generate(prompt, system=SYSTEM_PROMPT, temperature=temperature)
    answer = answer.strip()
    if not answer:
        return await _fallback_reasoning(question, "giai_thich_quy_dinh", context_docs)
    answer = strip_hallucinated_references(answer, context_docs)
    return _append_citations(answer, context_docs)


def _build_checklist_prompt(question: str, context_docs: List[dict],
                            analysis: Optional[Dict[str, Any]] = None) -> str:
    """Build prompt for checklist document listing."""
    blocks = []
    for i, doc in enumerate(context_docs, start=1):
        blocks.append(f"[{i}] {_source_label(doc, i)}\n{doc['text']}")
    context = "\n\n---\n\n".join(blocks)

    filters = analysis.get("filters", {}) if analysis else {}
    field = filters.get("field", "không xác định")
    position = filters.get("position", "không xác định")
    gov_level = filters.get("government_level", "không xác định")
    keywords = ", ".join(analysis.get("keywords", [])) if analysis else ""

    return CHECKLIST_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        field=field,
        position=position,
        government_level=gov_level,
        keywords=keywords,
    )


async def _generate_checklist_answer(question: str, context_docs: List[dict],
                                      temperature: float,
                                      analysis: Optional[Dict[str, Any]] = None) -> str:
    """Generate a checklist-style answer listing all relevant legal documents."""
    prompt = _build_checklist_prompt(question, context_docs, analysis)
    answer = await generate(prompt, system=CHECKLIST_SYSTEM_PROMPT, temperature=temperature)

    # Relaxed validation: checklist answers aggregate info, so lower threshold
    validation = validate_answer_grounding(answer, context_docs, threshold=0.15)
    if not validation["is_grounded"]:
        log.warning(
            "Checklist answer grounding low (%.2f) but returning anyway",
            validation["similarity"],
        )

    answer = answer.strip()
    if not answer:
        return NO_INFO_MESSAGE
    answer = strip_hallucinated_references(answer, context_docs)
    return _append_citations(answer, context_docs)


async def rag_query_stream(
    question: str,
    temperature: float = 0.3,
    top_k: int = TOP_K,
    filters: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    payload = await retrieve_context(question, top_k=top_k, filters=filters)
    results = payload["results"]
    intent = payload["analysis"].get("intent", "")

    sources = _format_sources(results)
    yield json.dumps(
        {
            "type": "sources",
            "data": sources,
            "query_analysis": payload["analysis"],
            "rewritten_query": payload["rewritten_query"],
            "expanded_queries": payload["expanded_queries"],
        },
        ensure_ascii=False,
    ) + "\n"

    answer = await _generate_grounded_answer(question, results, temperature=temperature, intent=intent)
    for i in range(0, len(answer), 60):
        yield answer[i : i + 60]


async def rag_query(
    question: str,
    temperature: float = 0.3,
    top_k: int = TOP_K,
    filters: Optional[Dict[str, Any]] = None,
) -> dict:
    payload = await retrieve_context(question, top_k=top_k, filters=filters)
    results = payload["results"]
    intent = payload["analysis"].get("intent", "")

    answer = await _generate_grounded_answer(question, results, temperature=temperature, intent=intent)
    return {
        "answer": answer,
        "sources": _format_sources(results),
        "query_analysis": payload["analysis"],
        "rewritten_query": payload["rewritten_query"],
        "expanded_queries": payload["expanded_queries"],
    }


async def rag_query_enhanced(
    question: str,
    temperature: float = 0.3,
    top_k: int = 8,
    filters: Optional[Dict[str, Any]] = None,
) -> dict:
    # Use more results for checklist queries
    analysis = analyze_query(question)
    intent = analysis.get("intent", "")
    if intent == "checklist_documents":
        top_k = max(top_k, 15)

    result = await rag_query(question, temperature=temperature, top_k=top_k, filters=filters)

    log.info("[RAG_ENHANCED] intent=%s, sources=%d, has_answer=%s",
             intent, len(result.get("sources", [])),
             bool(result["answer"] and result["answer"] != NO_INFO_MESSAGE))

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "query_analysis": result["query_analysis"],
        "rewritten_query": result["rewritten_query"],
        "expanded_queries": result["expanded_queries"],
        "system": COPILOT_SYSTEM_PROMPT,
    }
