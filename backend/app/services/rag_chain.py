"""Tiện ích RAG dùng chung: chống ảo giác trích dẫn + fallback suy luận.

Luồng chat production (retrieve + trả lời) nằm trong ``rag_chain_v2`` / ``rag_unified``.
Module này không còn pipeline ``rag_query``/``retrieve_context`` cũ (đã bỏ regex + retrieval trùng).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.config import (
    FALLBACK_REASONING_PROMPT,
    LEGAL_DOCUMENT_FORMAT_PROMPT,
    NO_INFO_MESSAGE,
    RAG_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from app.services.llm_client import generate
from app.services.query_understanding import analyze_query
from app.services.retrieval import search_by_metadata as _search_metadata

log = logging.getLogger(__name__)

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
    """Bỏ số hiệu văn bản trong *answer* không có trong *sources* (format context hoặc Copilot)."""
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


def _build_prompt(question: str, context_docs: List[dict]) -> str:
    blocks = []
    for i, doc in enumerate(context_docs, start=1):
        blocks.append(f"[{i}] {_source_label(doc, i)}\n{doc['text']}")
    context = "\n\n---\n\n".join(blocks)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def _append_citations(answer: str, docs: List[dict]) -> str:
    if not docs:
        return answer
    citations = []
    for i, doc in enumerate(docs[:5], start=1):
        citations.append(f"- [{i}] {_source_label(doc, i)}")
    if "Nguồn" not in answer and "nguồn" not in answer.lower():
        return f"{answer}\n\nNguồn:\n" + "\n".join(citations)
    return answer


async def _fallback_reasoning(
    question: str,
    intent: str = "",
    context_docs: Optional[List[dict]] = None,
) -> str:
    """Khi thiếu ngữ cảnh: metadata DB + LLM fallback (gọi từ rag_chain_v2)."""
    log.info("[FALLBACK] Starting fallback reasoning for: %s", question[:80])

    analysis = analyze_query(question)
    filters = analysis.get("filters", {})
    keywords = analysis.get("keywords", [])

    linh_vuc = filters.get("field", "")
    doc_type = filters.get("document_type", "")

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

    fallback_docs: List[dict] = []
    if db_results:
        log.info("[FALLBACK] Found %d results via metadata search", len(db_results))
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
