"""
Copilot Agent – trung tâm xử lý của Government AI Copilot.

Là brain chính điều phối toàn bộ pipeline:
    User Message → Intent Detection → Context Resolution → Tool/RAG Routing → LLM → Response

Hỗ trợ:
    - Intent detection mở rộng (CAN_CU_PHAP_LY, GIAI_THICH_QUY_DINH)
    - Context memory cho follow-up questions ("văn bản trên", "kế hoạch trên")
    - Fallback reasoning khi không tìm thấy tài liệu
    - Structured logging cho debug
"""

from __future__ import annotations

import logging
import re
from typing import Any, AsyncGenerator, Dict, Optional

from app.services.intent_detector import detect_intent
from app.services.domain_guard import is_in_document_domain, looks_like_follow_up
from app.services.query_router import route_query
from app.memory.conversation_store import conversation_store
from app.config import (
    OPENAI_API_KEY,
    OUT_OF_DOMAIN_MESSAGE,
    QUERY_UTTERANCE_CLASSIFIER_ENABLED,
)
from app.services.query_route_classifier import classify_user_utterance, UtteranceLabels
from app.database.session import get_db_context
from app.services.rag_unified import rag_query_unified, rag_query_stream_unified

# Intent cần routing chuyên biệt (không dùng RAG thuần)
SPECIALIZED_INTENTS = {
    "tao_bao_cao", "soan_thao_van_ban", "tom_tat_van_ban",
    "so_sanh_van_ban", "kiem_tra_ho_so", "huong_dan_thu_tuc",
    "trich_xuat_van_ban", "can_cu_phap_ly", "giai_thich_quy_dinh",
    "admin_planning",
    # Commune-level intents (Cán bộ VHXH cấp xã)
    "xu_ly_vi_pham_hanh_chinh", "kiem_tra_thanh_tra",
    "thu_tuc_hanh_chinh", "hoa_giai_van_dong",
    "bao_ve_xa_hoi", "to_chuc_su_kien_cong", "bao_ton_phat_trien",
}

# Patterns nhận diện câu hỏi tham chiếu ngữ cảnh hội thoại
CONTEXT_REFERENCE_PATTERNS = [
    r"văn bản (trên|này|đó|vừa rồi|vừa soạn)",
    r"kế hoạch (trên|này|đó|vừa rồi)",
    r"nội dung (trên|này|đó)",
    r"quyết định (trên|này|đó)",
    r"thông báo (trên|này|đó)",
    r"báo cáo (trên|này|đó)",
    r"công văn (trên|này|đó)",
    r"(nó|cái đó|cái này|cái trên)",
]

log = logging.getLogger(__name__)

# Ninh Bình tool: dùng shared router (keywords bao gồm huyện Yên Mô, Yên Khánh, ...)
from app.services.ninh_binh_router import should_use_ninh_binh_tool as _should_use_ninh_binh_tool
from app.services.ninh_binh_web_search import search_wikipedia, _is_wikipedia_insufficient


LEGAL_WEB_BLOCK_PATTERN = re.compile(
    r"\bluật\b|\bnghị\s*định\b|\bthông\s*tư\b|\bđiều\b|\bkhoản\b|\bpháp\s+luật\b",
    re.IGNORECASE,
)


def _is_context_reference_regex(question: str) -> bool:
    """Fallback regex: tham chiếu ngữ cảnh hội thoại."""
    q = question.lower().strip()
    return any(re.search(p, q) for p in CONTEXT_REFERENCE_PATTERNS)


def _is_context_reference(
    question: str,
    labels: Optional[UtteranceLabels],
) -> bool:
    if labels is not None:
        return bool(labels.references_prior_message_context)
    return _is_context_reference_regex(question)


def _resolve_follow_up_question(
    question: str,
    conversation_id: Optional[str],
    labels: Optional[UtteranceLabels] = None,
) -> str:
    """Bổ sung ngữ cảnh tài liệu cho câu hỏi nối tiếp.

    Handles two types of follow-up:
    1. Document reference ("văn bản trên", "kế hoạch trên") → uses document context
    2. Topic follow-up (short questions) → uses last_topic context
    """
    if not conversation_id:
        return question

    ctx = conversation_store.get_context(conversation_id)

    # Check for document context reference
    if _is_context_reference(question, labels):
        doc_ctx = conversation_store.get_last_document_context(conversation_id)
        if doc_ctx:
            # Build enriched question with document context
            enrichment_parts = []
            if doc_ctx.get("loai_van_ban"):
                enrichment_parts.append(f"loại văn bản: {doc_ctx['loai_van_ban']}")
            if doc_ctx.get("linh_vuc"):
                enrichment_parts.append(f"lĩnh vực: {doc_ctx['linh_vuc']}")
            if doc_ctx.get("co_quan"):
                enrichment_parts.append(f"cơ quan: {doc_ctx['co_quan']}")
            if doc_ctx.get("chu_de"):
                enrichment_parts.append(f"chủ đề: {doc_ctx['chu_de']}")

            if enrichment_parts:
                enrichment = ", ".join(enrichment_parts)
                resolved = f"{question} (ngữ cảnh: {enrichment})"
                log.info("[CONTEXT] Resolved context reference: %s → %s",
                         question[:50], resolved[:80])
                return resolved

    # Fallback: topic-based follow-up
    if looks_like_follow_up(question):
        topic = ctx.get("last_topic")
        if topic:
            resolved = f"{question} trong tài liệu: {topic}"
            log.info("[CONTEXT] Resolved topic follow-up: %s → %s",
                     question[:50], resolved[:80])
            return resolved

    return question


def _extract_document_metadata_from_answer(answer: str, question: str) -> Optional[dict]:
    """Extract document metadata from a generated draft answer for session memory."""
    from app.tools.draft_tool import extract_document_metadata
    return extract_document_metadata(question)


def _is_legal_question_regex(question: str) -> bool:
    """Fallback: từ khóa pháp lý — không đưa sang web tổng quát."""
    return bool(LEGAL_WEB_BLOCK_PATTERN.search((question or "").strip()))


def _is_legal_question(question: str, labels: Optional[UtteranceLabels]) -> bool:
    if labels is not None:
        return bool(labels.is_legal_or_admin_query)
    return _is_legal_question_regex(question)


async def _run_hybrid_general_search(question: str) -> Dict[str, Any]:
    """Wikipedia first, then OpenAI web search fallback."""
    wiki_result = await search_wikipedia(question)
    if not _is_wikipedia_insufficient(wiki_result):
        return {
            "answer": wiki_result.get("answer", ""),
            "sources": wiki_result.get("sources", []),
            "pipeline": "wikipedia",
        }

    from app.tools.openai_web_search_tool import run as openai_web_search_run
    web_result = await openai_web_search_run(question)
    return {
        "answer": web_result.get("answer", ""),
        "sources": web_result.get("sources", []),
        "pipeline": "openai_web_search",
    }


async def process(
    question: str,
    temperature: float = 0.5,
    filters: Optional[dict] = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Xử lý câu hỏi qua Copilot pipeline (non-streaming).

    Pipeline:
        1. Lưu message vào conversation memory (nếu có)
        2. Intent: ``detect_intent`` (guard → structural → PhoBERT nếu bật → semantic → LLM)
        3. Context Resolution (câu tham chiếu hội thoại)
        4. Routing: intent chuyên biệt + confidence → ``route_query``; ngược lại → ``rag_query_unified`` (v2)
        5. Lưu metadata văn bản (soạn thảo) nếu có
        6. Trả response kèm confidence + intent
    """
    # 1. Lưu user message
    if conversation_id:
        conversation_store.add_message(conversation_id, "user", question)

    utterance_labels: Optional[UtteranceLabels] = None
    if QUERY_UTTERANCE_CLASSIFIER_ENABLED and OPENAI_API_KEY:
        utterance_labels = await classify_user_utterance(
            question, has_conversation=bool(conversation_id)
        )

    # 2. Detect intent
    intent_result = await detect_intent(question)
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]
    log.info("[LOG] intent=%s, confidence=%.2f, question=%s",
             intent, confidence, question[:80])

    is_legal = _is_legal_question(question, utterance_labels)
    is_specialized = intent in SPECIALIZED_INTENTS and confidence >= 0.5

    # 2b. Hybrid web search (general, non-legal only): Wikipedia -> OpenAI web fallback
    # Skip web search if intent is specialized (admin_planning needs multi-step pipeline)
    if not is_legal and _should_use_ninh_binh_tool(question) and not is_specialized:
        hybrid = await _run_hybrid_general_search(question)
        answer = hybrid.get("answer", "")
        sources = hybrid.get("sources", [])
        pipeline = hybrid.get("pipeline", "wikipedia")
        log.info("[LOG] routed_to=%s", pipeline)
        if conversation_id:
            conversation_store.add_message(conversation_id, "assistant", answer)
        return {
            "answer": answer,
            "sources": sources,
            "confidence": 0.85,
            "intent": intent,
            "query_analysis": None,
            "metadata": {"pipeline": pipeline},
        }

    # 3. Domain guard trước retrieval
    if not is_in_document_domain(question):
        answer = OUT_OF_DOMAIN_MESSAGE
        sources = []
        query_analysis = None
        log.info("[LOG] domain_check=out_of_domain")
    elif intent in SPECIALIZED_INTENTS and confidence >= 0.5:
        # Route đến handler chuyên biệt (draft, report, summarize, ...)
        routed = await route_query(
            question=question,
            temperature=temperature,
            intent_override=intent_result,
            conversation_id=conversation_id,
            utterance_labels=utterance_labels,
        )
        answer = routed["answer"]
        sources = routed.get("sources", [])
        query_analysis = routed.get("metadata", {}).get("query_analysis")
        log.info("[LOG] routed_to=%s, retrieved_docs=%d",
                 routed.get("metadata", {}).get("pipeline", "unknown"), len(sources))

        # Save document metadata if draft intent
        if intent == "soan_thao_van_ban" and conversation_id:
            doc_meta = _extract_document_metadata_from_answer(answer, question)
            if doc_meta:
                conversation_store.update_document_context(conversation_id, doc_meta)
                log.info("[LOG] saved_document_context=%s", doc_meta)
    else:
        resolved_question = _resolve_follow_up_question(
            question, conversation_id, utterance_labels
        )
        async with get_db_context() as db:
            rag_result = await rag_query_unified(
                resolved_question,
                db,
                temperature=temperature,
                conversation_id=conversation_id,
                utterance_labels=utterance_labels,
            )
        answer = rag_result["answer"]
        sources = rag_result["sources"]
        query_analysis = rag_result.get("query_analysis")

        # Check if fallback was used
        fallback_used = not sources or "suy luận" in answer.lower()
        log.info("[LOG] retrieved_docs=%d, fallback_reasoning=%s",
                 len(sources), fallback_used)

    # 4. Lưu assistant response
    if conversation_id:
        conversation_store.add_message(conversation_id, "assistant", answer)
        if sources:
            first_meta = sources[0].get("metadata", {}) if sources else {}
            last_topic = (
                first_meta.get("title")
                or first_meta.get("law_name")
                or first_meta.get("source_file")
            )
            conversation_store.update_context(
                conversation_id,
                last_topic=last_topic,
                last_intent=intent,
                last_question=question,
            )

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "intent": intent,
        "query_analysis": query_analysis,
    }


async def process_stream(
    question: str,
    temperature: float = 0.5,
    filters: Optional[dict] = None,
    conversation_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streaming Copilot pipeline.

    Yields tokens. Token đầu tiên chứa intent metadata,
    token thứ hai chứa sources metadata, còn lại là text.
    """
    # Lưu user message
    if conversation_id:
        conversation_store.add_message(conversation_id, "user", question)

    utterance_labels: Optional[UtteranceLabels] = None
    if QUERY_UTTERANCE_CLASSIFIER_ENABLED and OPENAI_API_KEY:
        utterance_labels = await classify_user_utterance(
            question, has_conversation=bool(conversation_id)
        )

    # Detect intent (nhanh, trước khi stream)
    intent_result = await detect_intent(question)
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]
    log.info("[LOG] stream: intent=%s, confidence=%.2f", intent, confidence)
    is_legal = _is_legal_question(question, utterance_labels)
    is_specialized = intent in SPECIALIZED_INTENTS and confidence >= 0.5

    import json

    # Emit intent metadata
    yield json.dumps({"type": "intent", "data": intent_result}, ensure_ascii=False) + "\n"

    collected_tokens: list[str] = []

    # Hybrid web search (non-legal): Wikipedia first, then OpenAI web fallback
    # Skip web search if intent is specialized (admin_planning needs multi-step pipeline)
    if not is_legal and _should_use_ninh_binh_tool(question) and not is_specialized:
        hybrid = await _run_hybrid_general_search(question)
        answer = hybrid.get("answer", "")
        sources_meta = json.dumps({"type": "sources", "data": hybrid.get("sources", [])}, ensure_ascii=False) + "\n"
        yield sources_meta
        collected_tokens.append(sources_meta)
        yield answer
        collected_tokens.append(answer)
        if conversation_id:
            conversation_store.add_message(conversation_id, "assistant", answer)
        return

    if not is_in_document_domain(question):
        answer = OUT_OF_DOMAIN_MESSAGE
        sources_meta = json.dumps({"type": "sources", "data": []}, ensure_ascii=False) + "\n"
        collected_tokens.append(sources_meta)
        yield sources_meta
        collected_tokens.append(answer)
        yield answer
    elif intent in SPECIALIZED_INTENTS and confidence >= 0.5:
        # Specialized intents: gọi route_query (non-streaming) rồi emit kết quả
        routed = await route_query(
            question=question,
            temperature=temperature,
            intent_override=intent_result,
            conversation_id=conversation_id,
            utterance_labels=utterance_labels,
        )
        answer = routed["answer"]
        routed_sources = routed.get("sources", [])
        sources_meta = json.dumps({"type": "sources", "data": routed_sources}, ensure_ascii=False) + "\n"
        collected_tokens.append(sources_meta)
        yield sources_meta
        collected_tokens.append(answer)
        yield answer

        # Save document metadata if draft intent
        if intent == "soan_thao_van_ban" and conversation_id:
            doc_meta = _extract_document_metadata_from_answer(answer, question)
            if doc_meta:
                conversation_store.update_document_context(conversation_id, doc_meta)
                log.info("[LOG] stream: saved_document_context=%s", doc_meta)
    else:
        resolved_question = _resolve_follow_up_question(
            question, conversation_id, utterance_labels
        )
        async with get_db_context() as db:
            async for token in rag_query_stream_unified(
                resolved_question,
                db,
                temperature=temperature,
                conversation_id=conversation_id,
                utterance_labels=utterance_labels,
            ):
                collected_tokens.append(token)
                yield token

    # Lưu full response (bỏ token metadata đầu tiên)
    if conversation_id and collected_tokens:
        text_tokens = collected_tokens[1:]  # skip sources metadata
        full_answer = "".join(text_tokens)
        conversation_store.add_message(conversation_id, "assistant", full_answer)
