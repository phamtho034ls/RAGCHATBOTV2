"""
Query Router – điều phối pipeline xử lý dựa trên intent.

Pipeline:
    User Query
    → Intent Detection
    → Query Router (module này)
    → Chuyển đến tool/service phù hợp
    → Trả response

Routing rules:
    tra_cuu_van_ban     → RAG pipeline (enhanced)
    huong_dan_thu_tuc   → Procedure knowledge base + RAG
    kiem_tra_ho_so      → Document checker
    tom_tat_van_ban     → Document summarizer + RAG
    so_sanh_van_ban     → Document comparator + RAG
    tao_bao_cao         → Report generator + RAG
    trich_xuat_van_ban  → Extract tool (structured extraction)
    hoi_dap_chung       → RAG pipeline (standard)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.services.intent_detector import detect_intent

log = logging.getLogger(__name__)


async def route_query(
    question: str,
    temperature: float = 0.3,
    dataset_id: Optional[str] = None,
    intent_override: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Phân tích intent và điều phối đến pipeline xử lý phù hợp."""
    # 1. Nhận diện intent (dùng override nếu đã detect trước)
    intent_result = intent_override or await detect_intent(question)
    intent = intent_result["intent"]
    log.info("Router: intent=%s, routing to handler...", intent)

    # 2. Dispatch đến handler phù hợp dựa trên intent
    handler = ROUTE_MAP.get(intent, _handle_hoi_dap_chung)
    result = await handler(question, temperature, dataset_id)

    # 3. Gắn intent vào response
    result["intent"] = intent_result
    return result


# ── Route Handlers ──────────────────────────────────────────

async def _handle_tra_cuu_van_ban(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Tra cứu văn bản pháp luật qua RAG pipeline nâng cao."""
    from app.services.rag_chain import rag_query_enhanced

    result = await rag_query_enhanced(question, temperature=temperature)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "metadata": {"pipeline": "rag_enhanced", "query_analysis": result.get("query_analysis")},
    }


async def _handle_huong_dan_thu_tuc(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Hướng dẫn thủ tục hành chính: tìm procedure + bổ sung RAG."""
    from app.services.procedure_service import search_procedure
    from app.services.rag_chain import rag_query_enhanced

    # Tìm thủ tục phù hợp từ knowledge base
    procedure_result = search_procedure(question)

    # Bổ sung context từ RAG
    rag_result = await rag_query_enhanced(question, temperature=temperature, top_k=4)

    if procedure_result:
        # Format thông tin thủ tục
        proc = procedure_result
        procedure_text = _format_procedure(proc)

        # Kết hợp thủ tục + RAG context để LLM trả lời
        from app.services.llm_client import generate
        from app.config import COPILOT_SYSTEM_PROMPT

        rag_context = "\n\n".join(s["content"] for s in rag_result["sources"])
        prompt = f"""Dựa trên thông tin thủ tục hành chính và tài liệu liên quan, hãy hướng dẫn chi tiết:

THÔNG TIN THỦ TỤC:
{procedure_text}

TÀI LIỆU THAM KHẢO:
{rag_context}

CÂU HỎI: {question}

Hãy trả lời chi tiết, rõ ràng, bám sát thủ tục hành chính."""

        answer = await generate(prompt, system=COPILOT_SYSTEM_PROMPT, temperature=temperature)
        from app.services.rag_chain import strip_hallucinated_references
        answer = strip_hallucinated_references(answer, rag_result["sources"])
        return {
            "answer": answer,
            "sources": rag_result["sources"],
            "metadata": {"pipeline": "procedure_guide", "procedure": proc},
        }
    else:
        # Không tìm thấy thủ tục → dùng RAG thông thường
        return {
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "metadata": {"pipeline": "rag_enhanced", "note": "Không tìm thấy thủ tục cụ thể trong knowledge base."},
        }


async def _handle_kiem_tra_ho_so(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Kiểm tra hồ sơ: parse danh sách giấy tờ và kiểm tra thiếu/đủ."""
    from app.services.document_checker import check_documents_from_query

    result = check_documents_from_query(question)
    return {
        "answer": result["message"],
        "sources": [],
        "metadata": {
            "pipeline": "document_checker",
            "check_result": {
                "procedure_name": result.get("procedure_name", ""),
                "required": result.get("required_documents", []),
                "submitted": result.get("submitted_documents", []),
                "missing": result.get("missing_documents", []),
                "is_complete": result.get("is_complete", False),
            },
        },
    }


async def _handle_tom_tat_van_ban(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Tóm tắt văn bản pháp luật – ưu tiên danh sách điều luật từ DB."""
    from app.services.document_summarizer import list_document_articles, summarize_document

    result = await list_document_articles(question)
    if result.get("confidence_score", 0) >= 0.5:
        return {
            "answer": result["summary"],
            "sources": result["sources"],
            "metadata": {"pipeline": "document_summary_articles"},
        }

    fallback = await summarize_document(question, temperature=temperature)
    return {
        "answer": fallback["summary"],
        "sources": fallback["sources"],
        "metadata": {"pipeline": "document_summarizer"},
    }


async def _handle_so_sanh_van_ban(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """So sánh hai văn bản pháp luật."""
    from app.services.document_comparator import compare_documents

    result = await compare_documents(question, temperature=temperature)
    return {
        "answer": result["comparison"],
        "sources": result.get("sources", []),
        "metadata": {"pipeline": "document_comparator"},
    }


async def _handle_tao_bao_cao(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Tạo báo cáo hành chính."""
    from app.services.report_generator import generate_report

    result = await generate_report(question, temperature=temperature)
    return {
        "answer": result["report"],
        "sources": result["sources"],
        "metadata": {"pipeline": "report_generator"},
    }


async def _handle_hoi_dap_chung(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Hỏi đáp chung qua RAG pipeline."""
    from app.services.rag_chain import rag_query_enhanced

    result = await rag_query_enhanced(question, temperature=temperature)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "metadata": {"pipeline": "rag_general"},
    }


def _format_procedure(proc: dict) -> str:
    """Format thông tin thủ tục thành text."""
    lines = [f"Thủ tục: {proc['procedure_name']}"]
    if proc.get("description"):
        lines.append(f"Mô tả: {proc['description']}")
    if proc.get("steps"):
        lines.append("Các bước:")
        for step in proc["steps"]:
            lines.append(f"  {step['step_number']}. {step['description']}")
            if step.get("note"):
                lines.append(f"     Ghi chú: {step['note']}")
    if proc.get("required_documents"):
        lines.append("Hồ sơ yêu cầu:")
        for doc in proc["required_documents"]:
            lines.append(f"  - {doc}")
    if proc.get("processing_time"):
        lines.append(f"Thời gian xử lý: {proc['processing_time']}")
    if proc.get("fee"):
        lines.append(f"Lệ phí: {proc['fee']}")
    return "\n".join(lines)


async def _handle_soan_thao_van_ban(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Soạn thảo văn bản hành chính."""
    from app.tools.draft_tool import run as draft_run

    result = await draft_run(question, temperature=temperature)
    return {
        "answer": result["result"],
        "sources": result.get("sources", []),
        "metadata": {"pipeline": "draft_tool"},
    }


async def _handle_trich_xuat_van_ban(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Trích xuất thông tin/quy định cụ thể từ văn bản pháp luật."""
    from app.tools.extract_tool import run as extract_run

    result = await extract_run(question, temperature=temperature)
    return {
        "answer": result["result"],
        "sources": result.get("sources", []),
        "metadata": {"pipeline": "extract_tool"},
    }


async def _handle_can_cu_phap_ly(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Xác định căn cứ pháp lý cho văn bản/quy định."""
    from app.services.rag_chain import rag_query_enhanced

    result = await rag_query_enhanced(question, temperature=temperature, top_k=10)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "metadata": {"pipeline": "can_cu_phap_ly", "query_analysis": result.get("query_analysis")},
    }


async def _handle_giai_thich_quy_dinh(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Giải thích quy định pháp luật."""
    from app.services.rag_chain import rag_query_enhanced

    result = await rag_query_enhanced(question, temperature=temperature, top_k=8)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "metadata": {"pipeline": "giai_thich_quy_dinh", "query_analysis": result.get("query_analysis")},
    }


async def _handle_admin_planning(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Xử lý câu hỏi quy hoạch/quản lý hành chính – quy trình đa bước.

    Pipeline:
        1. Extract location → lookup local admin data
        2. Retrieve legal regulations via RAG
        3. Combine contexts → LLM multi-step reasoning
    """
    from app.services.local_data_service import lookup_local_data, extract_location
    from app.services.rag_chain import rag_query_enhanced
    from app.services.llm_client import generate
    from app.config import COPILOT_SYSTEM_PROMPT, ADMIN_PLANNING_PROMPT

    # Step 1-2: Extract location and retrieve local data
    location = extract_location(question)
    local_data = ""
    local_sources: list = []
    if location:
        local_result = await lookup_local_data(question, data_type="general")
        local_data = local_result.get("data", "")
        local_sources = local_result.get("sources", [])
        log.info("[ADMIN_PLANNING] location=%s, local_data_len=%d",
                 location, len(local_data))

    # Step 3: Retrieve legal regulations
    rag_result = await rag_query_enhanced(question, temperature=temperature, top_k=10)
    legal_parts: list[str] = []
    for s in rag_result.get("sources", []):
        meta = s.get("metadata", {}) or {}
        header_parts = []
        if meta.get("doc_number"):
            header_parts.append(f"Số hiệu: {meta['doc_number']}")
        if meta.get("law_name"):
            header_parts.append(f"Văn bản: {meta['law_name']}")
        if meta.get("article_number"):
            header_parts.append(f"Điều {meta['article_number']}")
        header = " | ".join(header_parts)
        content = s.get("content", "")
        legal_parts.append(f"[{header}]\n{content}" if header else content)
    legal_context = "\n\n---\n\n".join(legal_parts)
    legal_sources = rag_result.get("sources", [])
    log.info("[ADMIN_PLANNING] legal_docs_found=%d", len(legal_sources))

    # Step 4-6: Multi-step reasoning with combined context
    prompt = ADMIN_PLANNING_PROMPT.format(
        local_data=local_data or "Không có dữ liệu địa phương cụ thể.",
        legal_context=legal_context or "Không tìm thấy quy định pháp luật liên quan trong cơ sở dữ liệu.",
        question=question,
    )

    answer = await generate(
        prompt, system=COPILOT_SYSTEM_PROMPT, temperature=temperature,
    )

    all_sources = local_sources + legal_sources

    # Strip any doc numbers the LLM hallucinated
    from app.services.rag_chain import strip_hallucinated_references
    answer = strip_hallucinated_references(answer, all_sources)

    return {
        "answer": answer,
        "sources": all_sources,
        "metadata": {
            "pipeline": "admin_planning",
            "location": location,
            "local_data_available": bool(local_data),
            "legal_docs_found": len(legal_sources),
        },
    }


async def _handle_commune_level(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Handle commune-level administrative queries via the VHXH officer pipeline."""
    from app.services.rag_chain_v2 import _answer_commune_officer_query
    from app.database.session import get_db_context

    async with get_db_context() as db:
        result = await _answer_commune_officer_query(
            query=question,
            db=db,
            temperature=temperature,
            doc_number=None,
        )
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "metadata": {"pipeline": "commune_officer"},
    }


async def _handle_ninh_binh_info(
    question: str,
    temperature: float,
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Tra cứu thông tin chung về tỉnh Ninh Bình (phi pháp lý)."""
    from app.tools.ninh_binh_search_tool import run as ninh_binh_run

    result = await ninh_binh_run(question)
    return {
        "answer": result.get("result", ""),
        "sources": result.get("sources", []),
        "metadata": {"pipeline": "ninh_binh_search_tool"},
    }


# ── Route Map ───────────────────────────────────────────────
ROUTE_MAP = {
    "ninh_binh_info": _handle_ninh_binh_info,
    "admin_planning": _handle_admin_planning,
    "tra_cuu_van_ban": _handle_tra_cuu_van_ban,
    "can_cu_phap_ly": _handle_can_cu_phap_ly,
    "giai_thich_quy_dinh": _handle_giai_thich_quy_dinh,
    "huong_dan_thu_tuc": _handle_huong_dan_thu_tuc,
    "kiem_tra_ho_so": _handle_kiem_tra_ho_so,
    "tom_tat_van_ban": _handle_tom_tat_van_ban,
    "so_sanh_van_ban": _handle_so_sanh_van_ban,
    "tao_bao_cao": _handle_tao_bao_cao,
    "soan_thao_van_ban": _handle_soan_thao_van_ban,
    "trich_xuat_van_ban": _handle_trich_xuat_van_ban,
    # Commune-level intents → VHXH officer pipeline
    "xu_ly_vi_pham_hanh_chinh": _handle_commune_level,
    "kiem_tra_thanh_tra": _handle_commune_level,
    "thu_tuc_hanh_chinh": _handle_commune_level,
    "hoa_giai_van_dong": _handle_commune_level,
    "bao_ve_xa_hoi": _handle_commune_level,
    "to_chuc_su_kien_cong": _handle_commune_level,
    "bao_ton_phat_trien": _handle_commune_level,
    "hoi_dap_chung": _handle_hoi_dap_chung,
}
