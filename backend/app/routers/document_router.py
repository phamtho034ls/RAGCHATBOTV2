"""
Document Router – API endpoints cho xử lý văn bản pháp luật.

Endpoints:
    POST /api/document/summarize  — Tóm tắt văn bản
    POST /api/document/compare    — So sánh văn bản
    POST /api/report/generate     — Tạo báo cáo
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    CompareRequest,
    CompareResponse,
    ReportRequest,
    ReportResponse,
    SourceChunk,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["document"])


@router.post("/api/document/summarize", response_model=SummarizeResponse)
async def summarize_document(req: SummarizeRequest):
    """Tóm tắt văn bản pháp luật.

    Tìm văn bản liên quan qua RAG và tạo bản tóm tắt có cấu trúc.
    """
    if not req.query.strip():
        raise HTTPException(400, "Yêu cầu tóm tắt không được để trống")

    from app.services.document_summarizer import summarize_document as _summarize

    result = await _summarize(
        query=req.query,
        temperature=req.temperature,
        dataset_id=req.dataset_id,
    )

    return SummarizeResponse(
        summary=result["summary"],
        sources=[
            SourceChunk(
                content=s["content"],
                score=s["score"],
                dataset_id=s.get("dataset_id"),
                metadata=s.get("metadata"),
            )
            for s in result["sources"]
        ],
    )


@router.post("/api/document/compare", response_model=CompareResponse)
async def compare_documents(req: CompareRequest):
    """So sánh hai văn bản pháp luật.

    Phân tích điểm giống, khác và thay đổi chính giữa hai văn bản.
    """
    if not req.query.strip():
        raise HTTPException(400, "Yêu cầu so sánh không được để trống")

    from app.services.document_comparator import compare_documents as _compare

    result = await _compare(
        query=req.query,
        temperature=req.temperature,
        dataset_id_1=req.dataset_id_1,
        dataset_id_2=req.dataset_id_2,
    )

    return CompareResponse(
        comparison=result["comparison"],
        sources_doc1=[
            SourceChunk(
                content=s["content"],
                score=s["score"],
                dataset_id=s.get("dataset_id"),
                metadata=s.get("metadata"),
            )
            for s in result.get("sources_doc1", [])
        ],
        sources_doc2=[
            SourceChunk(
                content=s["content"],
                score=s["score"],
                dataset_id=s.get("dataset_id"),
                metadata=s.get("metadata"),
            )
            for s in result.get("sources_doc2", [])
        ],
    )


@router.post("/api/report/generate", response_model=ReportResponse)
async def generate_report(req: ReportRequest):
    """Tạo báo cáo hành chính.

    Tìm tài liệu liên quan và soạn báo cáo theo cấu trúc chuẩn.
    """
    if not req.request.strip():
        raise HTTPException(400, "Yêu cầu báo cáo không được để trống")

    from app.services.report_generator import generate_report as _generate

    result = await _generate(
        request=req.request,
        temperature=req.temperature,
        dataset_id=req.dataset_id,
    )

    return ReportResponse(
        report=result["report"],
        sources=[
            SourceChunk(
                content=s["content"],
                score=s["score"],
                dataset_id=s.get("dataset_id"),
                metadata=s.get("metadata"),
            )
            for s in result["sources"]
        ],
    )
