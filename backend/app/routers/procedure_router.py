"""
Procedure Router – API endpoints cho thủ tục hành chính.

Endpoints:
    POST /api/procedure/steps  — Tra cứu các bước thủ tục
    POST /api/procedure/check  — Kiểm tra hồ sơ
    GET  /api/procedure/list   — Liệt kê tất cả thủ tục
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ProcedureRequest,
    ProcedureResponse,
    ProcedureInfo,
    ProcedureStep,
    DocumentCheckRequest,
    DocumentCheckResponse,
)
from app.services.procedure_service import (
    search_procedure,
    list_procedures,
)
from app.services.document_checker import check_missing_documents

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/procedure", tags=["procedure"])


@router.post("/steps", response_model=ProcedureResponse)
async def get_procedure_steps(req: ProcedureRequest):
    """Tra cứu các bước thực hiện thủ tục hành chính."""
    procedure = search_procedure(req.procedure_name)

    if procedure:
        return ProcedureResponse(
            procedure=ProcedureInfo(
                procedure_id=procedure["procedure_id"],
                procedure_name=procedure["procedure_name"],
                description=procedure["description"],
                steps=[
                    ProcedureStep(
                        step_number=s["step_number"],
                        description=s["description"],
                        note=s.get("note"),
                    )
                    for s in procedure["steps"]
                ],
                required_documents=procedure["required_documents"],
                processing_time=procedure.get("processing_time"),
                fee=procedure.get("fee"),
                authority=procedure.get("authority"),
            ),
            message=f"Đã tìm thấy thủ tục: {procedure['procedure_name']}",
        )
    else:
        available = list_procedures()
        return ProcedureResponse(
            procedure=None,
            message=f"Không tìm thấy thủ tục '{req.procedure_name}'.",
            available_procedures=[p["procedure_name"] for p in available],
        )


@router.post("/check", response_model=DocumentCheckResponse)
async def check_documents(req: DocumentCheckRequest):
    """Kiểm tra hồ sơ còn thiếu so với yêu cầu thủ tục."""
    # Tìm thủ tục
    procedure = search_procedure(req.procedure_name)
    if not procedure:
        raise HTTPException(
            404,
            f"Không tìm thấy thủ tục '{req.procedure_name}'",
        )

    # Tìm procedure_id
    from app.services.procedure_service import PROCEDURES
    proc_id = ""
    for pid, proc in PROCEDURES.items():
        if proc["procedure_id"] == procedure["procedure_id"]:
            proc_id = pid
            break

    if not proc_id:
        raise HTTPException(500, "Lỗi nội bộ: không tìm thấy procedure_id")

    result = check_missing_documents(proc_id, req.submitted_documents)

    return DocumentCheckResponse(
        procedure_name=result["procedure_name"],
        required_documents=result["required_documents"],
        submitted_documents=result["submitted_documents"],
        missing_documents=result["missing_documents"],
        is_complete=result["is_complete"],
        message=result["message"],
    )


@router.get("/list")
async def list_all_procedures():
    """Liệt kê tất cả thủ tục hành chính có trong hệ thống."""
    procedures = list_procedures()
    return {
        "total": len(procedures),
        "procedures": procedures,
    }
