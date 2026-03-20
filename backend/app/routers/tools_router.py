"""
Tools Router – API endpoints cho AI Tool execution.

Endpoints:
    POST /api/tools/summarize  — Tóm tắt văn bản
    POST /api/tools/extract    — Trích xuất thông tin
    POST /api/tools/draft      — Soạn thảo văn bản
    POST /api/tools/classify   — Phân loại văn bản
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import ToolRequest, ToolResponse, SourceChunk
from app.services.retrieval import has_any_dataset

router = APIRouter(prefix="/api/tools", tags=["tools"])


def _build_tool_response(tool_result: dict) -> ToolResponse:
    """Chuyển kết quả tool sang ToolResponse schema."""
    return ToolResponse(
        tool=tool_result["tool"],
        result=tool_result["result"],
        sources=[
            SourceChunk(
                content=s["content"],
                score=s["score"],
                dataset_id=s.get("dataset_id"),
                metadata=s.get("metadata"),
            )
            for s in tool_result.get("sources", [])
        ],
    )


@router.post("/summarize", response_model=ToolResponse)
async def tool_summarize(req: ToolRequest):
    """Tóm tắt văn bản pháp luật.

    Tìm văn bản liên quan qua RAG và tạo bản tóm tắt có cấu trúc.
    """
    if not req.content.strip():
        raise HTTPException(400, "Nội dung không được để trống")
    if not await has_any_dataset():
        raise HTTPException(404, "Chưa có tài liệu nào được tải lên")

    from app.tools.summarize_tool import run

    result = await run(content=req.content, temperature=req.temperature)
    return _build_tool_response(result)


@router.post("/extract", response_model=ToolResponse)
async def tool_extract(req: ToolRequest):
    """Trích xuất thông tin quan trọng từ văn bản.

    Trích xuất: số hiệu, ngày ban hành, cơ quan, điều khoản chính, v.v.
    """
    if not req.content.strip():
        raise HTTPException(400, "Nội dung không được để trống")
    if not await has_any_dataset():
        raise HTTPException(404, "Chưa có tài liệu nào được tải lên")

    from app.tools.extract_tool import run

    result = await run(content=req.content, temperature=req.temperature)
    return _build_tool_response(result)


@router.post("/draft", response_model=ToolResponse)
async def tool_draft(req: ToolRequest):
    """Soạn thảo văn bản hành chính.

    Soạn công văn, báo cáo, tờ trình, v.v. theo thể thức chuẩn.
    """
    if not req.content.strip():
        raise HTTPException(400, "Nội dung không được để trống")

    from app.tools.draft_tool import run

    result = await run(content=req.content, temperature=req.temperature)
    return _build_tool_response(result)


@router.post("/classify", response_model=ToolResponse)
async def tool_classify(req: ToolRequest):
    """Phân loại văn bản theo loại, lĩnh vực, cấp ban hành.

    Phân loại: loại văn bản, lĩnh vực, cấp ban hành, trạng thái hiệu lực.
    """
    if not req.content.strip():
        raise HTTPException(400, "Nội dung không được để trống")

    from app.tools.classify_tool import run

    result = await run(content=req.content, temperature=req.temperature)
    return _build_tool_response(result)
