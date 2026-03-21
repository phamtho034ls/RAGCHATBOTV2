"""
Intent Router – API endpoint cho Intent Detection.

Endpoints:
    POST /api/intent  — Nhận diện ý định người dùng
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import IntentRequest, IntentResponse
from app.services.intent_detector import detect_intent, get_index_stats

router = APIRouter(tags=["intent"])


@router.get("/api/intent/index-stats")
async def intent_index_stats():
    """Thống kê prototype / intent coverage cho evaluation & UI."""
    return get_index_stats()


@router.post("/api/intent", response_model=IntentResponse)
async def detect_intent_endpoint(req: IntentRequest):
    """Nhận diện ý định (intent) của câu hỏi.

    Trả về loại yêu cầu:
        - tra_cuu_van_ban: Tra cứu văn bản pháp luật
        - huong_dan_thu_tuc: Hướng dẫn thủ tục hành chính
        - kiem_tra_ho_so: Kiểm tra hồ sơ
        - tom_tat_van_ban: Tóm tắt văn bản
        - so_sanh_van_ban: So sánh văn bản
        - tao_bao_cao: Tạo báo cáo
        - soan_thao_van_ban: Soạn thảo văn bản
        - hoi_dap_chung: Hỏi đáp chung
    """
    if not req.question.strip():
        raise HTTPException(400, "Câu hỏi không được để trống")

    result = await detect_intent(req.question)
    return IntentResponse(
        intent=result["intent"],
        confidence=result["confidence"],
    )
