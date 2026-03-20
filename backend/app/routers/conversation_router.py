"""
Conversation Router – API endpoints cho Conversation Memory.

Endpoints:
    GET  /api/conversations        — Liệt kê conversations
    POST /api/conversations        — Tạo conversation mới
    GET  /api/conversations/{id}   — Lấy chi tiết conversation
    DELETE /api/conversations/{id} — Xóa conversation
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationInfo,
    ConversationListResponse,
    ConversationMessage,
)
from app.memory.conversation_store import conversation_store

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations():
    """Liệt kê tất cả conversations (mới nhất trước)."""
    convs = conversation_store.list_all()
    return ConversationListResponse(
        conversations=[ConversationInfo(**c) for c in convs]
    )


@router.post("", response_model=ConversationInfo)
async def create_conversation(req: ConversationCreate):
    """Tạo conversation mới.

    Trả về conversation_id để sử dụng trong /api/chat.
    """
    conv = conversation_store.create(title=req.title)
    return ConversationInfo(
        id=conv["id"],
        title=conv["title"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        message_count=0,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str):
    """Lấy chi tiết conversation với toàn bộ messages."""
    conv = conversation_store.get(conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation không tồn tại")

    return ConversationDetail(
        id=conv["id"],
        title=conv["title"],
        messages=[
            ConversationMessage(
                role=m["role"],
                content=m["content"],
                timestamp=m["timestamp"],
            )
            for m in conv["messages"]
        ],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
    )


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Xóa conversation."""
    if not conversation_store.delete(conversation_id):
        raise HTTPException(404, "Conversation không tồn tại")
    return {"message": "Đã xóa conversation thành công."}
