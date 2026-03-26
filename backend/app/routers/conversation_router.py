"""
Conversation Router – API endpoints cho Conversation Memory.

Endpoints:
    GET  /api/conversations        — Liệt kê conversations
    POST /api/conversations        — Tạo conversation mới
    GET  /api/conversations/{id}   — Lấy chi tiết conversation
    DELETE /api/conversations/{id} — Xóa conversation
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationInfo,
    ConversationListResponse,
    ConversationMessage,
)
from app.services import conversation_repository as conv_repo

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Lịch sử tin nhắn (10 lượt gần nhất ≈ limit messages)."""
    if not await conv_repo.conversation_exists(db, conversation_id):
        raise HTTPException(404, "Conversation không tồn tại")
    msgs = await conv_repo.get_history(db, conversation_id, limit=limit)
    return {"conversation_id": conversation_id, "messages": msgs}


@router.get("", response_model=ConversationListResponse)
async def list_conversations(db: AsyncSession = Depends(get_db)):
    """Liệt kê tất cả conversations (mới nhất trước)."""
    convs = await conv_repo.list_conversations(db)
    return ConversationListResponse(
        conversations=[ConversationInfo(**c) for c in convs]
    )


@router.post("", response_model=ConversationInfo)
async def create_conversation(req: ConversationCreate, db: AsyncSession = Depends(get_db)):
    """Tạo conversation mới.

    Trả về conversation_id để sử dụng trong /api/chat.
    """
    conv = await conv_repo.create_conversation(db, title=req.title)
    return ConversationInfo(
        id=conv["id"],
        title=conv["title"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        message_count=0,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)):
    """Lấy chi tiết conversation với toàn bộ messages."""
    conv = await conv_repo.get_conversation_detail_dict(db, conversation_id)
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
async def delete_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)):
    """Xóa conversation."""
    if not await conv_repo.delete_conversation(db, conversation_id):
        raise HTTPException(404, "Conversation không tồn tại")
    return {"message": "Đã xóa conversation thành công."}
