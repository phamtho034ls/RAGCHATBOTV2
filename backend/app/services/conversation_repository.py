"""Persistent chat conversations (PostgreSQL) — thay thế in-memory store cho /api/chat và API CRUD."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import ChatConversation, ChatMessage

log = logging.getLogger(__name__)

_MAX_MESSAGES = 100


def _naive_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _utc_now_iso() -> str:
    return _naive_utc().isoformat()


async def create_conversation(db: AsyncSession, title: Optional[str] = None) -> Dict[str, Any]:
    conv_id = uuid.uuid4().hex[:12]
    t = title or f"Cuộc hội thoại {conv_id}"
    row = ChatConversation(id=conv_id, title=t, context_json="{}")
    db.add(row)
    await db.flush()
    return {
        "id": conv_id,
        "title": t,
        "messages": [],
        "context": {},
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }


async def get_conversation_row(db: AsyncSession, conv_id: str) -> Optional[ChatConversation]:
    r = await db.execute(
        select(ChatConversation).where(ChatConversation.id == conv_id).options(selectinload(ChatConversation.messages))
    )
    return r.scalar_one_or_none()


async def conversation_exists(db: AsyncSession, conv_id: str) -> bool:
    r = await db.execute(select(ChatConversation.id).where(ChatConversation.id == conv_id))
    return r.scalar_one_or_none() is not None


async def list_conversations(db: AsyncSession) -> List[Dict[str, Any]]:
    r = await db.execute(select(ChatConversation).order_by(ChatConversation.updated_at.desc()))
    convs = list(r.scalars())
    out: List[Dict[str, Any]] = []
    for conv in convs:
        mc = await db.scalar(
            select(func.count()).select_from(ChatMessage).where(ChatMessage.conversation_id == conv.id)
        )
        out.append(
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else "",
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else "",
                "message_count": int(mc or 0),
            }
        )
    return out


async def get_conversation_detail_dict(db: AsyncSession, conv_id: str) -> Optional[Dict[str, Any]]:
    row = await get_conversation_row(db, conv_id)
    if not row:
        return None
    msgs = sorted(row.messages, key=lambda m: m.id)
    ctx = {}
    try:
        ctx = json.loads(row.context_json or "{}")
    except json.JSONDecodeError:
        pass
    return {
        "id": row.id,
        "title": row.title,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.created_at.isoformat() if m.created_at else "",
            }
            for m in msgs
        ],
        "context": ctx,
        "created_at": row.created_at.isoformat() if row.created_at else "",
        "updated_at": row.updated_at.isoformat() if row.updated_at else "",
    }


async def get_history(db: AsyncSession, conv_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    row = await get_conversation_row(db, conv_id)
    if not row:
        return []
    msgs = sorted(row.messages, key=lambda m: m.id)[-limit:]
    return [
        {
            "role": m.role,
            "content": m.content,
            "timestamp": m.created_at.isoformat() if m.created_at else "",
        }
        for m in msgs
    ]


async def add_message(db: AsyncSession, conv_id: str, role: str, content: str) -> bool:
    row = await get_conversation_row(db, conv_id)
    if not row:
        return False
    db.add(ChatMessage(conversation_id=conv_id, role=role, content=content))
    row.updated_at = _naive_utc()
    await db.flush()
    r = await db.execute(
        select(ChatMessage.id)
        .where(ChatMessage.conversation_id == conv_id)
        .order_by(ChatMessage.id.desc())
        .offset(_MAX_MESSAGES)
    )
    old_ids = [x[0] for x in r.all()]
    if old_ids:
        await db.execute(delete(ChatMessage).where(ChatMessage.id.in_(old_ids)))
    return True


async def delete_conversation(db: AsyncSession, conv_id: str) -> bool:
    r = await db.execute(delete(ChatConversation).where(ChatConversation.id == conv_id))
    return r.rowcount > 0


async def update_context(db: AsyncSession, conv_id: str, **kwargs: Any) -> bool:
    row = await get_conversation_row(db, conv_id)
    if not row:
        return False
    try:
        ctx = json.loads(row.context_json or "{}")
    except json.JSONDecodeError:
        ctx = {}
    ctx.update(kwargs)
    row.context_json = json.dumps(ctx, ensure_ascii=False)
    row.updated_at = _naive_utc()
    return True


async def update_document_context(db: AsyncSession, conv_id: str, doc_meta: dict) -> bool:
    row = await get_conversation_row(db, conv_id)
    if not row:
        return False
    try:
        ctx = json.loads(row.context_json or "{}")
    except json.JSONDecodeError:
        ctx = {}
    doc_history = ctx.setdefault("document_history", [])
    doc_history.append({**doc_meta, "timestamp": _utc_now_iso()})
    if len(doc_history) > 5:
        ctx["document_history"] = doc_history[-5:]
    ctx["last_document"] = doc_meta
    row.context_json = json.dumps(ctx, ensure_ascii=False)
    row.updated_at = _naive_utc()
    return True


def get_last_document_context_from_ctx(ctx: dict) -> Optional[dict]:
    return ctx.get("last_document") if ctx else None
