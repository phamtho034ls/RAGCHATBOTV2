"""Chat interaction logger – stores all interactions in PostgreSQL.

Logs:
- user_query
- chatbot_answer
- retrieved_documents (JSON)
- latency_ms
- confidence_score
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ChatLog

log = logging.getLogger(__name__)


async def log_interaction(
    db: AsyncSession,
    user_query: str,
    chatbot_answer: str,
    documents_used: List[str],
    confidence_score: float,
    latency_ms: float,
) -> None:
    """Record a chat interaction to the chat_logs table."""
    try:
        entry = ChatLog(
            user_query=user_query,
            chatbot_answer=chatbot_answer,
            documents_used=json.dumps(documents_used, ensure_ascii=False),
            confidence_score=confidence_score,
            latency_ms=latency_ms,
        )
        db.add(entry)
        await db.flush()
    except Exception as e:
        log.error("Failed to log chat interaction: %s", e)


async def get_recent_logs(
    db: AsyncSession,
    limit: int = 50,
) -> List[dict]:
    """Retrieve recent chat logs for monitoring."""
    from sqlalchemy import select

    stmt = (
        select(ChatLog)
        .order_by(ChatLog.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    logs = result.scalars().all()

    return [
        {
            "id": entry.id,
            "user_query": entry.user_query,
            "chatbot_answer": entry.chatbot_answer[:200] + "..." if entry.chatbot_answer and len(entry.chatbot_answer) > 200 else entry.chatbot_answer,
            "documents_used": json.loads(entry.documents_used) if entry.documents_used else [],
            "confidence_score": entry.confidence_score,
            "latency_ms": entry.latency_ms,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
        }
        for entry in logs
    ]
