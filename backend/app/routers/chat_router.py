"""Chat API router – POST /api/chat and POST /api/chat/stream."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.schemas_v2 import ChatRequest, ChatResponse
from app.services.rag_chain_v2 import rag_query, rag_query_stream

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Synchronous chat endpoint – returns full answer at once."""
    result = await rag_query(
        query=req.query,
        db=db,
        temperature=req.temperature,
        doc_number=req.doc_number,
        conversation_id=req.conversation_id,
    )
    return ChatResponse(**result)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Streaming chat endpoint – returns answer tokens via SSE."""

    async def event_generator():
        async for token in rag_query_stream(
            query=req.query,
            db=db,
            temperature=req.temperature,
            doc_number=req.doc_number,
            conversation_id=req.conversation_id,
        ):
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
