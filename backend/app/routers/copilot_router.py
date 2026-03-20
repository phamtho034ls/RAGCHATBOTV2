"""
Copilot Router – API endpoints cho AI Copilot chính.

Endpoints:
    POST /api/copilot/chat  — Chat với AI Copilot (auto intent detection + routing)
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    CopilotRequest,
    CopilotResponse,
    IntentResult,
    SourceChunk,
)
from app.agents import copilot_agent

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/copilot", tags=["copilot"])


@router.post("/chat", response_model=CopilotResponse)
async def copilot_chat(req: CopilotRequest):
    """Chat với AI Copilot.

    Pipeline:
        Question → Intent Detection → Query Router → Tool/RAG → Response
    """
    if not req.question.strip():
        raise HTTPException(400, "Câu hỏi không được để trống")

    result = await copilot_agent.process(
        question=req.question,
        temperature=req.temperature,
    )
    intent_data = {
        "intent": result.get("intent", "hoi_dap_chung"),
        "confidence": result.get("confidence", 0.0),
    }

    return CopilotResponse(
        answer=result["answer"],
        intent=IntentResult(
            intent=intent_data["intent"],
            confidence=intent_data["confidence"],
        ),
        sources=[
            SourceChunk(
                content=s["content"],
                score=s["score"],
                dataset_id=s.get("dataset_id"),
                metadata=s.get("metadata"),
            )
            for s in result.get("sources", [])
        ],
        metadata=result.get("metadata", {}),
    )


@router.post("/chat/stream")
async def copilot_chat_stream(req: CopilotRequest):
    """Streaming chat với AI Copilot.

    Gửi intent + metadata trước, sau đó stream câu trả lời.
    """
    if not req.question.strip():
        raise HTTPException(400, "Câu hỏi không được để trống")

    async def event_generator():
        async for token in copilot_agent.process_stream(
            question=req.question,
            temperature=req.temperature,
        ):
            yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
