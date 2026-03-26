"""Thống nhất RAG: mọi luồng gọi ``rag_chain_v2.rag_query`` (PostgreSQL + hybrid + prompt V2)."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import COPILOT_SYSTEM_PROMPT
from app.services.query_route_classifier import UtteranceLabels

log = logging.getLogger(__name__)


def sources_v2_to_legacy_copilot(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chuẩn hóa sources từ ``rag_chain_v2`` sang format Copilot (content/score/metadata)."""
    out: List[Dict[str, Any]] = []
    for s in sources or []:
        doc_no = (s.get("doc_number") or "").strip()
        title = s.get("document_title") or ""
        meta: Dict[str, Any] = {
            "doc_number": doc_no,
            "law_name": title,
            "article_number": s.get("article_number"),
            "article_title": s.get("article_title"),
            "citation": s.get("citation"),
        }
        meta = {k: v for k, v in meta.items() if v}
        out.append(
            {
                "content": s.get("snippet") or "",
                "score": float(s.get("score") or 0.0),
                "dataset_id": s.get("dataset_id"),
                "metadata": meta,
            }
        )
    return out


async def rag_query_unified(
    question: str,
    db: AsyncSession,
    *,
    temperature: float = 0.3,
    conversation_id: Optional[str] = None,
    doc_number: Optional[str] = None,
    utterance_labels: Optional[UtteranceLabels] = None,
) -> Dict[str, Any]:
    from app.services.rag_chain_v2 import rag_query

    result = await rag_query(
        query=question,
        db=db,
        temperature=temperature,
        doc_number=doc_number,
        conversation_id=conversation_id,
        utterance_labels=utterance_labels,
    )
    return {
        "answer": result.get("answer", "") or "",
        "sources": sources_v2_to_legacy_copilot(result.get("sources", []) or []),
        "query_analysis": result.get("query_analysis"),
        "rewritten_query": None,
        "expanded_queries": [],
        "confidence_score": result.get("confidence_score", 0.0),
        "conversation_id": result.get("conversation_id"),
        "system": COPILOT_SYSTEM_PROMPT,
    }


async def rag_query_stream_unified(
    question: str,
    db: AsyncSession,
    *,
    temperature: float = 0.3,
    conversation_id: Optional[str] = None,
    doc_number: Optional[str] = None,
    utterance_labels: Optional[UtteranceLabels] = None,
) -> AsyncGenerator[str, None]:
    """Tương thích Copilot stream: dòng 1 = JSON sources (kiểu rag_chain cũ), sau đó chunk text."""
    from app.services.rag_chain_v2 import rag_query_stream

    async for item in rag_query_stream(
        query=question,
        db=db,
        temperature=temperature,
        doc_number=doc_number,
        conversation_id=conversation_id,
        utterance_labels=utterance_labels,
    ):
        if not item:
            continue
        s = item.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            t = obj.get("type")
            if t == "meta":
                continue
            if t == "sources":
                mapped = sources_v2_to_legacy_copilot(obj.get("data", []) or [])
                line = {
                    "type": "sources",
                    "data": mapped,
                    "query_analysis": None,
                    "rewritten_query": None,
                    "expanded_queries": [],
                }
                yield json.dumps(line, ensure_ascii=False) + "\n"
            elif t == "text_finalize":
                continue
            else:
                continue
        except json.JSONDecodeError:
            yield item
