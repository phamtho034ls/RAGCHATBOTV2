"""
Report Generator – tạo báo cáo hành chính.

Tìm nội dung liên quan qua RAG → LLM tạo báo cáo theo cấu trúc chuẩn.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.config import REPORT_PROMPT, COPILOT_SYSTEM_PROMPT
from app.services.llm_client import generate
from app.services.retrieval import search_all, format_sources, rewrite_query

log = logging.getLogger(__name__)


async def generate_report(
    request: str,
    temperature: float = 0.3,
    top_k: int = 10,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Tạo báo cáo hành chính dựa trên yêu cầu.

    Pipeline:
        Request → Rewrite → RAG Retrieval (top 10) → Report LLM

    Args:
        request: Yêu cầu tạo báo cáo (VD: "Viết báo cáo tóm tắt nghị định 01/2021")
        temperature: Temperature cho LLM.
        top_k: Số chunks lấy từ RAG.
        dataset_id: ID dataset cụ thể (optional).

    Returns:
        {"report": str, "sources": List[dict]}
    """
    # 1. Rewrite để tối ưu retrieval
    rewritten = await rewrite_query(request)

    # 2. Retrieval
    results = await search_all(rewritten, top_k=top_k)

    if not results:
        return {
            "report": "Không tìm thấy tài liệu liên quan để tạo báo cáo.",
            "sources": [],
        }

    # 3. Ghép nội dung tham khảo
    content = "\n\n---\n\n".join(
        f"[Nguồn {i+1}]\n{doc['text']}" for i, doc in enumerate(results)
    )

    # 4. Tạo prompt báo cáo
    prompt = REPORT_PROMPT.format(content=content, request=request)

    # 5. Generate report
    report = await generate(
        prompt,
        system=COPILOT_SYSTEM_PROMPT,
        temperature=temperature,
    )

    sources = format_sources(results)
    return {"report": report, "sources": sources}
