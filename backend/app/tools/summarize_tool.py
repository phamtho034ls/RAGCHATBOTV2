"""
Tool: Tóm tắt văn bản pháp luật.

Tìm văn bản liên quan qua RAG → LLM tóm tắt theo cấu trúc chuẩn.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.document_summarizer import summarize_document


async def run(content: str, temperature: float = 0.3) -> Dict[str, Any]:
    """Thực thi tool tóm tắt văn bản.

    Args:
        content: Câu truy vấn hoặc yêu cầu tóm tắt.
        temperature: Temperature cho LLM.

    Returns:
        {"tool": "summarize", "result": str, "sources": List[dict]}
    """
    result = await summarize_document(query=content, temperature=temperature)
    return {
        "tool": "summarize",
        "result": result["summary"],
        "sources": result["sources"],
    }
