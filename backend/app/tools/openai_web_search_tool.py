"""
OpenAI Web Search Tool.

Use OpenAI Responses API with web_search enabled for non-legal queries
when Wikipedia data is insufficient.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from openai import AsyncOpenAI

from app.config import OPENAI_API_KEY, OPENAI_BASE_URL

log = logging.getLogger(__name__)


def _strip_inline_citations(text: str) -> str:
    """Remove markdown-style inline citations like ([title](url)) or [title](url)."""
    text = re.sub(r"\(\[.*?\]\(https?://[^\)]*\)\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\(https?://[^\)]*\)", r"\1", text)
    text = re.sub(r"\(https?://[^\)]*\)", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _clean_answer(text: str) -> str:
    """Strip inline citations, keep full answer with formatting."""
    raw = _strip_inline_citations((text or "").strip())
    if not raw:
        return ""
    return raw


class OpenAIWebSearchTool:
    """Search the internet using OpenAI web_search tool."""

    def __init__(self) -> None:
        kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        self._client = AsyncOpenAI(**kwargs)

    @staticmethod
    def _extract_sources(response: Any) -> List[Dict[str, str]]:
        """Extract sources from Responses API output annotations when available."""
        sources: List[Dict[str, str]] = []

        # Best-effort: parse response.output[*].content[*].annotations
        output = getattr(response, "output", []) or []
        for item in output:
            content_blocks = getattr(item, "content", []) or []
            for block in content_blocks:
                annotations = getattr(block, "annotations", []) or []
                for ann in annotations:
                    url = getattr(ann, "url", "") or ""
                    title = getattr(ann, "title", "") or getattr(ann, "source", "") or ""
                    if url:
                        entry = {"title": str(title), "url": str(url)}
                        if entry not in sources:
                            sources.append(entry)

        # Fallback: extract links from text if annotations are missing
        if not sources:
            text = getattr(response, "output_text", "") or ""
            for url in re.findall(r"https?://[^\s)\]]+", text):
                entry = {"title": "", "url": url}
                if entry not in sources:
                    sources.append(entry)
        return sources

    async def search_web(self, query: str) -> Dict[str, Any]:
        """Search web and return structured answer + sources."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            optimized_prompt = (
                "Bạn là trợ lý tìm kiếm web cho Government AI Copilot.\n"
                "Mục tiêu: trả lời ngắn gọn, chính xác, ưu tiên thông tin mới nhất.\n\n"
                f"Ngày hiện tại (UTC): {today}\n"
                f"Câu hỏi người dùng: {query}\n\n"
                "Yêu cầu bắt buộc:\n"
                "1) Ưu tiên nguồn chính thức và đáng tin cậy (cổng thông tin cơ quan nhà nước, thống kê, báo chí uy tín).\n"
                "2) Ưu tiên dữ liệu mới nhất; nếu có nhiều số liệu mâu thuẫn, chọn số liệu có ngày cập nhật gần nhất.\n"
                "3) Nêu rõ mốc thời gian của thông tin khi có thể.\n"
                "4) Nếu không tìm thấy dữ liệu mới/cụ thể, nói rõ phần còn thiếu thay vì suy đoán.\n"
                "5) Trả lời bằng tiếng Việt.\n"
                "6) Trả lời đầy đủ ý, rõ ràng, đúng trọng tâm. Không dùng bullet hay heading."
            )
            response = await self._client.responses.create(
                model="gpt-4.1",
                input=optimized_prompt,
                tools=[{"type": "web_search"}],
            )
            answer = _clean_answer((getattr(response, "output_text", "") or "").strip())
            sources = self._extract_sources(response)
            return {
                "answer": answer or "Không tìm thấy thông tin phù hợp trên web.",
                "sources": sources,
            }
        except Exception as e:
            log.error("OpenAI web search failed: %s", e)
            return {
                "answer": "Không thể tìm kiếm web ở thời điểm hiện tại.",
                "sources": [],
            }


_tool: OpenAIWebSearchTool | None = None


def get_openai_web_search_tool() -> OpenAIWebSearchTool:
    global _tool
    if _tool is None:
        _tool = OpenAIWebSearchTool()
    return _tool


async def run(query: str) -> Dict[str, Any]:
    """Async wrapper for agent tool execution."""
    tool = get_openai_web_search_tool()
    result = await tool.search_web(query)
    return {
        "tool": "search_web",
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }
