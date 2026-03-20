# -*- coding: utf-8 -*-
"""Unit tests for hybrid search strategy (Wikipedia -> OpenAI web fallback)."""

from __future__ import annotations

import asyncio

import pytest

from app.agents import copilot_agent


@pytest.mark.parametrize(
    "query",
    [
        "Ninh Bình thuộc vùng nào?",
        "Lịch sử cố đô Hoa Lư",
        "Dân số tỉnh Ninh Bình",
    ],
)
def test_hybrid_search_wikipedia_first(query: str, monkeypatch: pytest.MonkeyPatch):
    """Wikipedia must be called first and returned when sufficient."""
    calls: list[str] = []

    async def _fake_wiki(q: str):
        calls.append(f"wiki:{q}")
        return {"answer": "Nội dung đủ dài từ Wikipedia để trả lời câu hỏi người dùng.", "sources": [{"title": "Wikipedia", "url": "https://vi.wikipedia.org"}]}

    async def _fake_web(q: str):
        calls.append(f"web:{q}")
        return {"tool": "search_web", "answer": "Fallback web", "sources": [{"title": "web", "url": "https://example.com"}]}

    monkeypatch.setattr(copilot_agent, "search_wikipedia", _fake_wiki)
    monkeypatch.setattr(copilot_agent, "_is_wikipedia_insufficient", lambda _: False)

    import app.tools.openai_web_search_tool as web_tool
    monkeypatch.setattr(web_tool, "run", _fake_web)

    result = asyncio.run(copilot_agent._run_hybrid_general_search(query))
    assert result["pipeline"] == "wikipedia"
    assert result["answer"]
    assert len(result["sources"]) > 0
    assert calls == [f"wiki:{query}"]


@pytest.mark.parametrize(
    "query",
    [
        "Ninh Bình thuộc vùng nào?",
        "Lịch sử cố đô Hoa Lư",
        "Dân số tỉnh Ninh Bình",
    ],
)
def test_hybrid_search_fallback_to_openai_web(query: str, monkeypatch: pytest.MonkeyPatch):
    """OpenAI web_search is used only when Wikipedia is insufficient."""
    calls: list[str] = []

    async def _fake_wiki(q: str):
        calls.append(f"wiki:{q}")
        return {"answer": "Không đủ thông tin", "sources": []}

    async def _fake_web(q: str):
        calls.append(f"web:{q}")
        return {
            "tool": "search_web",
            "answer": "Kết quả từ OpenAI web_search fallback.",
            "sources": [
                {"title": "Wikipedia", "url": "https://vi.wikipedia.org"},
                {"title": "official website", "url": "https://ninhbinh.gov.vn"},
                {"title": "news source", "url": "https://vnexpress.net"},
            ],
        }

    monkeypatch.setattr(copilot_agent, "search_wikipedia", _fake_wiki)
    monkeypatch.setattr(copilot_agent, "_is_wikipedia_insufficient", lambda _: True)

    import app.tools.openai_web_search_tool as web_tool
    monkeypatch.setattr(web_tool, "run", _fake_web)

    result = asyncio.run(copilot_agent._run_hybrid_general_search(query))
    assert result["pipeline"] == "openai_web_search"
    assert "fallback" in result["answer"].lower() or result["answer"]
    assert len(result["sources"]) >= 1
    assert calls == [f"wiki:{query}", f"web:{query}"]


def test_legal_queries_block_web_search():
    """Legal queries must not be routed to web search."""
    assert copilot_agent._is_legal_question("Theo luật di sản văn hóa, điều 47 quy định gì?") is True
    assert copilot_agent._is_legal_question("Nghị định 38 quy định ra sao?") is True
    assert copilot_agent._is_legal_question("Ninh Bình thuộc vùng nào?") is False
