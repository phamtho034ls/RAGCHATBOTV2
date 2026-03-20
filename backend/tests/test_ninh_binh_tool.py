# -*- coding: utf-8 -*-
"""Unit tests for Ninh Bình search tool."""

from __future__ import annotations

import pytest

from app.tools.ninh_binh_search_tool import (
    NinhBinhSearchTool,
    get_ninh_binh_tool,
    run,
    _has_legal_keywords,
)


class TestNinhBinhSearchTool:
    """Test NinhBinhSearchTool."""

    def test_ninh_binh_co_bao_nhieu_huyen(self):
        """Ninh Bình có bao nhiêu huyện?"""
        tool = get_ninh_binh_tool()
        result = tool.search_ninh_binh_info("Ninh Bình có bao nhiêu huyện?")
        assert "Câu trả lời" in result
        assert "8" in result or "8 đơn vị" in result or "Ninh Bình" in result
        assert "huyện" in result.lower() or "đơn vị" in result.lower()

    def test_cac_diem_du_lich_noi_tieng(self):
        """Các điểm du lịch nổi tiếng ở Ninh Bình."""
        tool = get_ninh_binh_tool()
        result = tool.search_ninh_binh_info("Các điểm du lịch nổi tiếng ở Ninh Bình")
        assert "Câu trả lời" in result
        assert any(
            x in result
            for x in ["Tràng An", "Tam Cốc", "Bái Đính", "du lịch", "Ninh Bình"]
        )

    def test_dan_so_tinh_ninh_binh(self):
        """Dân số tỉnh Ninh Bình."""
        tool = get_ninh_binh_tool()
        result = tool.search_ninh_binh_info("Dân số tỉnh Ninh Bình")
        assert "Câu trả lời" in result
        assert "dân số" in result.lower() or "người" in result.lower() or "triệu" in result.lower()

    def test_trang_an_thuoc_tinh_nao(self):
        """Tràng An thuộc tỉnh nào."""
        tool = get_ninh_binh_tool()
        result = tool.search_ninh_binh_info("Tràng An thuộc tỉnh nào")
        assert "Ninh Bình" in result

    def test_legal_keywords_refused(self):
        """Query with legal keywords must not use tool."""
        tool = get_ninh_binh_tool()
        result = tool.search_ninh_binh_info(
            "Nghị định về quản lý du lịch Ninh Bình quy định gì?"
        )
        assert "pháp luật" in result.lower() or "công cụ tra cứu" in result.lower()

    @pytest.mark.asyncio
    async def test_run_async(self):
        """Async run() returns valid result."""
        result = await run("Ninh Bình có bao nhiêu huyện?")
        assert result["tool"] == "search_ninh_binh_info"
        assert "result" in result
        assert isinstance(result.get("sources", []), list)
        assert "Ninh Bình" in result["result"]


class TestLegalKeywordGuard:
    """Test legal keyword detection."""

    def test_has_legal_keywords_positive(self):
        assert _has_legal_keywords("Luật Di sản văn hóa quy định gì?") is True
        assert _has_legal_keywords("Nghị định 38/2021") is True
        assert _has_legal_keywords("Điều 47 khoản 2") is True
        assert _has_legal_keywords("theo pháp luật") is True

    def test_has_legal_keywords_negative(self):
        assert _has_legal_keywords("Ninh Bình có bao nhiêu huyện?") is False
        assert _has_legal_keywords("Các điểm du lịch ở Tràng An") is False
