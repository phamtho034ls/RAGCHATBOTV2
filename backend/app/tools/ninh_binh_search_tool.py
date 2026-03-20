"""Tool: Tra cứu thông tin tỉnh Ninh Bình (phi pháp lý).

Pipeline: entity_resolver → query_rewriter → web_search → info_extractor → response.
Returns both structured JSON data and human-readable Vietnamese answer.

Targets: địa lý, du lịch, dân số, kinh tế, hành chính, văn hóa, hạ tầng, tin tức.
NOT for legal documents.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


async def run(query: str) -> Dict[str, Any]:
    """Execute the full Ninh Bình search pipeline.

    Returns:
        {
            "tool": "search_ninh_binh_info",
            "result": str,           # human-readable answer
            "sources": [...],        # source references
            "structured": {...},     # 6-field structured data (optional)
        }
    """
    from app.services.ninh_binh_web_search import search_ninh_binh

    result = await search_ninh_binh(query)
    return {
        "tool": "search_ninh_binh_info",
        "result": result.get("answer", ""),
        "sources": result.get("sources", []),
        "structured": result.get("structured", {}),
    }
