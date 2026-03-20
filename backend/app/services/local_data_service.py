"""
Local Data Service – tra cứu thông tin hành chính địa phương.

Cung cấp: dân số, diện tích, đơn vị hành chính, quy mô quản lý
cho các cấp chính quyền tại Ninh Bình.

Pipeline:
    Query → extract_location() → lookup_local_data()
          → Ninh Bình web search → structured result
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

ADMIN_LEVEL_PATTERNS = {
    "tinh": re.compile(
        r"\b(tỉnh|thành phố trực thuộc trung ương)\b", re.IGNORECASE
    ),
    "huyen": re.compile(
        r"\b(huyện|quận|thị xã|thành phố thuộc tỉnh)\b", re.IGNORECASE
    ),
    "xa": re.compile(r"\b(xã|phường|thị trấn)\b", re.IGNORECASE),
}

_LOCATION_RE = re.compile(
    r"(xã|phường|thị trấn|huyện|quận|thị xã|thành phố|tỉnh)"
    r"\s+([\w\s]{2,30}?)(?:\s*,|\s*$|\s+(?:tỉnh|huyện|quận|của))",
    re.IGNORECASE,
)


def detect_admin_level(text: str) -> str:
    """Detect administrative level from text."""
    for level, pattern in ADMIN_LEVEL_PATTERNS.items():
        if pattern.search(text):
            return level
    return "unknown"


def extract_location(text: str) -> Optional[str]:
    """Extract location name from query text."""
    m = _LOCATION_RE.search(text)
    if m:
        return f"{m.group(1)} {m.group(2)}".strip()
    return None


async def lookup_local_data(
    query: str,
    data_type: str = "general",
) -> Dict[str, Any]:
    """Tra cứu thông tin hành chính địa phương.

    Args:
        query: Câu hỏi hoặc tên địa phương.
        data_type: "general" | "population" | "area" | "admin_units".

    Returns:
        Dict with location, admin_level, data (text answer), sources.
    """
    from app.services.ninh_binh_web_search import search_ninh_binh

    location = extract_location(query) or query

    search_queries = {
        "general": (
            f"thông tin hành chính {location} Ninh Bình "
            f"dân số diện tích đơn vị hành chính"
        ),
        "population": f"dân số {location} Ninh Bình",
        "area": f"diện tích {location} Ninh Bình",
        "admin_units": (
            f"đơn vị hành chính {location} Ninh Bình "
            f"số xã phường thôn xóm"
        ),
    }

    search_q = search_queries.get(data_type, search_queries["general"])
    log.info("[LOCAL_DATA] query=%s, data_type=%s", location, data_type)

    result = await search_ninh_binh(search_q)
    admin_level = detect_admin_level(query)

    return {
        "location": location,
        "admin_level": admin_level,
        "data": result.get("answer", ""),
        "sources": result.get("sources", []),
        "data_type": data_type,
    }
