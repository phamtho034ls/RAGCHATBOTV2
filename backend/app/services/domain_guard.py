"""Domain guard to reject out-of-scope questions before retrieval."""

from __future__ import annotations

import re

DOMAIN_HINT_PATTERNS = [
    r"điều\s+\d+",
    r"khoản\s+\d+",
    r"nghị định",
    r"thông tư",
    r"quy định",
    r"văn bản",
    r"hồ sơ",
    r"thủ tục",
    r"cấp phép",
    r"quy trình",
]

OUT_OF_DOMAIN_CHAT_PATTERNS = [
    r"thời tiết",
    r"bóng đá",
    r"chứng khoán",
    r"âm nhạc",
    r"phim",
    r"nấu ăn",
    r"du lịch",
    r"truyện",
]

FOLLOW_UP_PATTERNS = [
    r"điều\s+\d+",
    r"khoản\s+\d+",
    r"mục\s+\d+",
    r"ý này",
    r"đoạn này",
    r"văn bản đó",
    r"tài liệu đó",
    r"văn bản trên",
    r"kế hoạch trên",
    r"nội dung trên",
    r"quyết định trên",
    r"thông báo trên",
    r"báo cáo trên",
    r"công văn trên",
    r"văn bản (này|đó|trên|vừa rồi|vừa soạn)",
    r"(căn cứ|dựa).*(trên|vào).*(luật|văn bản|quy định)\s+nào",
    r"(nó|văn bản đó|cái (này|đó|trên))\s+(dựa|căn cứ|theo)",
]


def looks_like_follow_up(question: str) -> bool:
    q = (question or "").strip().lower()
    return any(re.search(p, q) for p in FOLLOW_UP_PATTERNS)


def is_in_document_domain(question: str) -> bool:
    """Heuristic domain check before retrieval to reduce irrelevant calls."""
    q = (question or "").strip().lower()
    if not q:
        return False

    if any(re.search(p, q) for p in OUT_OF_DOMAIN_CHAT_PATTERNS):
        return False

    if any(re.search(p, q) for p in DOMAIN_HINT_PATTERNS):
        return True

    # Không đủ tín hiệu domain thì mặc định cho qua để tránh false negative quá mức.
    return True
