"""Heuristic signals that a query is in-scope for legal/admin RAG (reduce false OOS)."""

from __future__ import annotations

import re
from typing import Pattern

# Tín hiệu mạnh: tra cứu văn bản, điều khoản, cơ quan, lĩnh vực hành chính–pháp luật VN.
_STRONG_PATTERNS: tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:nghị\s*định|thông\s*tư|luật|quyết\s*định|chỉ\s*thị|thông\s*báo|công\s*văn)\s*(?:số\s*)?\d+",
        r"\b\d+\s*/\s*\d{4}\s*/\s*[A-ZĐa-zđ0-9\-]+",
        r"\bđiều\s+\d+",
        r"\bkhoản\s+\d+",
        r"\b(?:UBND|QH\d+|bộ\s+vhttdl|bộ\s+văn\s+hóa)\b",
        r"\b(?:thủ\s*tục\s*hành\s*chính|công\s*chức|viên\s*chức|cán\s*bộ\s+xã)\b",
        r"\b(?:thư\s*viện|nhà\s*văn\s*hóa|thiết\s*chế\s*văn\s*hóa|di\s*sản\s*văn\s*hóa)\b",
        r"\bthư\s*viện\s+công\s+lập\b",
        r"\b(?:pháp\s*luật|quy\s*định\s*pháp|hiến\s*pháp|bộ\s*luật)\b",
        r"\b(?:quyền\s*và\s*nghĩa\s*vụ|trách\s*nhiệm|thẩm\s*quyền|mức\s*phạt)\b",
        r"\b(?:ưu\s*tiên\s*đầu\s*tư|chính\s*sách\s*của\s*nhà\s*nước|vai\s*trò\s*của)\b",
        r"\bđề\s*xuất\s+\d*\s*giải\s*pháp\b",
        r"\b(?:đề\s*xuất\s+giải\s*pháp|đổi\s*mới\s+phương\s*thức|hoạt\s*động\s+của)\b",
    )
)

# Ngữ cảnh “du lịch / văn hóa” gắn pháp luật — không chặn như chat du lịch thuần.
_LEGAL_CONTEXT_NEAR_TRAVEL: tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:nghị\s*định|thông\s*tư|luật|quy\s*định|văn\s*bản|điều\s+\d+)\b",
        r"\b(?:phân\s*cấp|quản\s*lý\s*văn\s*hóa|thể\s*thao|du\s*lịch)\b.*\b(?:nghị\s*định|quy\s*định)\b",
        r"\b(?:nghị\s*định|quy\s*định)\b.*\b(?:văn\s*hóa|thể\s*thao|du\s*lịch)\b",
    )
)


def query_has_strong_legal_scope_signals(text: str) -> bool:
    """True nếu câu gần chắc thuộc phạm vi tra cứu / hành chính / chính sách gắn văn bản."""
    q = (text or "").strip()
    if len(q) < 4:
        return False
    return any(p.search(q) for p in _STRONG_PATTERNS)


def travel_term_is_likely_legal_context(text: str) -> bool:
    """'Du lịch' trong cụm pháp luật (Luật Du lịch, NĐ về du lịch…) — không coi là chat ngoài domain."""
    q = (text or "").strip().lower()
    if "du lịch" not in q and "dulịch" not in q.replace(" ", ""):
        return False
    return any(p.search(text or "") for p in _LEGAL_CONTEXT_NEAR_TRAVEL)


def should_block_out_of_domain_chat_pattern(text: str, pattern: str) -> bool:
    """Chặn pattern OUT_OF_DOMAIN chỉ khi không phải ngữ cảnh pháp luật (vd. du lịch + Luật)."""
    if pattern == r"du lịch" and travel_term_is_likely_legal_context(text):
        return False
    return True
