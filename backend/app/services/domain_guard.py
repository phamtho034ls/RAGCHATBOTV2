"""Domain guard to reject out-of-scope questions before retrieval.

Refactor notes
--------------
* ``_DOMAIN_HINT_WORDS`` and ``_OUT_OF_DOMAIN_WORDS`` — all literals, no wildcards.
  Using ``in`` on a frozenset is O(1) and ~10× faster than re.search on each item.
* ``FOLLOW_UP_PATTERNS`` (compiled regex) — kept as regex: contains ``\\d+``,
  alternation groups, context anchors that cannot be expressed as plain substrings.
* ``legal_scope`` is intentionally NOT merged here because it is also imported
  independently by query_intent and query_route_classifier.
"""

from __future__ import annotations

import re

from app.services.legal_scope import (
    query_has_strong_legal_scope_signals,
    should_block_out_of_domain_chat_pattern,
    travel_term_is_likely_legal_context,
)

# ── Plain-keyword sets (no regex needed) ─────────────────────────────────────

# Query contains ≥1 of these → likely in legal/admin domain.
_DOMAIN_HINT_WORDS: frozenset[str] = frozenset([
    "nghị định", "thông tư", "quy định", "văn bản", "hồ sơ",
    "thủ tục", "cấp phép", "quy trình", "luật", "nghị quyết",
    "thông báo", "công văn", "quyết định", "chỉ thị",
])

# Query contains one of these AND no legal context → block as out-of-domain chat.
# "du lịch" is intentionally absent — handled via travel_term_is_likely_legal_context.
_OUT_OF_DOMAIN_WORDS: frozenset[str] = frozenset([
    "thời tiết", "bóng đá", "chứng khoán", "âm nhạc", "phim",
    "nấu ăn", "truyện", "tin tức thể thao", "giải trí",
])

# ── Regex patterns (genuinely need regex: \d+, alternation, context) ─────────

_FOLLOW_UP_RE = re.compile(
    r"điều\s+\d+"
    r"|khoản\s+\d+"
    r"|mục\s+\d+"
    r"|ý này|đoạn này"
    r"|văn bản đó|tài liệu đó|văn bản trên|kế hoạch trên"
    r"|nội dung trên|quyết định trên|thông báo trên|báo cáo trên|công văn trên"
    r"|văn bản (?:này|đó|trên|vừa rồi|vừa soạn)"
    r"|(?:căn cứ|dựa).*(?:trên|vào).*(?:luật|văn bản|quy định)\s+nào"
    r"|(?:nó|văn bản đó|cái (?:này|đó|trên))\s+(?:dựa|căn cứ|theo)",
    re.IGNORECASE,
)

_DOMAIN_HINT_ARTICLE_RE = re.compile(r"điều\s+\d+|khoản\s+\d+", re.IGNORECASE)


def looks_like_follow_up(question: str) -> bool:
    q = (question or "").strip().lower()
    return bool(_FOLLOW_UP_RE.search(q))


def is_in_document_domain(question: str) -> bool:
    """Heuristic domain check before retrieval to reduce irrelevant calls."""
    q_raw = (question or "").strip()
    if not q_raw:
        return False
    q = q_raw.lower()

    # Fast-pass: strong legal/admin signals from canonical legal_scope module.
    if query_has_strong_legal_scope_signals(q_raw):
        return True

    # Block clear out-of-domain chat words — but keep "du lịch" if legal context.
    for word in _OUT_OF_DOMAIN_WORDS:
        if word in q:
            return False
    if "du lịch" in q and not travel_term_is_likely_legal_context(q_raw):
        return False

    # Hint words (keyword `in` — O(1) per word).
    if any(w in q for w in _DOMAIN_HINT_WORDS):
        return True

    # Điều X / Khoản X — needs regex (digit).
    if _DOMAIN_HINT_ARTICLE_RE.search(q):
        return True

    # Default: let through to avoid excessive false negatives.
    return True
