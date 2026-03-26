"""Query rewriting for Vietnamese Legal RAG.

Rewrites informal/colloquial user queries into clear, formal, legal-aware queries
that improve retrieval quality without changing the original meaning.

Design principles:
- LLM does the rewriting; original query is preserved for logging.
- Rewriting is skipped for queries that are already precise (contain Điều/Khoản/doc numbers).
- Falls back to original query on any LLM error or suspicious output.
- Vietnamese output only; no answering, only rewriting.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import OPENAI_API_KEY

log = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = (
    "Bạn là chuyên gia tối ưu truy vấn tìm kiếm văn bản pháp luật Việt Nam. "
    "Nhiệm vụ DUY NHẤT là VIẾT LẠI câu hỏi — KHÔNG trả lời, KHÔNG giải thích. "
    "Đầu ra: CHỈ MỘT câu hỏi đã viết lại, không có gì thêm."
)

_REWRITE_USER_TEMPLATE = """\
Viết lại câu hỏi sau để tối ưu cho hệ thống tìm kiếm pháp luật Việt Nam.

Yêu cầu:
- Giữ nguyên ý nghĩa gốc
- Chuyển sang văn phong trang trọng, pháp lý
- Mở rộng viết tắt (vd: NĐ → Nghị định, TT → Thông tư, QĐ → Quyết định)
- Bổ sung ngữ cảnh pháp lý rõ ràng nếu hiển nhiên
- Không trả lời, không giải thích — chỉ viết lại câu hỏi
- Đầu ra bằng tiếng Việt, CHỈ MỘT câu

Ví dụ:
  Đầu vào: "xây nhà trái phép bị sao"
  Đầu ra: "Các quy định xử phạt hành vi xây dựng nhà trái phép theo pháp luật Việt Nam"

  Đầu vào: "NĐ 100 phạt bao nhiêu"
  Đầu ra: "Mức phạt vi phạm hành chính theo Nghị định 100 là bao nhiêu"

  Đầu vào: "karaoke hoạt động đến mấy giờ"
  Đầu ra: "Quy định về giờ hoạt động của cơ sở kinh doanh karaoke theo pháp luật Việt Nam"

Câu hỏi cần viết lại:
{query}

Câu hỏi đã viết lại:\
"""

# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_QUERY_LEN = 8
_MAX_QUERY_LEN = 400

# Patterns that indicate the query is already precise enough
_ALREADY_PRECISE_PATTERNS = (
    re.compile(r"điều\s+\d+", re.IGNORECASE),
    re.compile(r"khoản\s+\d+", re.IGNORECASE),
    re.compile(r"\d{2,}/\d{4}/[A-ZĐa-zđ0-9\-]+"),        # doc number e.g. 100/2019/NĐ-CP
    re.compile(r"nghị định số\s+\d+", re.IGNORECASE),
    re.compile(r"thông tư số\s+\d+", re.IGNORECASE),
    re.compile(r"quyết định số\s+\d+", re.IGNORECASE),
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _should_rewrite(query: str) -> bool:
    """Return True when the query would benefit from rewriting.

    Skips very short/long queries and queries that already contain precise
    legal references (article numbers, document numbers).
    """
    q = (query or "").strip()
    length = len(q)
    if length < _MIN_QUERY_LEN or length > _MAX_QUERY_LEN:
        return False
    for pat in _ALREADY_PRECISE_PATTERNS:
        if pat.search(q):
            return False
    return True


def _is_safe_rewrite(original: str, rewritten: str) -> bool:
    """Guard against LLM hallucinating an answer instead of a rewrite."""
    if not rewritten:
        return False
    # Too long relative to original → likely answered the question
    if len(rewritten) > len(original) * 4:
        return False
    # Multi-paragraph output → LLM added an answer
    if "\n\n" in rewritten:
        return False
    # Rewrite should not end with a colon (means truncation) or be longer than 300 chars
    if rewritten.endswith(":") or len(rewritten) > 300:
        return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────

async def rewrite_query(query: str) -> str:
    """Rewrite a user query into a clearer, formal, legal-aware form for retrieval.

    Uses LLM to:
    - Expand abbreviations (NĐ → Nghị định, etc.)
    - Add obvious legal context
    - Convert colloquial phrasing to formal legal language

    Falls back gracefully to the original query on any error or when the
    rewrite does not pass safety checks.

    Args:
        query: Raw user query (may be informal / colloquial).

    Returns:
        Rewritten query optimised for legal document retrieval.
        Returns the original query if rewriting is skipped or fails.
    """
    if not OPENAI_API_KEY or not _should_rewrite(query):
        return query

    try:
        from app.services.llm_client import generate

        prompt = _REWRITE_USER_TEMPLATE.format(query=query.strip())
        rewritten: str = await generate(
            prompt=prompt,
            system=_REWRITE_SYSTEM,
            temperature=0.0,
            max_tokens=160,
        )
        rewritten = (rewritten or "").strip()

        if not _is_safe_rewrite(query, rewritten):
            log.debug(
                "Query rewrite rejected (unsafe output). Original: '%s'", query[:60]
            )
            return query

        log.info(
            "Query rewritten | original='%s' → rewritten='%s'",
            query[:80],
            rewritten[:100],
        )
        return rewritten

    except Exception as exc:
        log.warning("Query rewriting failed (using original query): %s", exc)
        return query
