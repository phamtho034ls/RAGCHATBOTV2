"""Tập trung toàn bộ biểu thức chính quy và khớp mẫu dùng cho RAG / phân tích câu hỏi.

``rag_chain_v2`` và ``query_understanding`` không import ``re`` — logic nằm ở đây.

Lưu ý: **không có regex** ở hai module trên ≠ bỏ khớp mẫu trong hệ thống; chỉ gom một chỗ.
Để thay thế hoàn toàn regex: classifier (ML/LLM) + từ khóa ``in`` đơn giản — đắt hơn hoặc kém chính xác.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.config import NO_INFO_MESSAGE

# ── Điều luật / số điều ───────────────────────────────────────────────


def extract_article_reference_from_text(text: str) -> Optional[str]:
    m = re.search(r"điều\s+(\d+[a-zA-Z]?)", text or "", re.IGNORECASE)
    return m.group(1) if m else None


def normalize_article_number_canonical(article_number: Optional[str]) -> Optional[str]:
    if not article_number:
        return None
    m = re.search(r"(\d+[a-zA-Z]?)", str(article_number))
    return m.group(1) if m else str(article_number).strip()


_ARTICLE_MENTION_RE = re.compile(r"điều\s*(\d+[a-zA-Z]?)", re.IGNORECASE)


def extract_article_numbers_mentioned_in_answer(answer: str) -> set[str]:
    """Các số Điều xuất hiện trong phần trả lời — dùng để siết khối 'Căn cứ pháp lý'."""
    out: set[str] = set()
    for m in _ARTICLE_MENTION_RE.finditer(answer or ""):
        cn = normalize_article_number_canonical(m.group(1))
        if cn:
            out.add(cn)
    return out


def query_demands_specific_article(query: str) -> bool:
    q = (query or "").lower()
    return bool(re.search(r"\bđiều\s+\d+[a-zA-Z]?\b", q))


# ── Mức phạt / thẩm quyền ─────────────────────────────────────────────

_MUC_PHAT_QUERY_RE = re.compile(
    r"mức\s+phạt|mức\s+xử\s+phạt|phạt\s+bao\s+nhiêu"
    r"|bị\s+phạt\s*(bao\s+nhiêu|như\s+thế\s+nào|thế\s+nào)"
    r"|tiền\s+phạt|mức\s+tiền\s+phạt|hình\s+thức\s+xử\s+phạt",
    re.IGNORECASE,
)

_THAM_QUYEN_TITLE_RE = re.compile(r"thẩm\s+quyền", re.IGNORECASE)

_THAM_QUYEN_CONTEXT_RE = re.compile(
    r"thẩm\s+quyền\s+xử\s+phạt"
    r"|có\s+quyền\s*[:\-]?\s*(?:[a-z]\)\s*(?:phạt|cảnh\s+cáo)|xử\s+phạt)"
    r"|được\s+quyền\s+phạt\s+tiền"
    r"|có\s+quyền\s+(?:phạt\s+tiền|cảnh\s+cáo|tịch\s+thu)",
    re.IGNORECASE,
)


def query_asks_fine_amount(query: str) -> bool:
    return bool(_MUC_PHAT_QUERY_RE.search(query or ""))


def title_contains_tham_quyen(title: str) -> bool:
    return bool(_THAM_QUYEN_TITLE_RE.search(title or ""))


def query_contains_tham_quyen(query: str) -> bool:
    return bool(_THAM_QUYEN_TITLE_RE.search(query or ""))


def context_describes_authority(context: str) -> bool:
    return bool(_THAM_QUYEN_CONTEXT_RE.search(context or ""))


def query_expects_llm_synthesis_from_context(query: str) -> bool:
    q = (query or "").lower().strip()
    if not q:
        return False
    if query_asks_fine_amount(query):
        return True
    if query_demands_specific_article(query):
        return True
    if query_contains_tham_quyen(query):
        return True
    if re.search(
        r"xử\s+phạt\s+vi\s+phạm\s+hành\s+chính|vi\s+phạm\s+hành\s+chính",
        q,
    ):
        return True
    if re.search(
        r"không\s+lập\s+biên\s+bản|lập\s+biên\s+bản|cưỡng\s+chế|"
        r"quyết\s+định\s+xử\s+phạt|hình\s+thức\s+xử\s+phạt",
        q,
    ):
        return True
    if re.search(
        r"luật\s+.+?\s+\d{4}|nghị\s+định\s+\d+/\d{4}|"
        r"theo\s+luật|theo\s+nghị\s+định|quy\s+định\s+của\s+pháp\s+luật",
        q,
    ):
        return True
    if re.search(
        r"\b(bao\s+nhiêu|là\s+gì\b|ở\s+đâu\b|như\s+thế\s+nào\b|"
        r"quy\s+định\s+thế\s+nào\b|văn\s+bản\s+nào\b|theo\s+văn\s+bản\s+nào\b)\b",
        q,
    ):
        return True
    if "căn cứ pháp lý" in q and ("văn bản" in q or "nào" in q):
        return True
    if re.search(r"gồm\s+những\s+(yêu\s+cầu|gì|nội\s+dung)\b", q):
        return True
    if re.search(r"trách\s+nhiệm.*gồm\s+những", q):
        return True
    if re.search(r"được\s+quy\s+định\s+ở\s+đâu", q):
        return True
    if re.search(r"\bđiều\s+kiện\b.+\blà\s+gì\b", q):
        return True
    if re.search(
        r"\b(thủ\s+tục|quy\s+trình)\b.+\b(ở\s+đâu|như\s+thế\s+nào|thế\s+nào)\b",
        q,
    ):
        return True
    if re.search(r"tiếp\s+nhận.*tin\s+báo|can\s+thiệp", q):
        return True
    return False


def query_looks_procedural(query: str) -> bool:
    """Detect procedural/admin queries that need broader retrieval recall."""
    q = (query or "").lower().strip()
    if not q:
        return False
    return bool(
        re.search(
            r"\b(thủ\s*tục|quy\s*trình|hồ\s*sơ|thành\s*phần\s*hồ\s*sơ|"
            r"giấy\s*phép|cấp\s*phép|đăng\s*ký|xin\s*phép|điều\s*kiện)\b",
            q,
        )
    )


def query_asks_comprehensive_statutory_coverage(query: str) -> bool:
    """Chính sách / tiêu chí / phân loại — cần quét nhiều điều trong cùng luật hoặc văn bản hướng dẫn."""
    q = (query or "").lower().strip()
    if len(q) < 12:
        return False
    if re.search(r"chính\s+sách", q) and re.search(
        r"(nhà\s+nước|quốc\s+gia|đối\s+với|của\s+nước)", q
    ):
        return True
    if re.search(r"tiêu\s+ch[íi]", q) and re.search(
        r"(phân\s+loại|dự\s+án|trọng\s+điểm|quốc\s+gia)", q
    ):
        return True
    if "trọng điểm quốc gia" in q or "dự án trọng điểm quốc gia" in q:
        return True
    if "quyền và nghĩa vụ" in q or "nghĩa vụ và quyền" in q:
        return True
    return False


def query_asks_structured_registration_conditions(query: str) -> bool:
    """Câu hỏi về điều kiện đăng ký/thành lập/cấp phép/hoạt động — cần tách CSVC, hoạt động, nhân lực."""
    q = (query or "").lower().strip()
    if not q:
        return False
    if not re.search(r"\bđiều\s+kiện\b|\bđủ\s+điều\s+kiện\b|\byêu\s+cầu\b", q):
        return False
    if re.search(
        r"đăng\s+ký|thành\s+lập|cấp\s+phép|hoạt\s+động|giấy\s+phép|"
        r"thủ\s+tục|xin\s+phép|mở\s+cơ\s+sở|trợ\s+giúp\s+xã\s+hội|"
        r"chứng\s+nhận\s+đủ\s+điều\s+kiện|kinh\s+doanh\s+có\s+điều\s+kiện",
        q,
    ):
        return True
    if re.search(
        r"điều\s+kiện\s+(là\s+gì|gì|như\s+thế\s+nào|ra\s+sao|bao\s+gồm)",
        q,
    ):
        return True
    if re.search(
        r"(cơ\s+sở\s+vật\s+chất|nhân\s+lực|trang\s+thiết\s+bị).{0,40}(yêu\s+cầu|điều\s+kiện|thế\s+nào)",
        q,
    ):
        return True
    return False


def query_requests_prohibited_acts_list(query: str) -> bool:
    """Detect extraction/list queries about prohibited acts (requires multi-article)."""
    q = (query or "").lower().strip()
    if not q:
        return False
    if "nghiêm cấm" in q:
        return True
    return bool(
        re.search(
            r"(hành\s*vi\s*bị\s*cấm|các\s*hành\s*vi\s*cấm|"
            r"trích\s*xuất.*hành\s*vi|liệt\s*kê.*hành\s*vi.*cấm)",
            q,
        )
    )


# ── Làm sạch output LLM ────────────────────────────────────────────────

_METADATA_JSON_PATTERN = re.compile(
    r'\{[^{}]*"(?:sources|confidence_score|document_id|vector_score|similarity|embedding)'
    r'[^{}]*\}',
    re.DOTALL,
)
_METADATA_FIELD_PATTERN = re.compile(
    r'"(?:sources|confidence_score|document_id|score|vector_score|similarity|embedding(?:_metadata)?)"'
    r'\s*:\s*(?:\[.*?\]|"[^"]*"|\d+(?:\.\d+)?|null|true|false)',
    re.DOTALL,
)


def fix_common_glued_doc_number_in_text(text: str) -> str:
    """Một số lỗi ghép số hiệu do model/DB: 1472024NĐ-CP → 147/2024/NĐ-CP."""
    if not text:
        return text
    return re.sub(
        r"\b(\d{2,4})(\d{4})((?:N[ĐD][-_]?CP)|(?:Q[ĐD][-_]?UBND)|(?:QH\d+))\b",
        r"\1/\2/\3",
        text,
        flags=re.IGNORECASE,
    )


def sanitize_rag_llm_output(text: str) -> str:
    if not text:
        return text
    cleaned = _METADATA_JSON_PATTERN.sub("", text)
    cleaned = _METADATA_FIELD_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\{\s*,?\s*\}", "", cleaned)
    cleaned = re.sub(r"\[\s*,?\s*\]", "", cleaned)
    cleaned = re.sub(r"```(?:\w+)?\n", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = fix_common_glued_doc_number_in_text(cleaned.strip())
    return cleaned.strip()


# ── Số hiệu văn bản / chống ảo giác ───────────────────────────────────

_DOC_NUMBER_RE = re.compile(r"\b(\d+[/_]\d{4}[/_][A-ZĐa-zđ\-]+)\b")


def extract_doc_numbers_from_text(text: str) -> set[str]:
    return {m.group(1).replace("_", "/") for m in _DOC_NUMBER_RE.finditer(text or "")}


def normalize_doc_number_for_compare(doc_num: str) -> str:
    return doc_num.replace("_", "/").replace("Đ", "D").replace("đ", "d").lower()


def strip_answer_lines_with_hallucinated_doc_numbers(
    answer: str,
    context_doc_numbers: set[str],
    *,
    no_info_message: str = NO_INFO_MESSAGE,
) -> str:
    if not context_doc_numbers or not answer:
        return answer
    context_normalized = {normalize_doc_number_for_compare(n) for n in context_doc_numbers}
    answer_doc_numbers = extract_doc_numbers_from_text(answer)
    hallucinated = {
        n for n in answer_doc_numbers
        if normalize_doc_number_for_compare(n) not in context_normalized
    }
    if not hallucinated:
        return answer
    cleaned = answer
    for bad_num in hallucinated:
        cleaned = re.sub(
            rf"[^\n]*{re.escape(bad_num)}[^\n]*\n?",
            "",
            cleaned,
        )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or no_info_message


def answer_contains_explicit_doc_number(answer: str) -> bool:
    return bool(re.search(r"\b\d+/\d{4}/[A-ZĐa-zđ0-9\-]+\b", answer or ""))


# ── Trích dẫn / tiêu đề ───────────────────────────────────────────────


def article_sort_key_tuple(article: str) -> tuple:
    """Sắp xếp Điều 2 trước Điều 10 — không dùng regex."""
    s = str(article).strip().lower()
    num_str = ""
    i = 0
    while i < len(s) and s[i].isdigit():
        num_str += s[i]
        i += 1
    suffix = s[i:]
    if not num_str:
        return (99999, s)
    return (int(num_str), suffix)


def shorten_title_long_parenthetical(title: str, *, min_inner_len: int = 100) -> str:
    """Rút gọn ngoặc đơn quá dài trong tiêu đề (giữ nguyên ngoặc ngắn)."""
    return re.sub(rf"\([^)]{{{min_inner_len},}}\)", "(…)", title)


# ── query_understanding helpers ────────────────────────────────────────


def match_first_mapping_value(text: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    for value, patterns in mapping.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return value
    return None


def extract_year_from_query_text(text: str) -> Optional[int]:
    match = re.search(r"năm\s+(20\d{2})", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(20\d{2})\b", text)
    if match:
        return int(match.group(1))
    return None


def extract_article_number_from_user_query(query: str) -> Optional[str]:
    match = re.search(r"điều\s+(\d+[A-Za-z]?)", query, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def tokenize_query_words_alnum(query_lower: str) -> List[str]:
    return re.findall(r"[\w]+", query_lower)


def document_type_quyet_dinh_is_false_positive(query_lower: str) -> bool:
    return bool(
        re.search(
            r"thẩm quyền\s+quyết định|quyền\s+quyết định|được\s+quyết định",
            query_lower,
        )
    )


def document_type_luat_is_false_positive(query_lower: str) -> bool:
    return bool(
        re.search(
            r"điều\s+luật|các\s+điều\s+luật|theo\s+điều\s+luật",
            query_lower,
        )
    )


def detect_sort_from_patterns(query_lower: str, sort_patterns: Dict[str, List[str]]) -> Optional[str]:
    for sort_type, patterns in sort_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return sort_type
    return None
