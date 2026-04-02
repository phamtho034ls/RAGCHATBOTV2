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

_FINE_QUERY_MARKERS = (
    "mức phạt",
    "mức xử phạt",
    "phạt bao nhiêu",
    "bị phạt bao nhiêu",
    "bị phạt như thế nào",
    "bị phạt thế nào",
    "tiền phạt",
    "mức tiền phạt",
    "hình thức xử phạt",
)

_THAM_QUYEN_MARKERS = ("thẩm quyền",)
_AUTHORITY_CONTEXT_MARKERS = (
    "thẩm quyền xử phạt",
    "có quyền xử phạt",
    "được quyền phạt tiền",
    "có quyền phạt tiền",
    "có quyền cảnh cáo",
    "có quyền tịch thu",
)

_LEGAL_SYNTHESIS_MARKERS = (
    "xử phạt vi phạm hành chính",
    "vi phạm hành chính",
    "không lập biên bản",
    "lập biên bản",
    "cưỡng chế",
    "quyết định xử phạt",
    "hình thức xử phạt",
    "theo luật",
    "theo nghị định",
    "quy định của pháp luật",
    "bao nhiêu",
    "là gì",
    "ở đâu",
    "như thế nào",
    "quy định thế nào",
    "văn bản nào",
    "theo văn bản nào",
    "gồm những yêu cầu",
    "gồm những nội dung",
    "trách nhiệm gồm những",
    "được quy định ở đâu",
    "tiếp nhận",
    "tin báo",
    "can thiệp",
)

_PROCEDURAL_MARKERS = (
    "thủ tục",
    "quy trình",
    "hồ sơ",
    "thành phần hồ sơ",
    "giấy phép",
    "cấp phép",
    "đăng ký",
    "xin phép",
    "điều kiện",
)

_CONDITION_GATE_MARKERS = ("điều kiện", "đủ điều kiện", "yêu cầu")
_CONDITION_ACTIVITY_MARKERS = (
    "đăng ký",
    "thành lập",
    "cấp phép",
    "hoạt động",
    "giấy phép",
    "thủ tục",
    "xin phép",
    "mở cơ sở",
    "trợ giúp xã hội",
    "chứng nhận đủ điều kiện",
    "kinh doanh có điều kiện",
)
_CONDITION_RESOURCE_LEFT_MARKERS = ("cơ sở vật chất", "nhân lực", "trang thiết bị")
_CONDITION_RESOURCE_RIGHT_MARKERS = ("yêu cầu", "điều kiện", "thế nào")

_PROHIBITED_ACTS_MARKERS = (
    "hành vi bị cấm",
    "các hành vi cấm",
    "trích xuất",
    "liệt kê",
)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    t = (text or "").lower()
    return any(m in t for m in markers)


def query_asks_fine_amount(query: str) -> bool:
    return _contains_any(query, _FINE_QUERY_MARKERS)


def title_contains_tham_quyen(title: str) -> bool:
    return _contains_any(title, _THAM_QUYEN_MARKERS)


def query_contains_tham_quyen(query: str) -> bool:
    return _contains_any(query, _THAM_QUYEN_MARKERS)


def context_describes_authority(context: str) -> bool:
    return _contains_any(context, _AUTHORITY_CONTEXT_MARKERS)


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
    if _contains_any(q, _LEGAL_SYNTHESIS_MARKERS):
        return True
    if "căn cứ pháp lý" in q and ("văn bản" in q or "nào" in q):
        return True
    if "gồm những" in q and ("yêu cầu" in q or "gì" in q or "nội dung" in q):
        return True
    if "điều kiện" in q and "là gì" in q:
        return True
    if ("thủ tục" in q or "quy trình" in q) and (
        "ở đâu" in q or "như thế nào" in q or "thế nào" in q
    ):
        return True
    return False


def query_looks_procedural(query: str) -> bool:
    """Detect procedural/admin queries that need broader retrieval recall."""
    q = (query or "").lower().strip()
    if not q:
        return False
    return _contains_any(q, _PROCEDURAL_MARKERS)


def query_asks_comprehensive_statutory_coverage(query: str) -> bool:
    """Chính sách / tiêu chí / phân loại — cần quét nhiều điều trong cùng luật hoặc văn bản hướng dẫn."""
    q = (query or "").lower().strip()
    if len(q) < 12:
        return False
    if "chính sách" in q and _contains_any(q, ("nhà nước", "quốc gia", "đối với", "của nước")):
        return True
    if ("tiêu chí" in q or "tiêu chi" in q) and _contains_any(
        q, ("phân loại", "dự án", "trọng điểm", "quốc gia")
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
    if not _contains_any(q, _CONDITION_GATE_MARKERS):
        return False
    if _contains_any(q, _CONDITION_ACTIVITY_MARKERS):
        return True
    if "điều kiện" in q and _contains_any(q, ("là gì", "gì", "như thế nào", "ra sao", "bao gồm")):
        return True
    if _contains_any(q, _CONDITION_RESOURCE_LEFT_MARKERS) and _contains_any(
        q, _CONDITION_RESOURCE_RIGHT_MARKERS
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
    if "hành vi" in q and "cấm" in q:
        return True
    return _contains_any(q, _PROHIBITED_ACTS_MARKERS)


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
    return _contains_any(
        query_lower,
        ("thẩm quyền quyết định", "quyền quyết định", "được quyết định"),
    )


def document_type_luat_is_false_positive(query_lower: str) -> bool:
    return _contains_any(
        query_lower,
        ("điều luật", "các điều luật", "theo điều luật"),
    )


def detect_sort_from_patterns(query_lower: str, sort_patterns: Dict[str, List[str]]) -> Optional[str]:
    for sort_type, patterns in sort_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return sort_type
    return None
