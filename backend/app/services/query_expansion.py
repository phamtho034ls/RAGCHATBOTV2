"""Query expansion for multi-article retrieval.

Expands a single query into multiple sub-queries (max 3) to improve
recall for questions that span multiple legal articles.

Triggered when query contains aggregation indicators like:
"các", "những", "liệt kê", "danh sách", "cấm", "tất cả", etc.
"""

from __future__ import annotations

import logging
import re
from typing import List

log = logging.getLogger(__name__)

_EXPANSION_TRIGGERS = re.compile(
    r"(?:"
    r"\bcác\b|\bnhững\b|\btất cả\b"
    r"|\bliệt kê\b|\bdanh sách\b"
    r"|\bcấm\b|\bhạn chế\b|\bnghiêm cấm\b"
    r"|\bbao gồm\b|\bgồm những gì\b"
    r"|\bnhiều\b.*\b(điều|khoản|quy định)\b"
    r"|\bcác\s+(ngành|nghề|lĩnh vực|hoạt động|hành vi|trường hợp|đối tượng)\b"
    r"|\bđiều kiện\b"
    r"|\bđiều kiện\b.*\b(kinh doanh|hoạt động|thành lập|đăng ký)\b"
    r"|\bquy định\b.*\b(về|về việc|của)\b"
    r"|\b(kinh doanh|hoạt động)\b.*\b(điều kiện|quy định)\b"
    r"|\bvăn bản\b.*\b(quy định|liên quan)\b"
    r"|\b(luật|nghị định)\b.*\b(điều kiện|quy định)\b"
    r"|\bso sánh\b|\bđối chiếu\b|\bkhác nhau\b"
    r")",
    re.IGNORECASE,
)


def needs_expansion(query: str) -> bool:
    """Check if a query would benefit from multi-query expansion."""
    return bool(_EXPANSION_TRIGGERS.search(query or ""))


def should_expand_query_v2(query: str, initial_results: List[dict]) -> bool:
    """V3: quyết định mở rộng truy vấn — ML similarity + diversity + keyword (_EXPANSION_TRIGGERS).

    Dùng embedding model hiện có (embed_query / embed_texts), không load model mới.
    Expand nếu BẤT KỲ signal nào đúng.
    """
    q = query or ""

    if needs_expansion(q):
        return True

    top5 = initial_results[:5]
    doc_ids = {p.get("document_id") for p in top5 if p.get("document_id") is not None}
    if len(doc_ids) == 1 and len(top5) >= 2:
        log.debug("should_expand_query_v2: diversity signal (single doc in top-5)")
        return True

    if not initial_results:
        return False

    try:
        import numpy as np

        from app.pipeline.embedding import embed_query, embed_texts

        qv = np.asarray(embed_query(q), dtype=np.float32)
        qn = float(np.linalg.norm(qv)) + 1e-12
        texts = [(p.get("text_chunk") or "")[:2000] for p in initial_results[:8]]
        texts = [t for t in texts if t.strip()]
        if not texts:
            return False
        pv = embed_texts(texts)
        if pv.size == 0:
            return False
        pn = np.linalg.norm(pv, axis=1, keepdims=True) + 1e-12
        sims = np.dot(pv, qv) / (pn.flatten() * qn)
        max_sim = float(np.max(sims))
        if max_sim < 0.55:
            log.debug("should_expand_query_v2: low max_similarity=%.3f", max_sim)
            return True
    except Exception as exc:
        log.warning("should_expand_query_v2 embedding signal failed: %s", exc)

    return False


async def expand_query(query: str, max_variants: int = 3) -> List[str]:
    """Expand query into sub-queries for broader multi-article retrieval.

    Always includes the original query as the first element.
    Returns at most `max_variants` queries total (including original).
    """
    queries = [query]

    if not needs_expansion(query):
        return queries

    try:
        from app.services.llm_client import generate

        prompt = f"""Bạn là chuyên gia tìm kiếm văn bản pháp luật Việt Nam.

Câu hỏi gốc: "{query}"

Câu hỏi này CẦN tìm kiếm NHIỀU điều luật/quy định khác nhau.
Hãy tạo thêm {max_variants - 1} câu truy vấn phụ để tìm kiếm đầy đủ hơn.

YÊU CẦU:
- Mỗi câu truy vấn tập trung vào một khía cạnh khác nhau
- Giữ nguyên tên luật/nghị định nếu có
- Bổ sung từ khóa pháp lý cụ thể
- TUYỆT ĐỐI KHÔNG thêm số hiệu văn bản (dạng xx/yyyy/NĐ-CP, TT-BTC, QĐ-UBND, …) nếu câu gốc không có số hiệu đó — không được bịa số hiệu
- Mỗi câu một dòng, KHÔNG đánh số, KHÔNG giải thích

Các câu truy vấn phụ:"""

        result = await generate(prompt, temperature=0.1)
        lines = [
            line.strip().lstrip("0123456789.-) ")
            for line in result.strip().split("\n")
            if line.strip() and len(line.strip()) > 8
        ]

        for line in lines[:max_variants - 1]:
            if line not in queries:
                queries.append(line)

        log.info("Query expansion: '%s' → %d variants", query[:60], len(queries))

    except Exception as e:
        log.warning("Query expansion failed: %s", e)

    return queries[:max_variants]
