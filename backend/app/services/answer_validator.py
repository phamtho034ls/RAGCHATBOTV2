"""Validate groundedness of generated answers against retrieved context.

Two validation layers:
  1. Embedding-based grounding check (``validate_answer_grounding``) — fast, no LLM call.
  2. LLM-based semantic validation  (``validate_answer``)            — slower, deeper check.

The LLM validator catches:
  - Fabricated legal references not present in the retrieved context.
  - Answers that contradict the context.
  - Missing law citations when the context contains them.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import ANSWER_VALIDATION_THRESHOLD, OPENAI_API_KEY

log = logging.getLogger(__name__)

try:
    from app.pipeline.embedding import embed_query, embed_texts
except Exception:  # pragma: no cover – embedding may not be available in tests
    embed_query = None  # type: ignore[assignment]
    embed_texts = None  # type: ignore[assignment]


def _normalize_answer(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _has_legal_keywords(text: str) -> bool:
    """Check if text contains legal terminology (lower threshold for legal answers)."""
    patterns = [
        r"điều\s+\d+", r"khoản\s+\d+", r"luật\s+\w+",
        r"nghị định", r"thông tư", r"quyết định",
        r"quy định", r"theo\s+.{3,30}(luật|nghị định|thông tư)",
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _extract_article_references(text: str) -> List[str]:
    """Extract article numbers referenced in the text (e.g. 'Điều 7' → ['7'])."""
    return re.findall(r"(?:điều|article)\s+(\d+[a-zA-Z]?)", text, re.IGNORECASE)


def _count_clauses_in_answer(answer: str, article_num: str) -> int:
    """Count how many clause numbers appear in the answer for a given article.

    Looks for patterns like:
    - '1. ...' (numbered clauses)
    - 'Khoản 1' references
    """
    # Find the section of the answer about this article
    article_pattern = re.compile(
        rf"Điều\s+{re.escape(article_num)}[.\s]",
        re.IGNORECASE,
    )
    match = article_pattern.search(answer)
    if not match:
        return 0

    # Get text from article header to next article or end
    start = match.start()
    next_article = re.search(r"\nĐiều\s+\d+[.\s]", answer[start + 1:], re.IGNORECASE)
    end = start + 1 + next_article.start() if next_article else len(answer)
    article_section = answer[start:end]

    # Count numbered clauses: lines starting with a number followed by a period
    clause_numbers = set()
    for m in re.finditer(r"(?:^|\n)\s*(\d+)\.\s", article_section):
        clause_numbers.add(m.group(1))

    return len(clause_numbers)


def validate_article_completeness(
    answer: str,
    context_docs: List[dict],
) -> Dict:
    """Check if the answer includes all clauses for each referenced article.

    Compares clause count in the answer against clauses found in context.
    Returns validation result with details per article.
    """
    article_refs = _extract_article_references(answer)
    if not article_refs:
        return {"is_complete": True, "articles": {}, "incomplete_articles": []}

    # Count clauses available in context per article
    context_clause_counts: Dict[str, int] = {}
    for doc in context_docs:
        text = doc.get("text_chunk", doc.get("text", ""))
        for art_num in article_refs:
            # Count clauses in context for this article
            art_pattern = re.compile(
                rf"Điều\s+{re.escape(art_num)}[.\s]",
                re.IGNORECASE,
            )
            if art_pattern.search(text):
                # Count clause-numbered lines in this chunk
                clause_matches = set(re.findall(r"(?:^|\n)\s*(\d+)\.\s", text))
                context_clause_counts[art_num] = max(
                    context_clause_counts.get(art_num, 0),
                    len(clause_matches),
                )

    # Compare answer clause count vs context clause count
    articles_info = {}
    incomplete = []
    for art_num in set(article_refs):
        answer_count = _count_clauses_in_answer(answer, art_num)
        context_count = context_clause_counts.get(art_num, 0)
        is_complete = answer_count >= context_count if context_count > 0 else True
        articles_info[art_num] = {
            "answer_clauses": answer_count,
            "context_clauses": context_count,
            "is_complete": is_complete,
        }
        if not is_complete:
            incomplete.append(art_num)

    return {
        "is_complete": len(incomplete) == 0,
        "articles": articles_info,
        "incomplete_articles": incomplete,
    }


def validate_answer_grounding(
    answer: str,
    context_docs: List[dict],
    threshold: float = ANSWER_VALIDATION_THRESHOLD,
) -> Dict:
    """Check whether answer semantically aligns with retrieved chunks.

    Nếu answer có chứa tham chiếu pháp luật (điều luật, tên văn bản),
    giảm threshold xuống để tránh reject nhầm câu trả lời hợp lệ.

    Also validates article completeness when articles are referenced.
    """
    normalized_answer = _normalize_answer(answer)
    if not normalized_answer:
        return {"is_grounded": False, "similarity": 0.0, "threshold": threshold}

    contexts = [doc.get("text", "").strip() for doc in context_docs if doc.get("text")]
    if not contexts:
        return {"is_grounded": False, "similarity": 0.0, "threshold": threshold}

    # Nếu answer chứa legal keywords → giảm threshold (tránh reject nhầm)
    effective_threshold = threshold
    if _has_legal_keywords(normalized_answer):
        effective_threshold = min(threshold, 0.15)

    ans_vec = embed_query(normalized_answer)
    ctx_vecs = embed_texts(contexts)

    ans_norm = ans_vec / (np.linalg.norm(ans_vec, axis=1, keepdims=True) + 1e-12)
    ctx_norm = ctx_vecs / (np.linalg.norm(ctx_vecs, axis=1, keepdims=True) + 1e-12)

    sims = np.dot(ctx_norm, ans_norm[0])
    best_sim = float(np.max(sims)) if len(sims) else 0.0

    result = {
        "is_grounded": best_sim >= effective_threshold,
        "similarity": round(best_sim, 4),
        "threshold": effective_threshold,
    }

    # Article completeness validation
    completeness = validate_article_completeness(answer, context_docs)
    result["article_completeness"] = completeness
    if not completeness["is_complete"]:
        result["needs_regeneration"] = True

    return result


# ── LLM-based answer validation ───────────────────────────────────────────────

_VALIDATE_SYSTEM = (
    "Bạn là chuyên gia kiểm tra chất lượng câu trả lời pháp lý Việt Nam. "
    "Nhiệm vụ: xác minh câu trả lời có bám sát ngữ cảnh và không bịa đặt thông tin pháp lý. "
    "Trả lời LUÔN LUÔN là JSON hợp lệ, không có markdown, không có giải thích bên ngoài JSON."
)

_VALIDATE_USER_TEMPLATE = """\
Kiểm tra câu trả lời pháp lý sau dựa trên ngữ cảnh được cung cấp.

CÂU HỎI:
{query}

NGỮ CẢNH (tài liệu đã truy xuất):
{context}

CÂU TRẢ LỜI CẦN KIỂM TRA:
{answer}

TIÊU CHÍ KIỂM TRA:
1. Câu trả lời có bám sát nội dung ngữ cảnh không? (không bịa đặt thông tin)
2. Số hiệu văn bản, số điều luật trong câu trả lời có tồn tại trong ngữ cảnh không?
3. Câu trả lời có mâu thuẫn với nội dung ngữ cảnh không?
4. Nếu ngữ cảnh có điều luật liên quan, câu trả lời có trích dẫn nguồn không?

Trả lời theo định dạng JSON sau (không thêm bất cứ thứ gì ngoài JSON):
{{
  "is_valid": true hoặc false,
  "confidence": số thực từ 0.0 đến 1.0,
  "issues": ["danh sách vấn đề nếu có, để trống nếu hợp lệ"]
}}\
"""

_FALLBACK_ANSWER = "Không đủ thông tin để trả lời chính xác dựa trên tài liệu hiện có."

# Maximum context characters sent to the validator (keep cost/latency low)
_MAX_CONTEXT_CHARS = 3000


def _truncate_context_for_validation(context: str) -> str:
    """Truncate context to avoid very large LLM calls during validation."""
    if not context or len(context) <= _MAX_CONTEXT_CHARS:
        return context
    return context[:_MAX_CONTEXT_CHARS] + "\n...[ngữ cảnh đã rút gọn]"


def _parse_validation_json(raw: str) -> Optional[Dict]:
    """Extract and parse JSON from LLM output, tolerating minor formatting issues."""
    raw = (raw or "").strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


async def validate_answer(
    query: str,
    context: str,
    answer: str,
) -> Dict:
    """LLM-based validation: check whether an answer is grounded in the context.

    Catches hallucinated legal references, contradictions, and missing citations.

    Args:
        query:   The original user question.
        context: The retrieved context string used to generate the answer.
        answer:  The generated answer to validate.

    Returns:
        Dict with keys:
          ``is_valid``   (bool)       – True when the answer passes all checks.
          ``confidence`` (float 0–1)  – LLM's estimated confidence in the answer.
          ``issues``     (List[str])  – List of detected issues (empty when valid).

        On any LLM error, returns a permissive default so the pipeline is not blocked.
    """
    _safe_default: Dict = {"is_valid": True, "confidence": 0.5, "issues": []}

    if not OPENAI_API_KEY:
        return _safe_default

    if not answer or not context:
        return {"is_valid": False, "confidence": 0.0, "issues": ["Thiếu câu trả lời hoặc ngữ cảnh"]}

    try:
        from app.services.llm_client import generate

        prompt = _VALIDATE_USER_TEMPLATE.format(
            query=(query or "").strip(),
            context=_truncate_context_for_validation(context),
            answer=(answer or "").strip()[:2000],
        )
        raw: str = await generate(
            prompt=prompt,
            system=_VALIDATE_SYSTEM,
            temperature=0.0,
            max_tokens=300,
        )

        parsed = _parse_validation_json(raw)
        if parsed is None:
            log.warning("validate_answer: could not parse LLM JSON response — defaulting to valid")
            return _safe_default

        result: Dict = {
            "is_valid": bool(parsed.get("is_valid", True)),
            "confidence": float(parsed.get("confidence", 0.5)),
            "issues": list(parsed.get("issues", [])),
        }
        log.info(
            "LLM answer validation | is_valid=%s confidence=%.2f issues=%s",
            result["is_valid"],
            result["confidence"],
            result["issues"],
        )
        return result

    except Exception as exc:
        log.warning("validate_answer failed (allowing answer through): %s", exc)
        return _safe_default


def get_fallback_answer() -> str:
    """Standard fallback answer returned when validation fails and regeneration is skipped."""
    return _FALLBACK_ANSWER
