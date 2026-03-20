"""Validate groundedness of generated answers against retrieved context."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import ANSWER_VALIDATION_THRESHOLD
from app.pipeline.embedding import embed_query, embed_texts


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
) -> Dict[str, float | bool]:
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
