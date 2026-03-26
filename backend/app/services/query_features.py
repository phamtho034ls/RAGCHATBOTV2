"""Feature extraction for query-based retrieval strategy routing.

Extracts structural and semantic features from user queries to power
score-based strategy selection (lookup vs semantic vs multi_query).

Regex logic delegates to ``query_text_patterns`` (the canonical regex module)
so there is a single source of truth for each pattern.
Only lightweight keyword `in` checks that are NOT already in query_text_patterns
are defined here.
"""

from __future__ import annotations

import re
from typing import Dict

# ── Delegate to the canonical regex module for patterns that already exist there
from app.services.query_text_patterns import (
    query_asks_fine_amount,   # mức phạt / tiền phạt
    query_looks_procedural,   # thủ tục / hồ sơ / quy trình
    query_demands_specific_article,  # điều X reference
)

# ── Compiled patterns NOT already in query_text_patterns ─────────────────────

_DOC_NUMBER_RE = re.compile(r"\d{2,}/\d{4}/[A-ZĐa-zđ0-9\-]+")
_CLAUSE_REF_RE = re.compile(r"khoản\s+\d+", re.IGNORECASE)

# ── Lightweight keyword sets (all `in`-based, no regex) ──────────────────────

_EXPLANATION_KEYWORDS: frozenset[str] = frozenset([
    "là gì", "nghĩa là gì", "hiểu như thế nào", "giải thích",
    "định nghĩa", "khái niệm", "có nghĩa", "ý nghĩa",
    "như thế nào", "thế nào là",
])

_COMPARISON_KEYWORDS: frozenset[str] = frozenset([
    "so sánh", "khác nhau", "giống nhau", "đối chiếu",
    "khác gì", "điểm khác", "so với", "phân biệt",
    "sự khác biệt", "điểm giống",
])

_SCOPE_KEYWORDS: frozenset[str] = frozenset([
    "chính sách", "các quy định", "toàn bộ", "tất cả",
    "danh mục", "liệt kê", "những quy định", "các điều",
])


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_query_features(query: str) -> Dict[str, object]:
    """Extract structural and semantic features from a user query.

    These features are consumed by ``strategy_router.compute_strategy_scores``
    to decide which retrieval strategies to activate.

    Returns:
        Dict with the following keys:

        has_article_ref (bool):     Query explicitly references "Điều X".
        has_clause_ref (bool):      Query explicitly references "Khoản X".
        has_doc_number (bool):      Query contains a legal document number
                                    (e.g. ``100/2019/NĐ-CP``).
        is_procedure_like (bool):   Query asks about process / requirements.
        needs_explanation (bool):   Query asks for definition / explanation.
        needs_comparison (bool):    Query asks to compare two things.
        asks_fine_amount (bool):    Query asks about penalty / fine amounts.
        needs_broad_coverage (bool):Query needs comprehensive multi-doc scan.
        query_length (int):         Character count of stripped query.
        is_short_query (bool):      Very brief (<30 chars) — likely vague.
        is_long_query (bool):       Detailed (>100 chars) — complex question.
    """
    q = (query or "").strip()
    q_lower = q.lower()

    # Delegate to query_text_patterns (single source of truth)
    has_article_ref = query_demands_specific_article(q)
    is_procedure_like = query_looks_procedural(q)
    asks_fine_amount = query_asks_fine_amount(q)

    # Own lightweight patterns
    has_clause_ref = bool(_CLAUSE_REF_RE.search(q))
    has_doc_number = bool(_DOC_NUMBER_RE.search(q))

    needs_explanation = any(kw in q_lower for kw in _EXPLANATION_KEYWORDS)
    needs_comparison = any(kw in q_lower for kw in _COMPARISON_KEYWORDS)
    needs_broad_coverage = any(kw in q_lower for kw in _SCOPE_KEYWORDS)

    query_length = len(q)
    is_short_query = query_length < 30
    is_long_query = query_length > 100

    return {
        "has_article_ref": has_article_ref,
        "has_clause_ref": has_clause_ref,
        "has_doc_number": has_doc_number,
        "is_procedure_like": is_procedure_like,
        "needs_explanation": needs_explanation,
        "needs_comparison": needs_comparison,
        "asks_fine_amount": asks_fine_amount,
        "needs_broad_coverage": needs_broad_coverage,
        "query_length": query_length,
        "is_short_query": is_short_query,
        "is_long_query": is_long_query,
    }
