"""Score-based retrieval strategy selection.

Replaces intent-based ``if intent == ...`` routing with a numeric scoring
system that maps query features to retrieval strategy weights.

Strategies:
    lookup      – Direct DB/index lookup for a specific article or document.
    semantic    – Dense vector similarity search.
    multi_query – Multiple sub-queries + merge (best for broad / complex queries).

Usage::

    from app.services.query_features import extract_query_features
    from app.services.strategy_router import compute_strategy_scores, select_strategies

    features = extract_query_features(user_query)
    scores   = compute_strategy_scores(features)
    selected = select_strategies(scores, top_k=2)

    # selected ∈ ["lookup", "semantic", "multi_query"]
"""

from __future__ import annotations

import logging
from typing import Dict, List

log = logging.getLogger(__name__)

# ── Strategy name constants ────────────────────────────────────────────────────

STRATEGY_LOOKUP = "lookup"
STRATEGY_SEMANTIC = "semantic"
STRATEGY_MULTI_QUERY = "multi_query"

_ALL_STRATEGIES = (STRATEGY_LOOKUP, STRATEGY_SEMANTIC, STRATEGY_MULTI_QUERY)


# ── Scoring ────────────────────────────────────────────────────────────────────

def compute_strategy_scores(features: Dict[str, object]) -> Dict[str, float]:
    """Compute retrieval strategy scores from query features.

    Scores are additive — multiple features boost the same strategy.
    Higher score ↔ higher priority.

    Args:
        features: Output of ``extract_query_features()``.

    Returns:
        Dict mapping each strategy name to a float score ≥ 0.
    """
    scores: Dict[str, float] = {k: 0.0 for k in _ALL_STRATEGIES}

    # ── Lookup boosters ────────────────────────────────────────────────────────
    # Explicit article reference → direct lookup is very reliable
    if features.get("has_article_ref"):
        scores[STRATEGY_LOOKUP] += 0.7

    # Explicit clause reference also implies a specific article
    if features.get("has_clause_ref"):
        scores[STRATEGY_LOOKUP] += 0.4

    # Document number in query → lookup by doc is the optimal path
    if features.get("has_doc_number"):
        scores[STRATEGY_LOOKUP] += 0.5

    # ── Multi-query boosters ───────────────────────────────────────────────────
    # Comparison queries need multiple context snippets from different sources
    if features.get("needs_comparison"):
        scores[STRATEGY_MULTI_QUERY] += 0.8

    # Long, detailed queries are complex; multiple sub-queries improve coverage
    if features.get("is_long_query"):
        scores[STRATEGY_MULTI_QUERY] += 0.5

    # Procedural queries require step-by-step coverage across articles
    if features.get("is_procedure_like"):
        scores[STRATEGY_MULTI_QUERY] += 0.4
        scores[STRATEGY_SEMANTIC] += 0.2

    # Broad / comprehensive scope queries need wide retrieval
    if features.get("needs_broad_coverage"):
        scores[STRATEGY_MULTI_QUERY] += 0.5
        scores[STRATEGY_SEMANTIC] += 0.1

    # Short vague queries → expand into sub-queries
    if features.get("is_short_query"):
        scores[STRATEGY_MULTI_QUERY] += 0.3

    # Fine amount queries often span multiple enforcement articles
    if features.get("asks_fine_amount"):
        scores[STRATEGY_MULTI_QUERY] += 0.3

    # ── Semantic boosters ─────────────────────────────────────────────────────
    # Definition / explanation queries are best served by semantic similarity
    if features.get("needs_explanation"):
        scores[STRATEGY_SEMANTIC] += 0.5

    # ── Fallback: ensure at least one strategy has a score ────────────────────
    if all(v == 0.0 for v in scores.values()):
        scores[STRATEGY_SEMANTIC] = 0.3

    log.debug("Strategy scores: %s | features: %s", scores, features)
    return scores


def select_strategies(
    scores: Dict[str, float],
    top_k: int = 2,
) -> List[str]:
    """Select the top-k retrieval strategies based on computed scores.

    Args:
        scores: Output of ``compute_strategy_scores()``.
        top_k:  Maximum number of strategies to return (default 2).

    Returns:
        List of strategy names sorted by descending score.
        Always contains at least one strategy (``semantic`` as hard fallback).
    """
    active = [
        (name, score) for name, score in scores.items() if score > 0.0
    ]
    active.sort(key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in active[:top_k]]

    if not selected:
        selected = [STRATEGY_SEMANTIC]

    log.info(
        "Selected retrieval strategies: %s (scores=%s)",
        selected,
        {k: round(v, 2) for k, v in scores.items()},
    )
    return selected
