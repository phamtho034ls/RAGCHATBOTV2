"""V3 — đa dạng hóa passage theo document + số article động (không phụ thuộc DB/Qdrant)."""

from __future__ import annotations

import re
from collections import deque
from typing import Dict, List

_AMEND_QUERY_HINT_RE = re.compile(
    r"sửa\s*đổi|bổ\s*sung|thay\s*thế|đính\s*chính|"
    r"những\s*điều\s*(?:được\s*)?(?:sửa|bổ\s*sung|thay)|"
    r"điều\s*nào\s*(?:được\s*)?(?:sửa|bổ\s*sung|thay)",
    re.IGNORECASE,
)


def _passage_rerank_score(p: Dict) -> float:
    return float(p.get("rerank_score", p.get("rrf_score", p.get("score", 0.0))))


def diversify_by_article(reranked_passages: List[Dict], min_docs: int = 3) -> List[Dict]:
    """Nếu top-5 có đủ nguồn (document_id), sắp lại round-robin theo document để đa dạng hóa."""
    if not reranked_passages:
        return []
    top5 = reranked_passages[:5]
    seen_docs = set()
    for p in top5:
        did = p.get("document_id")
        if did is not None:
            seen_docs.add(did)
    if len(seen_docs) < min_docs:
        return list(reranked_passages)

    order: List[int] = []
    seen_order = set()
    for p in reranked_passages:
        did = p.get("document_id")
        if did is None:
            continue
        if did not in seen_order:
            seen_order.add(did)
            order.append(did)
    if not order:
        return list(reranked_passages)

    buckets: Dict[int, deque] = {did: deque() for did in order}
    tail_other: List[Dict] = []
    for p in reranked_passages:
        did = p.get("document_id")
        if did in buckets:
            buckets[did].append(p)
        else:
            tail_other.append(p)

    out: List[Dict] = []
    while any(buckets[d] for d in order):
        for did in order:
            if buckets[did]:
                out.append(buckets[did].popleft())
    out.extend(tail_other)
    return out


def dynamic_max_articles(reranked_passages: List[Dict], query: str = "") -> int:
    """Quyết định số article/context nguồn: gap score thấp + nhiều doc → dùng nhiều nguồn."""
    if _AMEND_QUERY_HINT_RE.search(query or ""):
        return 5
    if len(reranked_passages) < 2:
        return 1
    s0 = _passage_rerank_score(reranked_passages[0])
    s1 = _passage_rerank_score(reranked_passages[1])
    gap = s0 - s1
    top5 = reranked_passages[:5]
    unique_docs = len({p.get("document_id") for p in top5 if p.get("document_id") is not None})
    if gap < 0.10 and unique_docs >= 3:
        return 5
    if gap < 0.20 and unique_docs >= 2:
        return 3
    return 1
