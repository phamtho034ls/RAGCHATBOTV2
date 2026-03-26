"""V3 hybrid: diversify_by_article + dynamic_max_articles."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval.article_selection import diversify_by_article, dynamic_max_articles


def _p(doc_id: int, score: float) -> dict:
    return {"document_id": doc_id, "rerank_score": score, "text_chunk": "x"}


def test_diversify_keeps_order_when_few_docs():
    passages = [_p(1, 0.9), _p(1, 0.8), _p(1, 0.7)]
    out = diversify_by_article(passages, min_docs=3)
    assert [x["document_id"] for x in out] == [1, 1, 1]


def test_diversify_round_robin_when_many_docs():
    passages = [_p(1, 0.9), _p(2, 0.85), _p(3, 0.8), _p(1, 0.7), _p(2, 0.6)]
    out = diversify_by_article(passages, min_docs=3)
    # order of first appearance doc ids: 1,2,3 then round-robin
    assert out[0]["document_id"] == 1
    assert out[1]["document_id"] == 2
    assert out[2]["document_id"] == 3


def test_dynamic_max_articles_gap():
    # gap nhỏ, 2 doc → return 3
    assert dynamic_max_articles([_p(1, 0.9), _p(2, 0.89)]) == 3
    p5 = [_p(1, 0.9), _p(2, 0.89), _p(3, 0.88), _p(4, 0.5), _p(5, 0.4)]
    assert dynamic_max_articles(p5) == 5
    # gap lớn → 1 nguồn
    p_wide = [_p(1, 0.9), _p(2, 0.5)]
    assert dynamic_max_articles(p_wide) == 1


def test_dynamic_max_articles_amendment_query():
    assert dynamic_max_articles([_p(1, 0.9)], "Luật sửa đổi bổ sung gì?") == 5
