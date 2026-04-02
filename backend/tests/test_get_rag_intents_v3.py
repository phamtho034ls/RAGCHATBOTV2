"""Unit tests: intent_detector.get_rag_intents / map_intent_to_rag_flags (3 RAG flags)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.intent_detector import get_rag_intents, map_intent_to_rag_flags


def test_map_intent_article_query_happy_path() -> None:
    flags = map_intent_to_rag_flags("article_query")
    assert flags["is_legal_lookup"] is True
    assert flags["needs_expansion"] is False
    assert flags["use_multi_article"] is False


def test_map_intent_invalid_and_violation_edge() -> None:
    assert all(v is False for v in map_intent_to_rag_flags("").values())
    assert all(v is False for v in map_intent_to_rag_flags("not_in_valid_intents").values())
    flags = map_intent_to_rag_flags("xu_ly_vi_pham_hanh_chinh")
    assert flags["is_legal_lookup"] is False
    assert flags["use_multi_article"] is True
    assert flags["needs_expansion"] is True


def test_get_rag_intents_empty_query() -> None:
    flags = get_rag_intents("")
    assert flags == {
        "is_legal_lookup": False,
        "use_multi_article": False,
        "needs_expansion": False,
    }


def test_get_rag_intents_structural_article_query() -> None:
    flags = get_rag_intents("Điều 6 Luật Đầu tư 2025 quy định gì?")
    assert flags["is_legal_lookup"] is True
    assert flags["use_multi_article"] is False
