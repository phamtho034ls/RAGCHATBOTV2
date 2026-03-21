"""Đánh giá detect_intent_rule_based trên eval_dataset.json (không LLM)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.intent_detector import detect_intent_rule_based


def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_eval_intent_accuracy():
    rows = load_dataset()
    ok = 0
    for r in rows:
        intent, _ = detect_intent_rule_based(r["query"])
        if intent == r.get("expected_intent"):
            ok += 1
    acc = ok / max(len(rows), 1)
    print(f"Intent accuracy (rule_based): {ok}/{len(rows)} = {acc:.2%}")
    assert acc >= 0.5, f"accuracy {acc} below 0.5 for tiny smoke set"
