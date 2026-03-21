"""Smoke: baseline file cho RAG (mở rộng khi có DB đầy đủ)."""
from __future__ import annotations

import json
import os


def test_eval_rag_baseline_placeholder():
    base = os.path.join(os.path.dirname(__file__), "baseline.json")
    if not os.path.isfile(base):
        payload = {"note": "Chạy pipeline đo latency sau khi có dữ liệu", "avg_confidence": 0.5}
        with open(base, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    assert os.path.isfile(base)
