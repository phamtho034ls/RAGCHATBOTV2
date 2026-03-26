#!/usr/bin/env python3
"""Smoke: lấy ngẫu nhiên N câu từ data.json (repo root) và kiểm tra routing không chặn nhầm OOS."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Repo root: .../rag_chatbot, backend: .../rag_chatbot/backend
ROOT = Path(__file__).resolve().parents[2]
DATA_JSON = ROOT / "data.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="Số câu ngẫu nhiên")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if not DATA_JSON.is_file():
        print(f"Missing {DATA_JSON}", file=sys.stderr)
        return 1

    lines = [ln.strip() for ln in DATA_JSON.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue

    if args.seed is not None:
        random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))

    from app.services.legal_scope import query_has_strong_legal_scope_signals
    from app.services.query_intent import compute_intent_bundle

    oos = 0
    nan_kept = 0
    weak_signal = 0
    for r in sample:
        q = (r.get("question") or "").strip()
        b = compute_intent_bundle(q)
        ri = b.get("routing_intent")
        di = b.get("detector_intent")
        if ri == "out_of_scope":
            oos += 1
            print(f"[OOS] {di!r} | {q[:100]}...")
        if di == "nan":
            nan_kept += 1
        if not query_has_strong_legal_scope_signals(q):
            weak_signal += 1

    print(f"Sampled {len(sample)} questions: out_of_scope={oos}, detector still nan={nan_kept}")
    print(f"(Heuristic) questions without strong legal-scope regex match: {weak_signal}")
    if oos > max(1, len(sample) // 10):
        print("Warning: unusually many OOS in legal-looking dataset", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
