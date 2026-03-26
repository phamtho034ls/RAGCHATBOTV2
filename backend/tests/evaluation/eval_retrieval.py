"""Retrieval metrics: xem ``retrieval_metrics.py`` và ``test_retrieval_metrics.py``."""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from retrieval_metrics import precision_recall_at_k_chunks  # noqa: E402


def test_retrieval_eval_smoke_import():
    prec, rec = precision_recall_at_k_chunks(["x"], {"x"}, 1)
    assert prec == 1.0 and rec == 1.0
