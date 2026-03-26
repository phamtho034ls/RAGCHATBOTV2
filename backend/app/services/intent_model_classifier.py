"""Fine-tuned PhoBERT (RobertaForSequenceClassification) — nhãn intent khớp ``VALID_INTENTS`` (id 0..17)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

_model = None
_tokenizer = None
_device: str = "cpu"
_load_failed: bool = False


def _intent_labels() -> List[str]:
    # Tránh import vòng: chỉ gọi sau khi intent_detector đã load xong.
    from app.services.intent_detector import VALID_INTENTS

    return list(VALID_INTENTS)


def warmup_intent_classifier() -> None:
    """Gọi từ lifespan — preload trọng số (tránh block request đầu tiên)."""
    from app import config as app_config

    if not getattr(app_config, "INTENT_MODEL_ENABLED", True):
        log.info("Intent classifier: disabled (INTENT_MODEL_ENABLED=false).")
        return
    try:
        _ensure_loaded()
        log.info("Intent classifier: model ready.")
    except Exception as exc:
        log.warning("Intent classifier: warmup failed, dùng semantic/LLM — %s", exc)


def _ensure_loaded() -> None:
    global _model, _tokenizer, _device, _load_failed
    if _model is not None:
        return
    if _load_failed:
        raise RuntimeError("intent classifier previously failed to load")

    from app import config as app_config

    path = Path(getattr(app_config, "INTENT_MODEL_DIR", ""))
    if not path.is_dir():
        raise FileNotFoundError(f"INTENT_MODEL_DIR không tồn tại: {path}")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        _load_failed = True
        raise RuntimeError(
            "Cần cài transformers để dùng intent_model: pip install transformers"
        ) from exc

    _device = str(getattr(app_config, "INTENT_MODEL_DEVICE", "cpu"))
    log.info("Loading intent classifier from %s (device=%s)", path, _device)

    tok = AutoTokenizer.from_pretrained(str(path), use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(str(path))
    mdl.eval()
    mdl.to(_device)

    labels = _intent_labels()
    n_out = int(getattr(mdl.config, "num_labels", 0) or 0)
    if n_out != len(labels):
        raise ValueError(
            f"num_labels={n_out} nhưng VALID_INTENTS có {len(labels)} — kiểm tra thứ tự training."
        )

    _tokenizer = tok
    _model = mdl


def classify_intent_sync(query: str) -> Optional[Tuple[str, float]]:
    """
    Trả (intent, confidence) nếu model bật, load được và softmax max >= ngưỡng;
    ngược lại None để pipeline rơi xuống semantic / LLM.
    """
    from app import config as app_config

    if not getattr(app_config, "INTENT_MODEL_ENABLED", True):
        return None
    if not query or not query.strip():
        return None

    try:
        _ensure_loaded()
    except Exception as exc:
        log.debug("Intent classifier unavailable: %s", exc)
        return None

    assert _model is not None and _tokenizer is not None

    min_conf = float(getattr(app_config, "INTENT_MODEL_MIN_CONFIDENCE", 0.50))
    max_len = int(getattr(app_config, "INTENT_MODEL_MAX_LENGTH", 256))

    import torch

    text = query.strip()
    enc = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.no_grad():
        logits = _model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    conf, pred = float(probs.max().item()), int(probs.argmax().item())
    labels = _intent_labels()
    if pred < 0 or pred >= len(labels):
        return None

    oos_max = float(getattr(app_config, "INTENT_MODEL_OOS_MAX_PROB", 0.20))
    if conf < oos_max:
        log.info("Intent classifier OOS: max_prob=%.3f < %.2f → nan", conf, oos_max)
        return "nan", conf

    if conf < min_conf:
        return None
    return labels[pred], conf
