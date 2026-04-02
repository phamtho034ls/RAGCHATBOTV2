"""PhoBERT multitask classifier — 8 grouped intent labels + flags_head monitoring.

Checkpoint: ``app/intent_model/phobert_multitask_a100.pt``
  - ``phobert.*``      → RobertaModel backbone (vinai/phobert-base, vocab=64001)
  - ``intent_head.*``  → Linear(768, 8) — 8 grouped intent classes (softmax)
  - ``flags_head.*``   → Linear(768, 4) — output thô để monitoring/retraining

Tokenizer: vinai/phobert-base (downloaded from HuggingFace on first run).

Label order (must match training):
  intent: sorted alphabetically → [admin_scenario, comparison, document_generation,
                                    legal_explanation, legal_lookup, procedure,
                                    summarization, violation]
  flags_head raw order: [is_legal_lookup, needs_expansion, use_multi_article, is_scenario]
  Runtime flags dùng trong hệ thống lấy từ ``map_intent_to_rag_flags`` (3 cờ).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ── Module-level singletons ──────────────────────────────────────────────────
_model: Optional["_PhoBERTMultitask"] = None
_tokenizer = None
_device: str = "cpu"
_load_failed: bool = False

# ── Label manifests ───────────────────────────────────────────────────────────

# 8 grouped intent labels — theo thứ tự xuất hiện đầu tiên trong INTENT_MAPPING
# (insertion order of dict values, NOT alphabetical — phải khớp thứ tự nhãn khi training)
MULTITASK_INTENT_LABELS: List[str] = [
    "legal_lookup",        # 0 — article_query, tra_cuu, trich_xuat, can_cu_phap_ly
    "legal_explanation",   # 1 — giai_thich_quy_dinh, hoi_dap_chung
    "procedure",           # 2 — huong_dan_thu_tuc, kiem_tra_ho_so
    "violation",           # 3 — xu_ly_vi_pham_hanh_chinh, kiem_tra_thanh_tra
    "comparison",          # 4 — so_sanh_van_ban
    "summarization",       # 5 — tom_tat_van_ban
    "document_generation", # 6 — soan_thao_van_ban, tao_bao_cao
    "admin_scenario",      # 7 — admin_planning, to_chuc_su_kien_cong, hoa_giai_van_dong, document_meta_relation
]

# 4 RAG flag outputs — order must match training (matches map_intent_to_rag_flags dict)
FLAGS_ORDER: List[str] = [
    "is_legal_lookup",
    "needs_expansion",
    "use_multi_article",
    "is_scenario",
]

_PT_FILENAME = "phobert_multitask_a100.pt"
_PHOBERT_HF_NAME = "vinai/phobert-base"
_HIDDEN_SIZE = 768


# ── Model architecture ────────────────────────────────────────────────────────

class _PhoBERTMultitask(nn.Module):
    """PhoBERT backbone + dual classification heads matching the .pt checkpoint."""

    def __init__(self, roberta_config):
        super().__init__()
        from transformers import RobertaModel

        self.phobert = RobertaModel(roberta_config, add_pooling_layer=True)
        self.intent_head = nn.Linear(_HIDDEN_SIZE, len(MULTITASK_INTENT_LABELS))
        self.flags_head = nn.Linear(_HIDDEN_SIZE, len(FLAGS_ORDER))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        # Dùng CLS token (không qua pooler.dense+tanh) — khớp với cách training
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        return self.intent_head(cls_emb), self.flags_head(cls_emb)


# ── Loading helpers ───────────────────────────────────────────────────────────

def _ensure_loaded() -> None:
    global _model, _tokenizer, _device, _load_failed

    if _model is not None:
        return
    if _load_failed:
        raise RuntimeError("PhoBERT multitask classifier previously failed to load")

    from app import config as app_config

    model_dir = Path(getattr(app_config, "INTENT_MODEL_DIR", ""))
    pt_path = model_dir / _PT_FILENAME
    if not pt_path.is_file():
        _load_failed = True
        raise FileNotFoundError(
            f"Checkpoint không tồn tại: {pt_path}. "
            f"Đặt file vào app/intent_model/{_PT_FILENAME}"
        )

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        _load_failed = True
        raise RuntimeError("Cần cài transformers: pip install transformers") from exc

    _device = str(getattr(app_config, "INTENT_MODEL_DEVICE", "cpu"))
    hf_name = str(getattr(app_config, "INTENT_MODEL_PHOBERT_NAME", _PHOBERT_HF_NAME))
    log.info("Loading PhoBERT multitask from %s (device=%s)", pt_path, _device)

    # ── 1. Tokenizer ─────────────────────────────────────────
    try:
        tok = AutoTokenizer.from_pretrained(hf_name, use_fast=False)
    except Exception as exc:
        _load_failed = True
        raise RuntimeError(
            f"Không tải được tokenizer '{hf_name}' từ HuggingFace: {exc}"
        ) from exc

    # ── 2. Build model shell ─────────────────────────────────
    try:
        cfg = AutoConfig.from_pretrained(hf_name)
        mdl = _PhoBERTMultitask(cfg)
    except Exception as exc:
        _load_failed = True
        raise RuntimeError(f"Không tạo được model shell: {exc}") from exc

    # ── 3. Load state dict ───────────────────────────────────
    try:
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        _load_failed = True
        raise RuntimeError(f"torch.load thất bại cho {pt_path}: {exc}") from exc

    # The checkpoint IS the state dict (OrderedDict of tensors)
    state_dict = ckpt
    if not isinstance(state_dict, dict) or not all(
        isinstance(v, torch.Tensor) for v in list(state_dict.values())[:3]
    ):
        _load_failed = True
        raise RuntimeError(
            "Checkpoint không phải OrderedDict tensor — kiểm tra lại file .pt"
        )

    missing, unexpected = mdl.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        log.warning(
            "State dict mismatch: missing=%s, unexpected=%s",
            missing[:5],
            unexpected[:5],
        )

    mdl.eval()
    mdl.to(_device)

    _tokenizer = tok
    _model = mdl
    log.info(
        "PhoBERT multitask ready: %d intent classes, %d flag outputs",
        len(MULTITASK_INTENT_LABELS),
        len(FLAGS_ORDER),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def warmup_intent_classifier() -> None:
    """Gọi từ lifespan — preload weights tránh block request đầu tiên."""
    from app import config as app_config

    if not getattr(app_config, "INTENT_MODEL_ENABLED", True):
        log.info("Intent classifier: disabled (INTENT_MODEL_ENABLED=false).")
        return
    try:
        _ensure_loaded()
        log.info("Intent classifier: PhoBERT multitask warm-up thành công.")
    except Exception as exc:
        log.warning(
            "Intent classifier: warm-up thất bại, fallback semantic/LLM — %s", exc
        )


def classify_multitask_sync(
    query: str,
) -> Optional[Tuple[str, float, Dict[str, bool]]]:
    """Chạy model, trả ``(intent, confidence, flags_dict)`` hoặc ``None``.

    ``flags_dict`` được tính từ ``map_intent_to_rag_flags(intent)`` (rule-based từ
    nhóm intent đã phân loại) để đảm bảo tính nhất quán. ``flags_head`` của model
    được giữ lại cho monitoring / future training.
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
    oos_max = float(getattr(app_config, "INTENT_MODEL_OOS_MAX_PROB", 0.20))
    max_len = int(getattr(app_config, "INTENT_MODEL_MAX_LENGTH", 256))

    enc = _tokenizer(
        query.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.no_grad():
        intent_logits, flags_logits = _model(**enc)
        intent_probs = torch.softmax(intent_logits, dim=-1).squeeze(0)
        # flags_head lưu để monitoring; flags_dict dùng rule-based map
        flags_probs = torch.sigmoid(flags_logits).squeeze(0)

    conf = float(intent_probs.max().item())
    pred = int(intent_probs.argmax().item())

    if pred < 0 or pred >= len(MULTITASK_INTENT_LABELS):
        return None

    if conf < oos_max:
        log.info("Multitask OOS: max_prob=%.3f < %.2f → nan", conf, oos_max)
        return "nan", conf, _empty_flags()

    if conf < min_conf:
        return None

    intent_label = MULTITASK_INTENT_LABELS[pred]

    # Derive flags từ intent (rule-based mapping — nhất quán, không phụ thuộc thứ tự flags_head)
    from app.services.intent_detector import map_intent_to_rag_flags
    flags = dict(map_intent_to_rag_flags(intent_label))

    # Log raw flags_head values for future analysis / retraining signal
    raw_flags = {FLAGS_ORDER[i]: round(float(flags_probs[i].item()), 3) for i in range(len(FLAGS_ORDER))}
    log.debug(
        "Multitask: intent=%s conf=%.3f flags=%s raw_flags=%s",
        intent_label, conf, flags, raw_flags,
    )
    return intent_label, conf, flags


def classify_intent_sync(query: str) -> Optional[Tuple[str, float]]:
    """Backward-compatible wrapper: trả ``(intent, confidence)`` hoặc ``None``.

    Dùng bởi ``_detect_intent_classifier`` trong ``intent_detector.py``.
    """
    result = classify_multitask_sync(query)
    if result is None:
        return None
    intent, conf, _ = result
    return intent, conf


def _empty_flags() -> Dict[str, bool]:
    # Runtime chỉ dùng 3 cờ chuẩn của pipeline.
    return {
        "is_legal_lookup": False,
        "use_multi_article": False,
        "needs_expansion": False,
    }
