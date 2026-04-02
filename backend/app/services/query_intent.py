"""Một nguồn intent cho routing + cờ RAG (đồng bộ query_understanding ↔ get_rag_intents).

Nhóm pattern checklist / multi-doc / substantive / consultation đọc từ
``app/intent_patterns/routing.yaml`` (khóa ``routing.*``) — xem ``intent_pattern_config``.

A/B test: env ``INTENT_AB_MODE`` = "model" | "rule" | "shadow"
  model  → chỉ dùng multitask model (intent + flags từ flags_head)
  rule   → chỉ dùng rule-based pipeline (detect_intent_rule_based + YAML)
  shadow → chạy cả hai, log diff, trả về kết quả model (default production)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

from app.services.intent_detector import (
    VALID_INTENTS,
    _RAG_LEGAL_LOOKUP_INTENTS,
    detect_intent_rule_based,
    map_intent_to_rag_flags,
    normalize_legacy_intent,
)
from app.services.intent_pattern_config import routing_group_matches
from app.services.intent_pattern_config import get_flag_override_set_flags
from app.services.legal_scope import query_has_strong_legal_scope_signals

# Routing intent strings dùng bởi rag_chain_v2.py (giữ "document_drafting" / "document_summary")
_DETECTOR_TO_ROUTING_INTENT: Dict[str, str] = {
    "summarization": "document_summary",
    "document_generation": "document_drafting",
    # 8 grouped intents còn lại pass-through (legal_lookup, legal_explanation, ...)
    # Legacy fine-grained → giữ backward compat (sẽ xóa khi YAML F1 vượt)
    "tom_tat_van_ban": "document_summary",
    "soan_thao_van_ban": "document_drafting",
    "tao_bao_cao": "document_drafting",
}


def _query_needs_substantive_expansion_not_checklist(query: str) -> bool:
    return routing_group_matches(query, "substantive_expansion")


def query_requires_multi_document_synthesis(query: str) -> bool:
    """True khi cần truy xuất/tổng hợp nhiều văn bản pháp luật (không phải checklist hành chính thuần)."""
    q = (query or "").strip().lower()
    if len(q) < 6:
        return False
    return routing_group_matches(query, "multi_doc_synthesis")


def _is_checklist_documents(query: str) -> bool:
    if query_requires_multi_document_synthesis(query):
        return False
    if _query_needs_substantive_expansion_not_checklist(query):
        return False
    return routing_group_matches(query, "checklist_documents")


def map_detector_to_routing_intent(det: str) -> str:
    det = normalize_legacy_intent(det)
    if det == "nan":
        return "out_of_scope"
    if det in _DETECTOR_TO_ROUTING_INTENT:
        return _DETECTOR_TO_ROUTING_INTENT[det]
    if det in VALID_INTENTS:
        return det
    return "legal_lookup"


def _force_multi_article_for_comprehensive_statutory_queries(
    query: str, flags: Dict[str, bool]
) -> None:
    """Chính sách, tiêu chí dự án, thư viện + điều kiện — cần multi-query + nhiều điều."""
    q = (query or "").lower().strip()
    if len(q) < 10:
        return

    if re.search(r"chính\s+sách", q) and re.search(
        r"(nhà\s+nước|quốc\s+gia|đối\s+với)", q
    ):
        flags["needs_expansion"] = True
        flags["use_multi_article"] = True
        return
    if re.search(r"tiêu\s+ch[íi]", q) and re.search(
        r"(phân\s+loại|dự\s+án|trọng\s+điểm)", q
    ):
        flags["needs_expansion"] = True
        flags["use_multi_article"] = True
        return
    if "trọng điểm quốc gia" in q or "dự án trọng điểm quốc gia" in q:
        flags["needs_expansion"] = True
        flags["use_multi_article"] = True
        return
    if "thư viện công" in q and ("là gì" in q or "điều kiện" in q):
        flags["needs_expansion"] = True
        flags["use_multi_article"] = True


def _narrow_multi_article_boost(query: str, flags: Dict[str, bool]) -> None:
    """Chỉ bật multi-article/expansion khi câu thật sự liên quan danh mục đầu tư có điều kiện (tránh 'danh mục'/'các điều' quá rộng)."""
    q = (query or "").lower()
    markers = (
        "ngành, nghề đầu tư kinh doanh có điều kiện",
        "ngành nghề đầu tư kinh doanh có điều kiện",
        "danh mục ngành nghề đầu tư",
        "danh mục ngành, nghề",
        "kinh doanh có điều kiện",
        "điều kiện đầu tư kinh doanh",
    )
    if any(m in q for m in markers):
        flags["needs_expansion"] = True
        if not flags.get("is_legal_lookup"):
            flags["use_multi_article"] = True


def is_consultation_or_advisory_query(query: str) -> bool:
    """Câu tham mưu / tình huống — ưu tiên LLM, không dùng template multi-source."""
    q = (query or "").lower()
    if len(q) < 12:
        return False
    return routing_group_matches(query, "consultation_advisory")


def _sync_lookup_and_multi_article(detector_intent: str, flags: Dict[str, bool]) -> None:
    """Cố định is_legal_lookup + use_multi_article theo detector (sau regex boost)."""
    if detector_intent not in VALID_INTENTS:
        return
    lk = detector_intent in _RAG_LEGAL_LOOKUP_INTENTS
    flags["is_legal_lookup"] = lk
    flags["use_multi_article"] = not lk


# ── A/B test helpers ──────────────────────────────────────────────────────────

def _get_ab_mode() -> str:
    """Đọc INTENT_AB_MODE từ config; mặc định 'shadow' (chạy cả hai, dùng model)."""
    try:
        from app import config as app_config
        return str(getattr(app_config, "INTENT_AB_MODE", "shadow")).lower()
    except Exception:
        return "shadow"


def _classify_model(query: str) -> Optional[Tuple[str, float, Dict[str, bool]]]:
    """Gọi multitask model; trả (intent, conf, flags) hoặc None."""
    try:
        from app.services.intent_model_classifier import classify_multitask_sync
        return classify_multitask_sync(query)
    except Exception as exc:
        log.debug("Multitask model skipped: %s", exc)
        return None


def _log_ab_diff(query: str, model_det: str, model_conf: float,
                 rule_det: str, rule_conf: float) -> None:
    """Log kết quả so sánh A/B để đánh giá F1."""
    if model_det != rule_det:
        log.info(
            "AB_DIFF query=%r model=%s(%.2f) rule=%s(%.2f)",
            query[:80], model_det, model_conf, rule_det, rule_conf,
        )
    else:
        log.debug(
            "AB_AGREE query=%r intent=%s model_conf=%.2f rule_conf=%.2f",
            query[:80], model_det, model_conf, rule_conf,
        )


# ── Main bundle ───────────────────────────────────────────────────────────────

def compute_intent_bundle(query: str) -> Dict[str, Any]:
    """Một bundle: detector (thô) + routing (rag_chain) + rag_flags (cùng nguồn detector).

    A/B test (INTENT_AB_MODE):
      shadow → chạy cả model và rule-based, log diff, dùng model làm kết quả chính.
      model  → chỉ dùng multitask model (intent + flags từ flags_head).
      rule   → chỉ dùng rule-based pipeline (detect_intent_rule_based + YAML).
    """
    raw_q = query or ""
    if not raw_q.strip():
        return {
            "detector_intent": "legal_explanation",
            "detector_confidence": 0.10,
            "routing_intent": "legal_explanation",
            "rag_flags": {
                "is_legal_lookup": False,
                "use_multi_article": False,
                "needs_expansion": False,
            },
            "is_checklist": False,
        }

    ab_mode = _get_ab_mode()
    multi_syn = query_requires_multi_document_synthesis(raw_q)
    checklist = _is_checklist_documents(raw_q)

    # ── Multitask model path ──────────────────────────────────
    model_result = None
    if ab_mode in ("model", "shadow"):
        model_result = _classify_model(raw_q)

    # ── Rule-based path ───────────────────────────────────────
    rule_det: Optional[str] = None
    rule_conf: float = 0.0
    if ab_mode in ("rule", "shadow") or model_result is None:
        rule_det, rule_conf = detect_intent_rule_based(raw_q)

    # ── Decide which result to use ────────────────────────────
    if model_result is not None:
        det, conf, model_flags = model_result

        # Log A/B comparison in shadow mode
        if ab_mode == "shadow" and rule_det is not None:
            _log_ab_diff(raw_q, det, conf, rule_det, rule_conf)

        if det == "nan":
            if query_has_strong_legal_scope_signals(raw_q):
                det = "legal_lookup"
                conf = max(conf, 0.35)
            else:
                return {
                    "detector_intent": "nan",
                    "detector_confidence": conf,
                    "routing_intent": "out_of_scope",
                    "rag_flags": {
                        k: False for k in ("is_legal_lookup", "use_multi_article", "needs_expansion")
                    },
                    "is_checklist": False,
                }

        routing = "checklist_documents" if checklist else map_detector_to_routing_intent(det)
        # Start with model flags_head output, then apply YAML boosts
        flags = dict(model_flags)
        _apply_yaml_flag_boosts(raw_q, flags, det, multi_syn)
        return {
            "detector_intent": det,
            "detector_confidence": conf,
            "routing_intent": routing,
            "rag_flags": flags,
            "is_checklist": checklist,
            "ab_method": "model",
        }

    # ── Fallback: rule-based ──────────────────────────────────
    det, conf = rule_det, rule_conf  # type: ignore[assignment]
    if det == "nan":
        if query_has_strong_legal_scope_signals(raw_q):
            det = "legal_lookup"
            conf = max(float(conf or 0.0), 0.35)
        else:
            return {
                "detector_intent": "nan",
                "detector_confidence": conf,
                "routing_intent": "out_of_scope",
                "rag_flags": {
                    k: False for k in ("is_legal_lookup", "use_multi_article", "needs_expansion")
                },
                "is_checklist": False,
            }

    routing = "checklist_documents" if checklist else map_detector_to_routing_intent(det)
    flags = dict(map_intent_to_rag_flags(det))
    _apply_yaml_flag_boosts(raw_q, flags, det, multi_syn)
    return {
        "detector_intent": det,
        "detector_confidence": conf,
        "routing_intent": routing,
        "rag_flags": flags,
        "is_checklist": checklist,
        "ab_method": "rule",
    }


def _apply_yaml_flag_boosts(
    query: str, flags: Dict[str, bool], det: str, multi_syn: bool
) -> None:
    """Áp dụng các YAML routing group boosts lên flags dict (in-place)."""
    _narrow_multi_article_boost(query, flags)
    if multi_syn:
        flags["needs_expansion"] = True
    if _query_needs_substantive_expansion_not_checklist(query):
        flags["needs_expansion"] = True
        if (
            routing_group_matches(query, "multi_article_boost_substantive")
            and not flags.get("is_legal_lookup")
        ):
            flags["use_multi_article"] = True
    _sync_lookup_and_multi_article(det, flags)
    _force_multi_article_for_comprehensive_statutory_queries(query, flags)
    _apply_targeted_flag_overrides(query, flags, det)


def _apply_targeted_flag_overrides(query: str, flags: Dict[str, bool], det: str) -> None:
    """Áp override cờ từ config để dễ audit/versioning."""
    override = get_flag_override_set_flags(query or "")
    if not override:
        return
    for k in ("is_legal_lookup", "use_multi_article", "needs_expansion"):
        if k in override:
            flags[k] = bool(override[k])


def compute_rag_flags_for_query(query: str) -> Dict[str, bool]:
    """API cho intent_detector.get_rag_intents — không lặp logic marker."""
    return dict(compute_intent_bundle(query)["rag_flags"])


def query_mentions_conditional_investment(query: str) -> bool:
    """Footer 'danh mục đầu tư có điều kiện' chỉ khi câu hỏi liên quan."""
    q = (query or "").lower()
    return any(
        x in q
        for x in (
            "đầu tư kinh doanh có điều kiện",
            "ngành nghề đầu tư",
            "danh mục ngành nghề",
            "kinh doanh có điều kiện",
            "điều kiện kinh doanh",
        )
    )
