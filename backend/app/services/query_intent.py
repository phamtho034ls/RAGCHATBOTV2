"""Một nguồn intent cho routing + cờ RAG (đồng bộ query_understanding ↔ get_rag_intents).

Nhóm pattern checklist / multi-doc / substantive / consultation đọc từ
``app/intent_patterns/routing.yaml`` (khóa ``routing.*``) — xem ``intent_pattern_config``.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from app.services.intent_detector import (
    VALID_INTENTS,
    _RAG_LEGAL_LOOKUP_INTENTS,
    detect_intent_rule_based,
    map_intent_to_rag_flags,
    normalize_legacy_intent,
)
from app.services.intent_pattern_config import routing_group_matches
from app.services.legal_scope import query_has_strong_legal_scope_signals

_DETECTOR_TO_ROUTING_INTENT: Dict[str, str] = {
    "tom_tat_van_ban": "document_summary",
    "soan_thao_van_ban": "document_drafting",
    "tra_cuu_van_ban": "legal_lookup",
    "trich_xuat_van_ban": "legal_lookup",
    "document_meta_relation": "legal_lookup",
    "article_query": "legal_lookup",
    "can_cu_phap_ly": "legal_lookup",
    "hoi_dap_chung": "hoi_dap_chung",
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


def compute_intent_bundle(query: str) -> Dict[str, Any]:
    """Một bundle: detector (thô) + routing (rag_chain) + rag_flags (cùng nguồn detector)."""
    raw_q = query or ""
    if not raw_q.strip():
        return {
            "detector_intent": "hoi_dap_chung",
            "detector_confidence": 0.10,
            "routing_intent": "hoi_dap_chung",
            "rag_flags": {
                "is_scenario": False,
                "is_legal_lookup": False,
                "use_multi_article": False,
                "needs_expansion": False,
            },
            "is_checklist": False,
        }
    multi_syn = query_requires_multi_document_synthesis(raw_q)
    checklist = _is_checklist_documents(raw_q)
    det, conf = detect_intent_rule_based(raw_q)
    if det == "nan":
        # Không chặn sớm nếu câu có tín hiệu pháp luật/hành chính rõ — cho RAG thử tra cứu.
        if query_has_strong_legal_scope_signals(raw_q):
            det = "tra_cuu_van_ban"
            conf = max(float(conf or 0.0), 0.35)
        else:
            return {
                "detector_intent": "nan",
                "detector_confidence": conf,
                "routing_intent": "out_of_scope",
                "rag_flags": {
                    "is_scenario": False,
                    "is_legal_lookup": False,
                    "use_multi_article": False,
                    "needs_expansion": False,
                },
                "is_checklist": False,
            }
    routing = "checklist_documents" if checklist else map_detector_to_routing_intent(det)
    flags = dict(map_intent_to_rag_flags(det))
    _narrow_multi_article_boost(raw_q, flags)
    if multi_syn:
        flags["needs_expansion"] = True
    if _query_needs_substantive_expansion_not_checklist(raw_q):
        flags["needs_expansion"] = True
        if (
            routing_group_matches(raw_q, "multi_article_boost_substantive")
            and not flags.get("is_legal_lookup")
        ):
            flags["use_multi_article"] = True
    _sync_lookup_and_multi_article(det, flags)
    _force_multi_article_for_comprehensive_statutory_queries(raw_q, flags)
    return {
        "detector_intent": det,
        "detector_confidence": conf,
        "routing_intent": routing,
        "rag_flags": flags,
        "is_checklist": checklist,
    }


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
