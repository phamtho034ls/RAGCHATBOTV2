"""DEPRECATED (v3): RAG intent flags dùng `intent_detector.get_rag_intents()` + `map_intent_to_rag_flags`.

Module này giữ lại để tham khảo; `rag_chain_v2` và `main` không còn import/warmup.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# INTENT PROTOTYPES – câu ví dụ tiêu biểu cho mỗi loại
# Càng nhiều ví dụ đa dạng, nhận diện càng ổn định. Có thể bổ sung từ log thực tế.
# ══════════════════════════════════════════════════════════════

INTENT_PROTOTYPES: Dict[str, List[str]] = {
    "legal_lookup": [
        "Điều 9 Luật Di sản văn hóa quy định gì?",
        "Tóm tắt Nghị định 144/2021",
        "Nội dung Điều 31 Luật Thể dục thể thao",
        "Quy định của điều 5 Nghị định 36",
        "Nghị định 112/2007 hướng dẫn những gì?",
        "Điều 50 Nghị định 36/2019 quy định về điều kiện kinh doanh thể thao",
    ],
    "scenario": [
        "Karaoke gây ồn vào ban đêm thì xử lý thế nào?",
        "Hãy đề xuất giải pháp xử lý vi phạm quảng cáo trên địa bàn xã",
        "Tình huống biển quảng cáo sai quy định cần làm gì?",
        "Cần tham mưu UBND xử lý vi phạm văn hóa",
        "Kế hoạch ra quân kiểm tra dịch vụ karaoke",
        "Trên địa bàn phường có cơ sở kinh doanh vi phạm, cần xử lý ra sao?",
    ],
    "multi_article": [
        "Điều kiện kinh doanh hoạt động thể thao theo các văn bản hiện hành",
        "Quy định về tổ chức giải thể thao quần chúng",
        "So sánh Luật Thể dục thể thao 2006 và 2018",
        "Các văn bản quy định về điều kiện thành lập câu lạc bộ thể thao",
        "Phát triển thể thao thành tích cao",
        "Xây dựng cơ sở vật chất cho phát triển thể thao thành tích cao",
        "Quy định của luật và nghị định về hoạt động thể thao",
    ],
    "need_expansion": [
        "Liệt kê các hành vi bị cấm trong lĩnh vực văn hóa",
        "Những điều kiện kinh doanh thể thao cần đáp ứng",
        "Các ngành nghề cấm đầu tư theo quy định",
        "Danh sách các điều luật liên quan đến xử phạt",
        "Bao gồm những gì khi đăng ký hoạt động thể thao",
        "Nhiều điều quy định về quảng cáo ngoài trời",
    ],
}

# Intent mặc định khi không đủ tin cậy
DEFAULT_INTENT = "general"

# ══════════════════════════════════════════════════════════════
# SEMANTIC INDEX (built at startup or first use)
# ══════════════════════════════════════════════════════════════

_intent_labels: List[str] = []
_intent_matrix: Optional[np.ndarray] = None


def _build_intent_index() -> None:
    global _intent_labels, _intent_matrix

    from app.pipeline.embedding import embed_texts

    sentences: List[str] = []
    labels: List[str] = []
    for intent, examples in INTENT_PROTOTYPES.items():
        for s in examples:
            sentences.append(s)
            labels.append(intent)

    if not sentences:
        return

    t0 = time.monotonic()
    _intent_matrix = embed_texts(sentences)
    _intent_labels = labels
    elapsed = (time.monotonic() - t0) * 1000
    log.info(
        "Intent index built: %d examples, %d intents (%.0f ms)",
        len(sentences),
        len(INTENT_PROTOTYPES),
        elapsed,
    )


def warmup_intent_index() -> None:
    """Gọi khi khởi động app để build index sẵn (tránh latency lần đầu)."""
    try:
        if _intent_matrix is None or not _intent_labels:
            _build_intent_index()
    except Exception as exc:
        log.warning("Intent index warmup failed: %s", exc)


def _semantic_classify(query: str, top_n: int = 4) -> List[tuple]:
    """Trả về top_n (intent, score) theo similarity với ngân hàng câu ví dụ."""
    if _intent_matrix is None or not _intent_labels:
        _build_intent_index()
    if _intent_matrix is None or not _intent_labels:
        return []

    from app.pipeline.embedding import embed_texts

    q_vec = embed_texts([query])
    if q_vec.size == 0:
        return []
    q_vec = q_vec.reshape(1, -1)

    # Similarity từng câu ví dụ
    sims = (q_vec @ _intent_matrix.T).flatten()

    # Lấy điểm cao nhất theo từng intent (mỗi intent có nhiều câu ví dụ)
    intent_best: Dict[str, float] = {}
    for i, label in enumerate(_intent_labels):
        s = float(sims[i])
        if label not in intent_best or s > intent_best[label]:
            intent_best[label] = s

    sorted_intents = sorted(intent_best.items(), key=lambda x: x[1], reverse=True)
    return sorted_intents[:top_n]


def classify_intent(
    query: str,
    top_n: int = 2,
    confidence_threshold: Optional[float] = None,
) -> List[Dict]:
    """Phân loại ý định câu hỏi dựa trên embedding so với ngân hàng câu ví dụ.

    Returns:
        List[{"intent": str, "confidence": float}], giảm dần theo confidence.
        Nếu mọi score dưới ngưỡng, vẫn trả về intent có score cao nhất (để fallback).
    """
    if not query or not query.strip():
        return [{"intent": DEFAULT_INTENT, "confidence": 0.0}]

    from app.config import INTENT_CONFIDENCE_THRESHOLD

    threshold = confidence_threshold if confidence_threshold is not None else INTENT_CONFIDENCE_THRESHOLD
    raw = _semantic_classify(query, top_n=top_n + 2)

    results = [
        {"intent": intent, "confidence": round(score, 3)}
        for intent, score in raw
    ]

    if not results:
        return [{"intent": DEFAULT_INTENT, "confidence": 0.0}]

    return results[:top_n]


def get_rag_intents(
    query: str,
    confidence_threshold: Optional[float] = None,
) -> Dict[str, bool]:
    """Trả về các flag dùng trực tiếp cho RAG: is_scenario, is_legal_lookup, use_multi_article, needs_expansion.

    Dựa trên intent có confidence cao nhất (và có thể intent thứ hai).
    Kết quả có thể dùng thay thế hoặc bổ sung cho regex trong rag_chain_v2 / query_expansion.
    """
    from app.config import INTENT_CONFIDENCE_THRESHOLD, USE_MULTI_ARTICLE_FOR_CONDITIONS

    threshold = confidence_threshold if confidence_threshold is not None else INTENT_CONFIDENCE_THRESHOLD
    intents = classify_intent(query, top_n=3, confidence_threshold=threshold)

    out = {
        "is_scenario": False,
        "is_legal_lookup": False,
        "use_multi_article": False,
        "needs_expansion": False,
    }

    for item in intents:
        intent = item["intent"]
        conf = item["confidence"]
        if conf < threshold:
            continue
        if intent == "scenario":
            out["is_scenario"] = True
        elif intent == "legal_lookup":
            out["is_legal_lookup"] = True
        elif intent == "multi_article" and USE_MULTI_ARTICLE_FOR_CONDITIONS:
            out["use_multi_article"] = True
        elif intent == "need_expansion":
            out["needs_expansion"] = True

    # Ưu tiên: legal_lookup rõ ràng thì không coi là scenario
    if out["is_legal_lookup"] and out["is_scenario"]:
        # Chỉ giữ scenario nếu legal_lookup không phải intent cao nhất
        if intents and intents[0]["intent"] == "legal_lookup":
            out["is_scenario"] = False

    return out
