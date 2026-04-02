"""
Intent detection — guard → PhoBERT multitask → prototype embedding → structural (YAML) → LLM.

Pipeline (dừng khi một tầng trả intent đủ tin cậy):

  Layer 0: Guard — câu rỗng / quá ngắn / không có chữ cái
  Layer 1: Classifier — PhoBERT multitask (``phobert_multitask_a100.pt``, 8 nhóm intent + 4 cờ RAG)
  Layer 2: Semantic — cosine với prototype (SentenceTransformer); câu mẫu + ``intent_patterns/routing.yaml``
  Layer 3: Structural — regex fallback từ YAML (``INTENT_PATTERNS_YAML``), không ưu tiên trước ML
  Layer 4: LLM — zero-shot khi các tầng trên không quyết định

Nhãn intent: 8 nhóm (INTENT_MAPPING gộp 18 nhãn cũ → 8 nhóm mới).
Structural/routing patterns: ``app/intent_patterns/routing.yaml`` (mở rộng không cần sửa Python).
API công khai giữ tương thích: ``detect_intent``, ``detect_intent_rule_based``, ``map_intent_to_rag_flags``.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

from app.services.intent_pattern_config import (
    get_prototype_sentences_extra,
    structural_match,
)

# ══════════════════════════════════════════════════════════════
# CONSTANTS — KHÔNG THAY ĐỔI (các module khác import trực tiếp)
# ══════════════════════════════════════════════════════════════

# 8 nhóm intent (thứ tự khớp nhãn của model phobert_multitask_a100.pt — sorted alphabetically).
# Không thêm/bớt khi chưa train lại.
VALID_INTENTS: List[str] = [
    "admin_scenario",       # admin_planning + to_chuc_su_kien_cong + hoa_giai_van_dong + document_meta_relation
    "comparison",           # so_sanh_van_ban
    "document_generation",  # soan_thao_van_ban + tao_bao_cao
    "legal_explanation",    # giai_thich_quy_dinh + hoi_dap_chung
    "legal_lookup",         # article_query + tra_cuu_van_ban + trich_xuat_van_ban + can_cu_phap_ly
    "procedure",            # huong_dan_thu_tuc + kiem_tra_ho_so
    "summarization",        # tom_tat_van_ban
    "violation",            # xu_ly_vi_pham_hanh_chinh + kiem_tra_thanh_tra
]

# INTENT_MAPPING: nhãn 18 cũ → 8 nhóm mới (dùng bởi LEGACY_INTENT_ALIASES và normalize)
INTENT_MAPPING: Dict[str, str] = {
    "article_query": "legal_lookup",
    "tra_cuu_van_ban": "legal_lookup",
    "trich_xuat_van_ban": "legal_lookup",
    "can_cu_phap_ly": "legal_lookup",
    "giai_thich_quy_dinh": "legal_explanation",
    "hoi_dap_chung": "legal_explanation",
    "huong_dan_thu_tuc": "procedure",
    "kiem_tra_ho_so": "procedure",
    "xu_ly_vi_pham_hanh_chinh": "violation",
    "kiem_tra_thanh_tra": "violation",
    "so_sanh_van_ban": "comparison",
    "tom_tat_van_ban": "summarization",
    "soan_thao_van_ban": "document_generation",
    "tao_bao_cao": "document_generation",
    "admin_planning": "admin_scenario",
    "to_chuc_su_kien_cong": "admin_scenario",
    "hoa_giai_van_dong": "admin_scenario",
    "document_meta_relation": "admin_scenario",
}

# Alias nhãn cũ (pre-mapping + legacy 23-label era) → nhãn mới 8 nhóm
LEGACY_INTENT_ALIASES: Dict[str, str] = {
    # 18 fine-grained → 8 nhóm
    **INTENT_MAPPING,
    # Legacy 23-label era
    "document_metadata": "admin_scenario",
    "document_relation": "admin_scenario",
    "thu_tuc_hanh_chinh": "procedure",
    "bao_ve_xa_hoi": "legal_explanation",
    "bao_ton_phat_trien": "legal_explanation",
    "program_goal": "legal_explanation",
}

# Nhóm intent cấp xã (commune-level) — dùng bởi rag_chain_v2 + commune arbiter
COMMUNE_LEVEL_INTENTS: frozenset = frozenset({
    "violation",
    "admin_scenario",
    "procedure",
    "legal_explanation",
})


def normalize_legacy_intent(intent: Optional[str]) -> str:
    """Chuẩn hoá intent từ LLM / log cũ; giữ nguyên ``nan``."""
    if not intent:
        return "hoi_dap_chung"
    if intent == "nan":
        return "nan"
    return LEGACY_INTENT_ALIASES.get(intent, intent)


def _finalize_detector_tuple(det: str, conf: float) -> Tuple[str, float]:
    """Chuẩn hoá nhãn classifier/semantic/structural; ``nan`` giữ nguyên."""
    if det == "nan":
        return "nan", conf
    nd = normalize_legacy_intent(det)
    if nd == "nan":
        return "nan", conf
    if nd not in VALID_INTENTS:
        return "hoi_dap_chung", min(float(conf), 0.35)
    return nd, float(conf)


# ══════════════════════════════════════════════════════════════
# LAYER 3: STRUCTURAL — app/intent_patterns/routing.yaml
# ══════════════════════════════════════════════════════════════


def _detect_structural(query: str) -> Optional[Tuple[str, float]]:
    """Regex structural từ ``intent_patterns/routing.yaml`` (fallback)."""
    hit = structural_match(query.lower().strip())
    if hit:
        intent, confidence = hit
        intent, confidence = _finalize_detector_tuple(intent, confidence)
        log.debug("Structural (YAML) [%s] conf=%.2f", intent, confidence)
        return intent, confidence
    return None


def _merged_intent_prototypes() -> Dict[str, List[str]]:
    extra = get_prototype_sentences_extra()
    out = {k: list(v) for k, v in INTENT_PROTOTYPES.items()}
    for k, sentences in extra.items():
        out.setdefault(k, [])
        out[k].extend(sentences)
    return out


# ══════════════════════════════════════════════════════════════
# LAYER 2: SEMANTIC SIMILARITY — Tier 1
# ~5 prototype sentences per intent, pre-embedded, cosine match
# ══════════════════════════════════════════════════════════════

INTENT_PROTOTYPES: Dict[str, List[str]] = {
    "legal_lookup": [
        # tra_cuu_van_ban
        "Tìm văn bản pháp luật về quản lý lễ hội",
        "Có văn bản nào quy định về quảng cáo không?",
        "Tra cứu các nghị định liên quan đến văn hóa",
        "Văn bản nào quy định về lễ hội dân gian?",
        "Tìm kiếm quy định pháp luật về tôn giáo tín ngưỡng",
        # article_query
        "Điều 47 Luật Di sản văn hóa quy định gì?",
        "Nội dung khoản 2 Điều 6 Luật Đầu tư 2025",
        "Điều 9 Nghị định 144 nói về vấn đề gì?",
        "Quy định cụ thể tại Điều 15 Luật Quảng cáo",
        "Mục 3 Chương II Luật Thư viện quy định thế nào?",
        # trich_xuat_van_ban
        "Các ngành nghề cấm đầu tư kinh doanh theo Luật Đầu tư",
        "Liệt kê tất cả quyền của nhà đầu tư nước ngoài",
        "Danh sách các hành vi bị cấm trong Luật Quảng cáo",
        "Các trường hợp được miễn giảm thuế theo Luật Đầu tư",
        "Những nghĩa vụ của tổ chức kinh doanh karaoke",
        # can_cu_phap_ly
        "Căn cứ pháp lý của kế hoạch quản lý di tích",
        "Quyết định này dựa trên luật nào?",
        "Cơ sở pháp lý để ban hành quy chế quản lý lễ hội",
        "Khi Luật Đầu tư mâu thuẫn Luật Doanh nghiệp, áp dụng luật nào?",
        "Căn cứ vào đâu để xử phạt vi phạm hành chính?",
    ],
    "legal_explanation": [
        # giai_thich_quy_dinh
        "Chính sách bảo trợ xã hội đối với người cao tuổi là gì?",
        "An sinh xã hội bao gồm những gì?",
        "Quyền của người khuyết tật theo quy định pháp luật",
        "Thế nào là di sản văn hóa phi vật thể?",
        "Giải thích quy định về quản lý hoạt động karaoke",
        "Mục tiêu chương trình mục tiêu quốc gia nông thôn mới?",
        "Đề án phát triển văn hóa 2030 nhằm mục đích gì?",
        "Trẻ em bị bố đánh đập hàng ngày, cần can thiệp ngay",
        "Người già bị con cái bỏ rơi không chăm sóc",
        "Ngôi chùa cổ bị hư hỏng sau bão, cần trùng tu",
        "Bảo tồn làng nghề truyền thống đan lát",
        # hoi_dap_chung (những câu không thuộc scope pháp luật)
        "Xin chào",
        "Bạn là ai?",
        "Hệ thống này làm được gì?",
        "Cảm ơn bạn",
        "Tạm biệt",
    ],
    "procedure": [
        # huong_dan_thu_tuc
        "Thủ tục đăng ký kinh doanh gồm mấy bước?",
        "Hồ sơ xin cấp phép xây dựng cần giấy tờ gì?",
        "Quy trình đăng ký hộ khẩu thường trú",
        "Các bước xin cấp giấy chứng nhận quyền sử dụng đất",
        "Làm sao để đăng ký kết hôn?",
        "Thủ tục xin phép tổ chức lễ hội dân gian cấp xã",
        "Hồ sơ đăng ký sinh hoạt tôn giáo tập trung",
        "Thủ tục xin cấp phép tu bổ ngôi đình cổ",
        "Đăng ký hoạt động biểu diễn nghệ thuật tại xã",
        "Mẫu đơn xin phép có tải ở đâu?",
        # kiem_tra_ho_so
        "Tôi đã nộp giấy đề nghị và điều lệ công ty, còn thiếu gì?",
        "Hồ sơ của tôi đã đủ chưa?",
        "Đã nộp đơn xin phép và giấy tờ nhà, còn cần gì nữa?",
        "Kiểm tra xem hồ sơ xin cấp phép đã hoàn chỉnh chưa",
        "Tôi đã có giấy phép kinh doanh, còn thiếu giấy tờ nào?",
    ],
    "violation": [
        # xu_ly_vi_pham_hanh_chinh
        "Karaoke gây ồn ào quá giờ quy định, xử phạt thế nào?",
        "Biển quảng cáo vi phạm kích thước, xử lý ra sao?",
        "Cơ sở kinh doanh internet không có giấy phép, lập biên bản",
        "Quán bia hát karaoke gây mất trật tự khu dân cư",
        "Tổ chức sinh hoạt tôn giáo trái phép trên địa bàn xã",
        # kiem_tra_thanh_tra
        "Lập kế hoạch kiểm tra đột xuất các quán karaoke",
        "Tổ chức đoàn thanh tra cơ sở kinh doanh dịch vụ văn hóa",
        "Kiểm tra định kỳ các cơ sở internet trên địa bàn",
        "Rà soát giấy phép kinh doanh các cơ sở dịch vụ",
        "Thanh tra việc chấp hành quy định về quảng cáo",
    ],
    "comparison": [
        "Luật Đầu tư 2020 và 2025 khác gì nhau?",
        "So sánh Nghị định 110/2018 với Nghị định mới",
        "Điểm mới của Luật Di sản văn hóa sửa đổi 2009 so với 2001",
        "Sự khác biệt giữa hai phiên bản Luật Quảng cáo",
        "Luật Thể dục thể thao 2006 và 2018 khác nhau thế nào?",
    ],
    "summarization": [
        "Tóm tắt Luật Đầu tư 2025",
        "Nội dung chính của Nghị định 144/2020",
        "Luật Thể dục thể thao 2006 quy định những gì?",
        "Khái quát Luật Di sản văn hóa sửa đổi 2009",
        "Cho biết nội dung chính của Thông tư 13/2024",
    ],
    "document_generation": [
        # soan_thao_van_ban
        "Soạn công văn xin gia hạn giấy phép kinh doanh karaoke",
        "Viết tờ trình đề nghị cấp kinh phí tu bổ di tích",
        "Soạn thông báo về việc kiểm tra cơ sở kinh doanh",
        "Tạo mẫu quyết định xử phạt vi phạm hành chính",
        "Viết đơn xin phép tổ chức sự kiện văn hóa",
        # tao_bao_cao
        "Lập báo cáo tổng kết hoạt động văn hóa năm 2025",
        "Viết báo cáo đánh giá phong trào toàn dân đoàn kết",
        "Tạo báo cáo kết quả kiểm tra cơ sở kinh doanh",
        "Soạn báo cáo tình hình quản lý di tích trên địa bàn",
        "Lập báo cáo thống kê vi phạm hành chính lĩnh vực văn hóa",
    ],
    "admin_scenario": [
        # admin_planning
        "Xây dựng kế hoạch quản lý di tích trên địa bàn xã năm 2025",
        "Phân bổ nhân sự cho bộ phận văn hóa xã hội",
        "Lập phương án triển khai quản lý lễ hội trên địa bàn",
        "Kế hoạch giám sát hoạt động kinh doanh dịch vụ văn hóa",
        "Đề xuất biện pháp quản lý hoạt động quảng cáo trên địa bàn xã",
        # to_chuc_su_kien_cong
        "Tổ chức Đại hội Thể dục thể thao cấp xã",
        "Lên kế hoạch biểu diễn văn nghệ chào mừng ngày lễ",
        "Tổ chức hội thi tìm hiểu pháp luật cho thanh niên",
        "Chuẩn bị lễ hội truyền thống hàng năm của xã",
        "Kế hoạch tổ chức giải bóng chuyền cấp thôn",
        # hoa_giai_van_dong
        "Hai hàng xóm tranh chấp ranh giới đất, cần hòa giải",
        "Vận động người dân thực hiện nếp sống văn minh trong lễ hội",
        "Hòa giải mâu thuẫn tiếng ồn giữa hai gia đình",
        "Tuyên truyền phổ biến pháp luật cho nhân dân",
        "Vận động cộng đồng chấp hành hương ước thôn bản",
        # document_meta_relation
        "Nghị định 36/2019 do cơ quan nào ban hành?",
        "Luật Di sản văn hóa có hiệu lực từ ngày nào?",
        "Ai ký ban hành Thông tư 13/2024?",
        "Luật nào sửa đổi Luật Di sản văn hóa 2001?",
        "Nghị định 36/2019 thay thế nghị định nào?",
    ],
}

# ── Prototype embedding index (built at warmup) ─────────────

_proto_intents: List[str] = []       # intent label for each prototype row
_proto_matrix: Optional[np.ndarray] = None  # (N, dim) normalized


def _build_prototype_index() -> None:
    """Pre-embed all prototypes into a single matrix for fast cosine search."""
    global _proto_intents, _proto_matrix

    from app.pipeline.embedding import embed_texts

    sentences: List[str] = []
    labels: List[str] = []
    merged = _merged_intent_prototypes()
    for intent, protos in merged.items():
        for s in protos:
            sentences.append(s)
            labels.append(intent)

    if not sentences:
        log.warning("No prototypes to embed")
        return

    t0 = time.monotonic()
    _proto_matrix = embed_texts(sentences)
    _proto_intents = labels
    elapsed = (time.monotonic() - t0) * 1000
    log.info(
        "Intent prototype index built: %d sentences, %d intent keys (%.0f ms)",
        len(sentences), len(merged), elapsed,
    )


def warmup_intent_index() -> None:
    """Call at app startup (after embedding model loaded)."""
    try:
        _build_prototype_index()
    except Exception as exc:
        log.error("Failed to build intent prototype index: %s", exc)


def _detect_semantic(query: str) -> Optional[Tuple[str, float]]:
    """Layer 2: Semantic similarity against prototype embeddings.

    Returns (intent, confidence) or None.
    Confidence = scaled cosine similarity mapped to [0, 1].
    """
    if _proto_matrix is None or len(_proto_intents) == 0:
        return None

    from app.pipeline.embedding import embed_texts

    q_vec = embed_texts([query])  # (1, dim) normalized
    if q_vec.size == 0:
        return None

    sims = (q_vec @ _proto_matrix.T).flatten()  # cosine similarities

    # Per-intent: take max similarity across all prototypes for that intent
    intent_scores: Dict[str, float] = {}
    for i, sim in enumerate(sims):
        label = _proto_intents[i]
        if label not in intent_scores or sim > intent_scores[label]:
            intent_scores[label] = float(sim)

    if not intent_scores:
        return None

    best_intent = max(intent_scores, key=intent_scores.get)  # type: ignore[arg-type]
    best_sim = intent_scores[best_intent]

    # Compute gap to runner-up for disambiguation
    sorted_scores = sorted(intent_scores.values(), reverse=True)
    gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else sorted_scores[0]

    # Map cosine similarity → confidence
    # sim >= 0.72 and gap >= 0.06 → high confidence
    # sim >= 0.55 but ambiguous     → low confidence → defer to LLM
    if best_sim >= 0.72 and gap >= 0.06:
        confidence = round(0.70 + (best_sim - 0.72) * 1.5, 2)
        confidence = min(max(confidence, 0.70), 0.93)
    elif best_sim >= 0.60 and gap >= 0.08:
        confidence = round(0.55 + (best_sim - 0.60) * 1.2, 2)
        confidence = min(max(confidence, 0.55), 0.72)
    else:
        return None  # defer to LLM

    best_intent, confidence = _finalize_detector_tuple(best_intent, confidence)
    log.info(
        "Semantic match: %s (sim=%.3f, gap=%.3f, conf=%.2f)",
        best_intent, best_sim, gap, confidence,
    )
    return best_intent, confidence


# ══════════════════════════════════════════════════════════════
# LAYER 3: ZERO-SHOT LLM CLASSIFICATION — Tier 2
# ══════════════════════════════════════════════════════════════

INTENT_SEMANTIC_DESCRIPTIONS: Dict[str, str] = {
    "legal_lookup": (
        "Tra cứu văn bản pháp luật, Điều/Khoản cụ thể, căn cứ pháp lý, "
        "liệt kê danh mục từ văn bản (gộp: article_query, tra_cuu, trich_xuat, can_cu_phap_ly)"
    ),
    "legal_explanation": (
        "Giải thích quy định, chính sách, bảo vệ xã hội/di sản, chào hỏi hoặc câu chung chung "
        "(gộp: giai_thich_quy_dinh + hoi_dap_chung)"
    ),
    "procedure": (
        "Hướng dẫn thủ tục, hồ sơ, mẫu đơn, kiểm tra hồ sơ còn thiếu gì "
        "(gộp: huong_dan_thu_tuc + kiem_tra_ho_so)"
    ),
    "violation": (
        "Xử lý vi phạm hành chính, kế hoạch/đoàn thanh tra kiểm tra cơ sở "
        "(gộp: xu_ly_vi_pham_hanh_chinh + kiem_tra_thanh_tra)"
    ),
    "comparison": "So sánh, đối chiếu nội dung hai hoặc nhiều văn bản pháp luật",
    "summarization": "Tóm tắt, khái quát nội dung một văn bản đã biết tên/số hiệu",
    "document_generation": (
        "Soạn công văn, tờ trình, biên bản, quyết định, thông báo, báo cáo hành chính "
        "(gộp: soan_thao_van_ban + tao_bao_cao)"
    ),
    "admin_scenario": (
        "Lập kế hoạch quản lý, tổ chức sự kiện, hòa giải tranh chấp, vận động nhân dân, "
        "tra cứu metadata/quan hệ văn bản (gộp: admin_planning + to_chuc_su_kien_cong + "
        "hoa_giai_van_dong + document_meta_relation)"
    ),
}

_INTENT_CATALOG: Dict[str, Dict] = {
    "legal_lookup": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["legal_lookup"],
        "vi_du": [
            "Điều 47 Luật Di sản văn hóa quy định gì?",
            "Văn bản nào quy định về lễ hội?",
            "Căn cứ pháp lý để xử phạt vi phạm?",
            "Liệt kê ngành nghề cấm đầu tư",
        ],
        "khong_phai": "Tóm tắt cả văn bản → summarization; So sánh hai văn bản → comparison",
    },
    "legal_explanation": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["legal_explanation"],
        "vi_du": [
            "Chính sách bảo trợ xã hội là gì?",
            "Mục tiêu chương trình nông thôn mới?",
            "Xin chào",
            "Bảo tồn di tích theo luật quy định ra sao?",
        ],
        "khong_phai": "Thủ tục nộp hồ sơ → procedure",
    },
    "procedure": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["procedure"],
        "vi_du": [
            "Thủ tục xin phép lễ hội cấp xã?",
            "Mẫu đơn nộp ở đâu?",
            "Hồ sơ đã nộp còn thiếu gì?",
        ],
        "khong_phai": "",
    },
    "violation": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["violation"],
        "vi_du": [
            "Karaoke ồn quá giờ, xử phạt thế nào?",
            "Kế hoạch kiểm tra đột xuất karaoke",
            "Đoàn thanh tra cơ sở internet",
        ],
        "khong_phai": "",
    },
    "comparison": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["comparison"],
        "vi_du": ["Luật Đầu tư 2020 và 2025 khác gì?", "Đối chiếu hai nghị định"],
        "khong_phai": "VB A thay thế VB B → admin_scenario (document_meta_relation)",
    },
    "summarization": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["summarization"],
        "vi_du": ["Tóm tắt Luật Đầu tư 2025", "Nội dung chính NĐ 144"],
        "khong_phai": "Mục tiêu CT quốc gia → legal_explanation",
    },
    "document_generation": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["document_generation"],
        "vi_du": [
            "Soạn công văn xin gia hạn giấy phép",
            "Viết tờ trình cấp kinh phí",
            "Lập báo cáo tổng kết năm",
        ],
        "khong_phai": "",
    },
    "admin_scenario": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["admin_scenario"],
        "vi_du": [
            "Kế hoạch quản lý di tích 2025",
            "Tổ chức đại hội TDTT xã",
            "Hàng xóm tranh chấp đất, hòa giải?",
            "Nghị định 36/2019 do ai ban hành?",
        ],
        "khong_phai": "",
    },
}


def _build_classification_prompt(query: str) -> str:
    """Build zero-shot LLM classification prompt."""
    catalog_lines = []
    for intent, info in _INTENT_CATALOG.items():
        line = f"• {intent}: {info['mo_ta']}"
        if info["vi_du"]:
            examples = " | ".join(f'"{e}"' for e in info["vi_du"])
            line += f"\n  VD: {examples}"
        if info["khong_phai"]:
            line += f"\n  ⚠️ {info['khong_phai']}"
        catalog_lines.append(line)

    catalog_text = "\n\n".join(catalog_lines)

    prompt = f"""Bạn là hệ thống phân loại ý định (intent classifier) cho chatbot hành chính Việt Nam dành cho cán bộ cấp xã.

## DANH SÁCH INTENT (8 nhóm)

{catalog_text}

## QUY TẮC PHÂN LOẠI (CHỈ 8 INTENT)

1. legal_lookup: tra Điều/Khoản cụ thể, tìm văn bản theo chủ đề, trích xuất danh mục, căn cứ pháp lý.
2. legal_explanation: giải thích quy định/chính sách, bảo vệ xã hội/di sản, chào hỏi thông thường.
3. procedure: thủ tục, hồ sơ, mẫu đơn, kiểm tra hồ sơ còn thiếu gì.
4. violation: xử phạt vi phạm, kế hoạch/đoàn thanh tra kiểm tra cơ sở.
5. comparison: so sánh nội dung hai hay nhiều văn bản khác nhau.
6. summarization: tóm tắt nội dung tổng thể của một văn bản.
7. document_generation: soạn công văn, tờ trình, biên bản, quyết định, báo cáo.
8. admin_scenario: kế hoạch quản lý, tổ chức sự kiện, hòa giải, vận động dân, metadata/quan hệ văn bản.
Chọn đúng một intent trong danh sách; ưu tiên ý chính của câu hỏi.

## CÂU HỎI

"{query}"

## OUTPUT (ĐÚNG 2 dòng)

REASON: [1 câu giải thích ngắn]
INTENT: [tên_intent]

Chỉ chọn: {", ".join(VALID_INTENTS)}"""

    return prompt


def _parse_llm_output(raw: str) -> Tuple[Optional[str], float]:
    """Parse LLM response → (intent, confidence)."""
    if not raw:
        return None, 0.0

    raw_lower = raw.lower().strip()

    intent_match = re.search(r"intent:\s*([a-z0-9_]+)", raw_lower)
    if intent_match:
        candidate = normalize_legacy_intent(intent_match.group(1).strip())
        if candidate == "nan":
            return "nan", 0.85
        if candidate in VALID_INTENTS:
            return candidate, 0.85

    for intent in VALID_INTENTS:
        if intent in raw_lower:
            return intent, 0.72

    return None, 0.0


# ══════════════════════════════════════════════════════════════
# TIER 3: AUTO-INDEX FROM DB
# Extract representative sentences from newly uploaded documents
# and merge into the prototype embedding index.
# ══════════════════════════════════════════════════════════════

_TIER3_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "violation": ["xử phạt", "vi phạm hành chính", "mức phạt", "biên bản", "hình thức xử phạt",
                  "kiểm tra", "thanh tra", "rà soát", "giám sát"],
    "admin_scenario": ["lễ hội", "thể dục thể thao", "biểu diễn nghệ thuật", "sự kiện văn hóa",
                       "hòa giải", "tranh chấp", "vận động", "nếp sống văn minh"],
    "legal_explanation": [
        "chính sách", "quyền lợi", "chế độ", "bảo trợ xã hội", "an sinh",
        "bạo lực gia đình", "bảo vệ trẻ em", "di tích", "di sản", "bảo tồn",
        "mục tiêu chương trình",
    ],
    "procedure": ["thủ tục", "đăng ký", "cấp phép", "hồ sơ", "một cửa", "mẫu đơn", "sinh hoạt tôn giáo"],
    "legal_lookup": ["cấm", "nghiêm cấm", "danh mục", "hành vi bị cấm", "điều khoản", "quy định"],
}


async def auto_index_from_document(document_id: int) -> int:
    """Tier 3: Extract representative sentences from a document in DB
    and add them to the prototype index.

    Called after document upload/ingestion.
    Returns number of new prototypes added.
    """
    global _proto_intents, _proto_matrix

    from sqlalchemy import select
    from app.database.session import get_db_context
    from app.database.models import Article, Document
    from app.pipeline.embedding import embed_texts

    new_sentences: List[Tuple[str, str]] = []  # (sentence, intent)

    try:
        async with get_db_context() as db:
            doc_result = await db.execute(
                select(Document.title, Document.doc_number).where(Document.id == document_id)
            )
            doc_row = doc_result.first()
            if not doc_row:
                log.warning("Auto-index: document_id=%d not found", document_id)
                return 0

            doc_title = doc_row.title or doc_row.doc_number or ""

            art_result = await db.execute(
                select(Article.title, Article.content, Article.article_number)
                .where(Article.document_id == document_id)
                .order_by(Article.id)
            )
            articles = art_result.all()

        for art_title_raw, art_content, art_num in articles:
            art_title = art_title_raw or ""
            snippet = (art_content or "")[:300].lower()

            for intent, keywords in _TIER3_INTENT_KEYWORDS.items():
                if any(kw in snippet for kw in keywords):
                    sentence = f"{art_title} theo {doc_title}" if art_title else f"Điều {art_num} {doc_title}"
                    new_sentences.append((sentence, intent))
                    break

        if not new_sentences:
            log.info("Auto-index: no matching articles in doc_id=%d", document_id)
            return 0

        sentences = [s for s, _ in new_sentences]
        labels = [l for _, l in new_sentences]
        new_embeddings = embed_texts(sentences)

        if _proto_matrix is not None:
            _proto_matrix = np.vstack([_proto_matrix, new_embeddings])
            _proto_intents.extend(labels)
        else:
            _proto_matrix = new_embeddings
            _proto_intents = labels

        log.info(
            "Auto-indexed %d prototypes from doc_id=%d (total: %d)",
            len(new_sentences), document_id, len(_proto_intents),
        )
        return len(new_sentences)

    except Exception as exc:
        log.error("Auto-index failed for doc_id=%d: %s", document_id, exc)
        return 0


def get_index_stats() -> Dict[str, Any]:
    """Return current prototype index statistics."""
    if _proto_matrix is None:
        return {"total_prototypes": 0, "intents_covered": 0, "index_loaded": False}

    from collections import Counter
    counts = Counter(_proto_intents)
    return {
        "total_prototypes": len(_proto_intents),
        "intents_covered": len(counts),
        "index_loaded": True,
        "per_intent": dict(counts),
    }


# ══════════════════════════════════════════════════════════════
# CONFIDENCE CALIBRATION
# ══════════════════════════════════════════════════════════════

def _calibrate_confidence(
    intent: str,
    raw_confidence: float,
    query: str,
    method: str,
) -> float:
    if intent == "nan":
        return round(min(float(raw_confidence), 0.95), 2)

    conf = raw_confidence

    word_count = len(query.split())
    if word_count <= 2 and method == "llm":
        conf = max(conf - 0.10, 0.40)
    elif word_count <= 4 and method == "llm":
        conf = max(conf - 0.05, 0.50)

    if intent in COMMUNE_LEVEL_INTENTS and method in ("llm", "semantic", "classifier"):
        _BOOST_SIGNALS = [
            "địa bàn", "xã", "phường", "thôn", "ông/bà", "tham mưu",
            "ra quân", "vi phạm", "xử lý", "tình huống",
        ]
        q_lower = query.lower()
        if any(sig in q_lower for sig in _BOOST_SIGNALS):
            conf = min(conf + 0.05, 0.95)

    return round(conf, 2)


def _detect_intent_classifier(query: str) -> Optional[Tuple[str, float]]:
    """Layer 2: mô hình PhoBERT cục bộ; None nếu tắt / lỗi / thấp hơn ngưỡng."""
    try:
        from app.services.intent_model_classifier import classify_intent_sync

        return classify_intent_sync(query)
    except Exception as exc:
        log.debug("Intent classifier skipped: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

async def detect_intent_llm(query: str) -> Tuple[str, float]:
    """Tier 2: Zero-shot LLM classification for complex/ambiguous queries."""
    from app.services.llm_client import generate

    prompt = _build_classification_prompt(query)

    try:
        t0 = time.monotonic()
        raw = await generate(prompt, temperature=0.0)
        latency_ms = (time.monotonic() - t0) * 1000
        log.info("LLM classification latency: %.0fms", latency_ms)

        intent, confidence = _parse_llm_output(raw)
        if intent:
            intent, confidence = _finalize_detector_tuple(intent, confidence)
            log.info("LLM classified: %s (conf=%.2f)", intent, confidence)
            return intent, confidence

        log.warning("LLM output parse failed. Raw: %s", raw[:200])
        return "hoi_dap_chung", 0.30

    except Exception as exc:
        log.error("LLM intent detection error: %s", exc)
        return "hoi_dap_chung", 0.30


async def detect_intent(query: str) -> Dict[str, object]:
    """Pipeline intent: guard → structural → classifier (tuỳ cấu hình) → semantic → LLM.

    Returns: {"intent": str, "confidence": float, "method": str}
    """
    # ── Layer 0: Guard ──────────────────────────────────────
    if not query or not query.strip():
        log.warning("Empty query received")
        return {"intent": "hoi_dap_chung", "confidence": 0.10, "method": "guard"}

    query = query.strip()

    if len(query) < 3:
        log.warning("Query too short: '%s'", query)
        return {"intent": "hoi_dap_chung", "confidence": 0.10, "method": "guard"}

    if not re.search(r'[a-zA-ZÀ-ỹ]', query):
        log.warning("Noise query (no letters): '%s'", query)
        return {"intent": "hoi_dap_chung", "confidence": 0.15, "method": "guard"}

    # ── Layer 1: Fine-tuned classifier (PhoBERT) ─────────────
    clf_result = _detect_intent_classifier(query)
    if clf_result:
        intent, confidence = clf_result
        calibrated = _calibrate_confidence(intent, confidence, query, "classifier")
        log.info("Classifier intent: %s (conf=%.2f)", intent, calibrated)
        return {"intent": intent, "confidence": calibrated, "method": "classifier"}

    # ── Layer 2: Semantic similarity (prototypes) ─────────────
    semantic_result = _detect_semantic(query)
    if semantic_result:
        intent, confidence = semantic_result
        calibrated = _calibrate_confidence(intent, confidence, query, "semantic")
        log.info("Semantic intent: %s (conf=%.2f)", intent, calibrated)
        return {"intent": intent, "confidence": calibrated, "method": "semantic"}

    # ── Layer 3: Structural regex (YAML fallback) ───────────
    structural_result = _detect_structural(query)
    if structural_result:
        intent, confidence = structural_result
        calibrated = _calibrate_confidence(intent, confidence, query, "structural")
        log.info("Structural intent: %s (conf=%.2f)", intent, calibrated)
        return {"intent": intent, "confidence": calibrated, "method": "structural"}

    # ── Layer 4: Zero-shot LLM ───────────────────────────────
    log.info("Calling LLM for intent classification: '%s'", query[:60])
    intent, confidence = await detect_intent_llm(query)
    calibrated = _calibrate_confidence(intent, confidence, query, "llm")

    return {"intent": intent, "confidence": calibrated, "method": "llm"}


# ══════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ══════════════════════════════════════════════════════════════

def detect_intent_rule_based(query: str) -> Tuple[str, float]:
    """Classifier → structural (YAML) → semantic; không gọi LLM.

    Dùng bởi `query_intent.compute_intent_bundle` / cờ RAG.
    """
    if not query or not query.strip():
        return "hoi_dap_chung", 0.10

    q = query.strip()

    clf = _detect_intent_classifier(q)
    if clf:
        return _finalize_detector_tuple(*clf)

    structural = _detect_structural(q)
    if structural:
        return structural

    # Heuristic: tình huống quản trị cấp xã/phường (thường bị semantic kéo về legal_explanation)
    ql = q.lower()
    if re.search(r"\b(xã|thôn|phường|ubnd\s*xã|thôn\s*trưởng)\b", ql) and re.search(
        r"(xử\s*lý|can\s*thiệp|chỉ\s*đạo|giải\s*quyết|cơ\s*quan\s+nào|ra\s*xã|lên\s*xã)",
        ql,
    ):
        return "admin_scenario", 0.90

    semantic = _detect_semantic(q)
    if semantic:
        return semantic

    return "hoi_dap_chung", 0.30


# ══════════════════════════════════════════════════════════════
# RAG FLAGS (v3) — ánh xạ intent → luồng rag_chain_v2
# ══════════════════════════════════════════════════════════════

# Cờ RAG — ánh xạ 8 nhóm intent → 3 cờ cho rag_chain_v2.
# use_multi_article: True cho mọi intent NGOẠI TRỪ legal_lookup (tra cứu chính xác một điều).
_RAG_LEGAL_LOOKUP_INTENTS: frozenset = frozenset({
    "legal_lookup",
})

_RAG_NEEDS_EXPANSION_INTENTS: frozenset = frozenset({
    "legal_explanation",
    "comparison",
    "violation",
    "admin_scenario",
    "summarization",
    "document_generation",
})

def map_intent_to_rag_flags(intent: str) -> Dict[str, bool]:
    """Ánh xạ intent → 3 cờ RAG. Intent không trong VALID_INTENTS → cả ba False.

    Nhận cả nhãn fine-grained cũ (``article_query``, ``xu_ly_vi_pham_hanh_chinh``, …)
    qua ``normalize_legacy_intent`` trước khi áp dụng quy tắc.

    Quy tắc ``use_multi_article``: bật cho mọi intent hợp lệ **trừ** ``_RAG_LEGAL_LOOKUP_INTENTS``
    (tra Điều/metadata/căn cứ/trích xuất — ưu tiên ngữ cảnh thu hẹp).
    """
    if not intent or not str(intent).strip():
        return {
            "is_legal_lookup": False,
            "use_multi_article": False,
            "needs_expansion": False,
        }
    norm = normalize_legacy_intent(str(intent).strip())
    if norm == "nan" or norm not in VALID_INTENTS:
        return {
            "is_legal_lookup": False,
            "use_multi_article": False,
            "needs_expansion": False,
        }
    intent = norm
    legal_lookup = intent in _RAG_LEGAL_LOOKUP_INTENTS
    return {
        "is_legal_lookup": legal_lookup,
        "needs_expansion": intent in _RAG_NEEDS_EXPANSION_INTENTS,
        "use_multi_article": not legal_lookup,
    }


def get_rag_intents(query: str) -> Dict[str, bool]:
    """Cờ RAG — đồng bộ với ``query_intent.compute_intent_bundle``."""
    try:
        from app.services.query_intent import compute_rag_flags_for_query

        flags = compute_rag_flags_for_query(query or "")
        log.debug("get_rag_intents (query_intent): %s", flags)
        return flags
    except Exception as exc:
        log.warning("get_rag_intents failed: %s", exc)
        return {
            "is_legal_lookup": False,
            "use_multi_article": False,
            "needs_expansion": False,
        }


async def get_rag_intents_async(query: str) -> Dict[str, bool]:
    """Giống get_rag_intents nhưng dùng full `detect_intent` (có LLM khi cần)."""
    try:
        result = await detect_intent(query or "")
        intent = str(result.get("intent", "hoi_dap_chung"))
        return map_intent_to_rag_flags(intent)
    except Exception as exc:
        log.warning("get_rag_intents_async failed: %s", exc)
        return {
            "is_legal_lookup": False,
            "use_multi_article": False,
            "needs_expansion": False,
        }
