"""
Intent detection — guard → PhoBERT → prototype embedding → structural (YAML) → LLM.

Pipeline (dừng khi một tầng trả intent đủ tin cậy):

  Layer 0: Guard — câu rỗng / quá ngắn / không có chữ cái
  Layer 1: Classifier — PhoBERT trong ``app/intent_model`` (``INTENT_MODEL_ENABLED``)
  Layer 2: Semantic — cosine với prototype (SentenceTransformer); câu mẫu + ``intent_patterns/routing.yaml``
  Layer 3: Structural — regex fallback từ YAML (``INTENT_PATTERNS_YAML``), không ưu tiên trước ML
  Layer 4: LLM — zero-shot khi các tầng trên không quyết định

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

# 18 nhãn (thứ tự khớp lớp 0..17 của model PhoBERT fine-tune). Không thêm/bớt khi chưa train lại.
VALID_INTENTS: List[str] = [
    "admin_planning",
    "article_query",
    "can_cu_phap_ly",
    "document_meta_relation",
    "giai_thich_quy_dinh",
    "hoa_giai_van_dong",
    "hoi_dap_chung",
    "huong_dan_thu_tuc",
    "kiem_tra_ho_so",
    "kiem_tra_thanh_tra",
    "so_sanh_van_ban",
    "soan_thao_van_ban",
    "tao_bao_cao",
    "tom_tat_van_ban",
    "to_chuc_su_kien_cong",
    "tra_cuu_van_ban",
    "trich_xuat_van_ban",
    "xu_ly_vi_pham_hanh_chinh",
]

# Alias nhãn cũ (23 intent) → nhãn mới — dùng sau LLM / dữ liệu huấn luyện cũ
LEGACY_INTENT_ALIASES: Dict[str, str] = {
    "document_metadata": "document_meta_relation",
    "document_relation": "document_meta_relation",
    "thu_tuc_hanh_chinh": "huong_dan_thu_tuc",
    "bao_ve_xa_hoi": "giai_thich_quy_dinh",
    "bao_ton_phat_trien": "giai_thich_quy_dinh",
    "program_goal": "giai_thich_quy_dinh",
}

COMMUNE_LEVEL_INTENTS: frozenset = frozenset({
    "xu_ly_vi_pham_hanh_chinh",
    "kiem_tra_thanh_tra",
    "hoa_giai_van_dong",
    "to_chuc_su_kien_cong",
    "huong_dan_thu_tuc",
    "giai_thich_quy_dinh",
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
    "tra_cuu_van_ban": [
        "Tìm văn bản pháp luật về quản lý lễ hội",
        "Có văn bản nào quy định về quảng cáo không?",
        "Tra cứu các nghị định liên quan đến văn hóa",
        "Văn bản nào quy định về lễ hội dân gian?",
        "Tìm kiếm quy định pháp luật về tôn giáo tín ngưỡng",
    ],
    "article_query": [
        "Điều 47 Luật Di sản văn hóa quy định gì?",
        "Nội dung khoản 2 Điều 6 Luật Đầu tư 2025",
        "Điều 9 Nghị định 144 nói về vấn đề gì?",
        "Quy định cụ thể tại Điều 15 Luật Quảng cáo",
        "Mục 3 Chương II Luật Thư viện quy định thế nào?",
    ],
    "document_meta_relation": [
        "Nghị định 36/2019 do cơ quan nào ban hành?",
        "Luật Di sản văn hóa có hiệu lực từ ngày nào?",
        "Ai ký ban hành Thông tư 13/2024?",
        "Luật nào sửa đổi Luật Di sản văn hóa 2001?",
        "Nghị định 36/2019 thay thế nghị định nào?",
        "Văn bản nào bãi bỏ Thông tư 04/2010?",
        "Quyết định 706 có còn hiệu lực không?",
        "Luật Đầu tư 2025 sửa đổi bổ sung những luật nào?",
    ],
    "can_cu_phap_ly": [
        "Căn cứ pháp lý của kế hoạch quản lý di tích",
        "Quyết định này dựa trên luật nào?",
        "Cơ sở pháp lý để ban hành quy chế quản lý lễ hội",
        "Khi Luật Đầu tư mâu thuẫn Luật Doanh nghiệp, áp dụng luật nào?",
        "Căn cứ vào đâu để xử phạt vi phạm hành chính?",
    ],
    "soan_thao_van_ban": [
        "Soạn công văn xin gia hạn giấy phép kinh doanh karaoke",
        "Viết tờ trình đề nghị cấp kinh phí tu bổ di tích",
        "Soạn thông báo về việc kiểm tra cơ sở kinh doanh",
        "Tạo mẫu quyết định xử phạt vi phạm hành chính",
        "Viết đơn xin phép tổ chức sự kiện văn hóa",
    ],
    "giai_thich_quy_dinh": [
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
    ],
    "huong_dan_thu_tuc": [
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
    ],
    "kiem_tra_ho_so": [
        "Tôi đã nộp giấy đề nghị và điều lệ công ty, còn thiếu gì?",
        "Hồ sơ của tôi đã đủ chưa?",
        "Đã nộp đơn xin phép và giấy tờ nhà, còn cần gì nữa?",
        "Kiểm tra xem hồ sơ xin cấp phép đã hoàn chỉnh chưa",
        "Tôi đã có giấy phép kinh doanh, còn thiếu giấy tờ nào?",
    ],
    "tom_tat_van_ban": [
        "Tóm tắt Luật Đầu tư 2025",
        "Nội dung chính của Nghị định 144/2020",
        "Luật Thể dục thể thao 2006 quy định những gì?",
        "Khái quát Luật Di sản văn hóa sửa đổi 2009",
        "Cho biết nội dung chính của Thông tư 13/2024",
    ],
    "so_sanh_van_ban": [
        "Luật Đầu tư 2020 và 2025 khác gì nhau?",
        "So sánh Nghị định 110/2018 với Nghị định mới",
        "Điểm mới của Luật Di sản văn hóa sửa đổi 2009 so với 2001",
        "Sự khác biệt giữa hai phiên bản Luật Quảng cáo",
        "Luật Thể dục thể thao 2006 và 2018 khác nhau thế nào?",
    ],
    "tao_bao_cao": [
        "Lập báo cáo tổng kết hoạt động văn hóa năm 2025",
        "Viết báo cáo đánh giá phong trào toàn dân đoàn kết",
        "Tạo báo cáo kết quả kiểm tra cơ sở kinh doanh",
        "Soạn báo cáo tình hình quản lý di tích trên địa bàn",
        "Lập báo cáo thống kê vi phạm hành chính lĩnh vực văn hóa",
    ],
    "trich_xuat_van_ban": [
        "Các ngành nghề cấm đầu tư kinh doanh theo Luật Đầu tư",
        "Liệt kê tất cả quyền của nhà đầu tư nước ngoài",
        "Danh sách các hành vi bị cấm trong Luật Quảng cáo",
        "Các trường hợp được miễn giảm thuế theo Luật Đầu tư",
        "Những nghĩa vụ của tổ chức kinh doanh karaoke",
    ],
    "admin_planning": [
        "Xây dựng kế hoạch quản lý di tích trên địa bàn xã năm 2025",
        "Phân bổ nhân sự cho bộ phận văn hóa xã hội",
        "Lập phương án triển khai quản lý lễ hội trên địa bàn",
        "Kế hoạch giám sát hoạt động kinh doanh dịch vụ văn hóa",
        "Đề xuất biện pháp quản lý hoạt động quảng cáo trên địa bàn xã",
    ],
    "xu_ly_vi_pham_hanh_chinh": [
        "Karaoke gây ồn ào quá giờ quy định, xử phạt thế nào?",
        "Biển quảng cáo vi phạm kích thước, xử lý ra sao?",
        "Cơ sở kinh doanh internet không có giấy phép, lập biên bản",
        "Quán bia hát karaoke gây mất trật tự khu dân cư",
        "Tổ chức sinh hoạt tôn giáo trái phép trên địa bàn xã",
    ],
    "kiem_tra_thanh_tra": [
        "Lập kế hoạch kiểm tra đột xuất các quán karaoke",
        "Tổ chức đoàn thanh tra cơ sở kinh doanh dịch vụ văn hóa",
        "Kiểm tra định kỳ các cơ sở internet trên địa bàn",
        "Rà soát giấy phép kinh doanh các cơ sở dịch vụ",
        "Thanh tra việc chấp hành quy định về quảng cáo",
    ],
    "hoa_giai_van_dong": [
        "Hai hàng xóm tranh chấp ranh giới đất, cần hòa giải",
        "Vận động người dân thực hiện nếp sống văn minh trong lễ hội",
        "Hòa giải mâu thuẫn tiếng ồn giữa hai gia đình",
        "Tuyên truyền phổ biến pháp luật cho nhân dân",
        "Vận động cộng đồng chấp hành hương ước thôn bản",
    ],
    "to_chuc_su_kien_cong": [
        "Tổ chức Đại hội Thể dục thể thao cấp xã",
        "Lên kế hoạch biểu diễn văn nghệ chào mừng ngày lễ",
        "Tổ chức hội thi tìm hiểu pháp luật cho thanh niên",
        "Chuẩn bị lễ hội truyền thống hàng năm của xã",
        "Kế hoạch tổ chức giải bóng chuyền cấp thôn",
    ],
    "hoi_dap_chung": [
        "Công thức nấu phở bò truyền thống như thế nào?",
        "Tỷ giá USD hôm nay là bao nhiêu?",
        "Viết chương trình Python đọc file Excel",
        "Kết quả bóng đá Ngoại hạng Anh đêm qua",
        "Thời tiết Hà Nội ngày mai có mưa không?",
        "Bitcoin có nên đầu tư lúc này?",
        "Cách học IELTS nhanh trong 2 tuần",
        "Xin chào",
        "Bạn là ai?",
        "Hệ thống này làm được gì?",
        "Cảm ơn bạn",
        "Tạm biệt",
        "Gợi ý khách sạn giá rẻ ở Đà Lạt",
        "Cách làm bánh flan không bị rỗ",
        "Chiến thuật mở đầu ván cờ vua",
        "So sánh iPhone với Samsung (điện thoại)",
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
    "admin_planning": "Lập kế hoạch quản lý, phân bổ nguồn lực, tổ chức thực hiện cấp địa phương",
    "article_query": "Hỏi nội dung Điều/Khoản/Mục cụ thể hoặc tra cứu theo số hiệu văn bản đã biết",
    "can_cu_phap_ly": "Hỏi căn cứ pháp lý: văn bản nào làm cơ sở, mâu thuẫn luật áp dụng thế nào",
    "document_meta_relation": "Hỏi metadata văn bản (ban hành, hiệu lực) hoặc quan hệ sửa đổi/thay thế/bãi bỏ giữa các văn bản",
    "giai_thich_quy_dinh": "Giải thích quy định, chính sách, mục tiêu chương trình; bảo vệ xã hội/di sản theo góc độ quy định (gộp program_goal, bảo vệ XH, bảo tồn)",
    "hoa_giai_van_dong": "Hòa giải tranh chấp, vận động nhân dân, tuyên truyền vận động",
    "hoi_dap_chung": "Chào hỏi hoặc câu không thuộc phạm vi pháp luật–hành chính cụ thể",
    "huong_dan_thu_tuc": "Hướng dẫn thủ tục, hồ sơ, mẫu đơn, thời hạn, nộp tại UBND/một cửa (gộp thủ tục hành chính cấp xã)",
    "kiem_tra_ho_so": "Đã nộp hồ sơ — kiểm tra còn thiếu gì, đủ chưa",
    "kiem_tra_thanh_tra": "Kế hoạch/đoàn thanh tra, kiểm tra định kỳ cơ sở",
    "so_sanh_van_ban": "So sánh, đối chiếu nội dung hai hoặc nhiều văn bản",
    "soan_thao_van_ban": "Soạn công văn, tờ trình, biên bản, quyết định, thông báo",
    "tao_bao_cao": "Lập/viết báo cáo hành chính, tổng kết",
    "tom_tat_van_ban": "Tóm tắt, khái quát nội dung một văn bản đã biết tên/số",
    "to_chuc_su_kien_cong": "Tổ chức sự kiện, lễ hội, đại hội TDTT, biểu diễn",
    "tra_cuu_van_ban": "Tìm văn bản theo chủ đề (chưa rõ điều khoản cụ thể)",
    "trich_xuat_van_ban": "Liệt kê/trích toàn bộ mục từ văn bản (danh mục cấm, quyền, nghĩa vụ)",
    "xu_ly_vi_pham_hanh_chinh": "Xử lý/xử phạt vi phạm hành chính đang xảy ra",
}

_INTENT_CATALOG: Dict[str, Dict] = {
    "admin_planning": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["admin_planning"],
        "vi_du": ["Kế hoạch quản lý di tích 2025", "Phân bổ nhân sự bộ phận VH"],
        "khong_phai": "Sự kiện → to_chuc_su_kien_cong",
    },
    "article_query": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["article_query"],
        "vi_du": ["Điều 47 Luật Di sản văn hóa quy định gì?", "Khoản 2 Điều 6 Luật Đầu tư 2025"],
        "khong_phai": "Không có Điều/Khoản cụ thể → tra_cuu_van_ban / trich_xuat_van_ban",
    },
    "can_cu_phap_ly": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["can_cu_phap_ly"],
        "vi_du": ["Căn cứ pháp lý của kế hoạch?", "Hai luật mâu thuẫn áp dụng luật nào?"],
        "khong_phai": "",
    },
    "document_meta_relation": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["document_meta_relation"],
        "vi_du": ["Nghị định 36/2019 do ai ban hành?", "Luật nào sửa Luật DSVH 2001?", "TT 04 còn hiệu lực không?"],
        "khong_phai": "So sánh nội dung chi tiết → so_sanh_van_ban",
    },
    "giai_thich_quy_dinh": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["giai_thich_quy_dinh"],
        "vi_du": [
            "Chính sách bảo trợ xã hội là gì?",
            "Mục tiêu chương trình nông thôn mới?",
            "Trẻ em bị bạo hành cần quy định can thiệp thế nào?",
            "Bảo tồn di tích theo luật quy định ra sao?",
        ],
        "khong_phai": "Thủ tục nộp hồ sơ → huong_dan_thu_tuc",
    },
    "hoa_giai_van_dong": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["hoa_giai_van_dong"],
        "vi_du": ["Hàng xóm tranh chấp đất, hòa giải?", "Vận động dân chấp hành hương ước"],
        "khong_phai": "",
    },
    "hoi_dap_chung": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["hoi_dap_chung"],
        "vi_du": ["Xin chào", "Công thức nấu ăn", "Tỷ giá USD"],
        "khong_phai": "",
    },
    "huong_dan_thu_tuc": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["huong_dan_thu_tuc"],
        "vi_du": ["Thủ tục xin phép lễ hội cấp xã?", "Mẫu đơn nộp ở đâu?", "Hồ sơ một cửa gồm gì?"],
        "khong_phai": "",
    },
    "kiem_tra_ho_so": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["kiem_tra_ho_so"],
        "vi_du": ["Đã nộp đơn, còn thiếu gì?", "Hồ sơ đủ chưa?"],
        "khong_phai": "Chưa nộp, hỏi cần gì → huong_dan_thu_tuc",
    },
    "kiem_tra_thanh_tra": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["kiem_tra_thanh_tra"],
        "vi_du": ["Kế hoạch kiểm tra đột xuất karaoke", "Đoàn thanh tra cơ sở internet"],
        "khong_phai": "Đã có vi phạm cần xử phạt → xu_ly_vi_pham_hanh_chinh",
    },
    "so_sanh_van_ban": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["so_sanh_van_ban"],
        "vi_du": ["Luật 2020 và 2025 khác gì?", "Đối chiếu hai nghị định"],
        "khong_phai": "VB A thay thế VB B → document_meta_relation",
    },
    "soan_thao_van_ban": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["soan_thao_van_ban"],
        "vi_du": ["Soạn công văn xin gia hạn", "Viết tờ trình"],
        "khong_phai": "Báo cáo → tao_bao_cao",
    },
    "tao_bao_cao": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["tao_bao_cao"],
        "vi_du": ["Lập báo cáo tổng kết năm", "Viết báo cáo phong trào"],
        "khong_phai": "Công văn ngắn → soan_thao_van_ban",
    },
    "tom_tat_van_ban": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["tom_tat_van_ban"],
        "vi_du": ["Tóm tắt Luật Đầu tư 2025", "Nội dung chính NĐ 144"],
        "khong_phai": "Mục tiêu CT quốc gia → giai_thich_quy_dinh",
    },
    "to_chuc_su_kien_cong": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["to_chuc_su_kien_cong"],
        "vi_du": ["Tổ chức đại hội TDTT xã", "Kế hoạch biểu diễn ngày lễ"],
        "khong_phai": "Kế hoạch quản lý chung → admin_planning",
    },
    "tra_cuu_van_ban": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["tra_cuu_van_ban"],
        "vi_du": ["Văn bản nào quy định về lễ hội?", "Tìm nghị định về quảng cáo"],
        "khong_phai": "Đã rõ Điều Khoản → article_query",
    },
    "trich_xuat_van_ban": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["trich_xuat_van_ban"],
        "vi_du": ["Các ngành nghề cấm đầu tư", "Liệt kê quyền nhà đầu tư"],
        "khong_phai": "Một điều cụ thể → article_query",
    },
    "xu_ly_vi_pham_hanh_chinh": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["xu_ly_vi_pham_hanh_chinh"],
        "vi_du": ["Karaoke ồn quá giờ, xử phạt?", "Quảng cáo sai, xử lý?"],
        "khong_phai": "Chỉ kế hoạch kiểm tra → kiem_tra_thanh_tra",
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

## DANH SÁCH INTENT

{catalog_text}

## QUY TẮC PHÂN LOẠI

1. trich_xuat_van_ban vs article_query: liệt kê toàn bộ mục trong luật → trich_xuat; một Điều/Khoản cụ thể → article_query.
2. document_meta_relation vs so_sanh_van_ban: ai ban hành / thay thế sửa đổi / hiệu lực → document_meta_relation; so sánh nội dung chi tiết → so_sanh_van_ban.
3. huong_dan_thu_tuc: mọi thủ tục, hồ sơ, mẫu đơn, thời hạn, UBND/một cửa, kể cả lễ hội–tôn giáo–di tích cấp xã.
4. giai_thich_quy_dinh: giải thích chính sách, mục tiêu chương trình, quy định bảo vệ trẻ em/người yếu thế, bảo tồn di sản (theo văn bản).
5. xu_ly_vi_pham_hanh_chinh vs kiem_tra_thanh_tra: đang xử lý vi phạm → xu_ly; kế hoạch/đoàn thanh tra → kiem_tra_thanh_tra.
6. admin_planning vs to_chuc_su_kien_cong: kế hoạch quản lý chung → admin; tổ chức một sự kiện cụ thể → to_chuc_su_kien_cong.
7. Chọn đúng một intent trong danh sách; ưu tiên ý chính của câu hỏi.

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
    "xu_ly_vi_pham_hanh_chinh": ["xử phạt", "vi phạm hành chính", "mức phạt", "biên bản", "hình thức xử phạt"],
    "to_chuc_su_kien_cong": ["lễ hội", "thể dục thể thao", "biểu diễn nghệ thuật", "sự kiện văn hóa"],
    "giai_thich_quy_dinh": [
        "chính sách",
        "quyền lợi",
        "chế độ",
        "bảo trợ xã hội",
        "an sinh",
        "bạo lực gia đình",
        "bảo vệ trẻ em",
        "di tích",
        "di sản",
        "bảo tồn",
        "mục tiêu chương trình",
    ],
    "huong_dan_thu_tuc": ["thủ tục", "đăng ký", "cấp phép", "hồ sơ", "một cửa", "mẫu đơn", "sinh hoạt tôn giáo"],
    "kiem_tra_thanh_tra": ["kiểm tra", "thanh tra", "rà soát", "giám sát"],
    "hoa_giai_van_dong": ["hòa giải", "tranh chấp", "vận động", "nếp sống văn minh"],
    "trich_xuat_van_ban": ["cấm", "nghiêm cấm", "danh mục", "hành vi bị cấm"],
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
    """Classifier → semantic → structural (YAML); không gọi LLM.

    Dùng bởi `query_intent.compute_intent_bundle` / cờ RAG.
    """
    if not query or not query.strip():
        return "hoi_dap_chung", 0.10

    q = query.strip()

    clf = _detect_intent_classifier(q)
    if clf:
        return _finalize_detector_tuple(*clf)

    semantic = _detect_semantic(q)
    if semantic:
        return semantic

    structural = _detect_structural(q)
    if structural:
        return structural

    return "hoi_dap_chung", 0.30


# ══════════════════════════════════════════════════════════════
# RAG FLAGS (v3) — ánh xạ intent → luồng rag_chain_v2
# ══════════════════════════════════════════════════════════════

# Cờ RAG — is_legal_lookup / needs_expansion / is_scenario theo nhóm intent;
# use_multi_article: True cho mọi intent hợp lệ NGOẠI TRỪ nhóm tra cứu điều khoản/metadata (dưới).
_RAG_LEGAL_LOOKUP_INTENTS: frozenset = frozenset({
    "article_query",
    "document_meta_relation",
    "can_cu_phap_ly",
    "trich_xuat_van_ban",
})

_RAG_NEEDS_EXPANSION_INTENTS: frozenset = frozenset({
    "giai_thich_quy_dinh",
    "hoi_dap_chung",
    "so_sanh_van_ban",
    "xu_ly_vi_pham_hanh_chinh",
})

_RAG_SCENARIO_INTENTS: frozenset = frozenset({
    "huong_dan_thu_tuc",
    "kiem_tra_ho_so",
    "xu_ly_vi_pham_hanh_chinh",
    "kiem_tra_thanh_tra",
    "admin_planning",
    "to_chuc_su_kien_cong",
    "hoa_giai_van_dong",
    "soan_thao_van_ban",
    "tao_bao_cao",
    "giai_thich_quy_dinh",
})


def map_intent_to_rag_flags(intent: str) -> Dict[str, bool]:
    """Ánh xạ intent → 4 cờ RAG. Intent không trong VALID_INTENTS → cả bốn False.

    Quy tắc ``use_multi_article``: bật cho mọi intent hợp lệ **trừ** ``_RAG_LEGAL_LOOKUP_INTENTS``
    (tra Điều/metadata/căn cứ/trích xuất — ưu tiên ngữ cảnh thu hẹp).
    """
    if not intent or intent not in VALID_INTENTS:
        return {
            "is_scenario": False,
            "is_legal_lookup": False,
            "use_multi_article": False,
            "needs_expansion": False,
        }
    legal_lookup = intent in _RAG_LEGAL_LOOKUP_INTENTS
    return {
        "is_legal_lookup": legal_lookup,
        "needs_expansion": intent in _RAG_NEEDS_EXPANSION_INTENTS,
        "use_multi_article": not legal_lookup,
        "is_scenario": intent in _RAG_SCENARIO_INTENTS,
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
            "is_scenario": False,
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
            "is_scenario": False,
            "is_legal_lookup": False,
            "use_multi_article": False,
            "needs_expansion": False,
        }
