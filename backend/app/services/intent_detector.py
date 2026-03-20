"""
Intent Detection Module v5 — 3-Tier Semantic + LLM + Auto-index

Pipeline (executed in order, stops at first high-confidence match):

  Layer 0: Guard       → empty / too-short / noise queries              (~0 ms)
  Layer 1: Structural  → "Điều X", số hiệu VB, "Tóm tắt NĐ"          (~0 ms)
  Layer 2: Semantic    → cosine similarity vs prototype embeddings      (~2 ms)
  Layer 3: LLM         → zero-shot GPT classification (complex queries) (~1-3 s)

Tier 1 – Semantic Similarity:  ~5 prototype sentences per intent,
         pre-embedded at startup, cosine-matched against query embedding.
Tier 2 – Zero-shot LLM:       Full catalog + disambiguation rules + CoT.
Tier 3 – Auto-index from DB:  On document upload, extract representative
         sentences and merge into the prototype index automatically.

All public interfaces are backward-compatible.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# CONSTANTS — KHÔNG THAY ĐỔI (các module khác import trực tiếp)
# ══════════════════════════════════════════════════════════════

VALID_INTENTS: List[str] = [
    "tra_cuu_van_ban",
    "article_query",
    "document_metadata",
    "program_goal",
    "document_relation",
    "can_cu_phap_ly",
    "soan_thao_van_ban",
    "giai_thich_quy_dinh",
    "huong_dan_thu_tuc",
    "kiem_tra_ho_so",
    "tom_tat_van_ban",
    "so_sanh_van_ban",
    "tao_bao_cao",
    "trich_xuat_van_ban",
    "admin_planning",
    "xu_ly_vi_pham_hanh_chinh",
    "kiem_tra_thanh_tra",
    "thu_tuc_hanh_chinh",
    "hoa_giai_van_dong",
    "bao_ve_xa_hoi",
    "to_chuc_su_kien_cong",
    "bao_ton_phat_trien",
    "hoi_dap_chung",
]

COMMUNE_LEVEL_INTENTS: set = {
    "xu_ly_vi_pham_hanh_chinh",
    "kiem_tra_thanh_tra",
    "thu_tuc_hanh_chinh",
    "hoa_giai_van_dong",
    "bao_ve_xa_hoi",
    "to_chuc_su_kien_cong",
    "bao_ton_phat_trien",
}

# ══════════════════════════════════════════════════════════════
# LAYER 1: STRUCTURAL DETECTION — deterministic, ~0 ms
# ══════════════════════════════════════════════════════════════

_STRUCTURAL_RULES: List[Tuple[str, str, float]] = [
    (r"điều\s+\d+\s+(luật|nghị\s*định|thông\s*tư|quyết\s*định|pháp\s*lệnh|chỉ\s*thị)", "article_query", 0.97),
    (r"khoản\s+\d+\s+điều\s+\d+", "article_query", 0.97),
    (r"^\s*điều\s+\d+[\s,\.]*$", "article_query", 0.95),
    (r"\b\d{1,3}/\d{4}/[A-ZĐa-zđ\-]+\b", "article_query", 0.95),
    (r"\b\d{1,3}/[A-ZĐa-zđ]{2,}-[A-ZĐa-zđ]{2,}\b", "article_query", 0.93),
    (r"(tóm\s*tắt|tổng\s*hợp|khái\s*quát)\s+(nghị\s*định|luật|thông\s*tư|quyết\s*định|chỉ\s*thị)", "tom_tat_van_ban", 0.95),
    (r"so\s*sánh\s+.{0,30}(luật|nghị\s*định|thông\s*tư).{0,30}(và|với|so\s*với)", "so_sanh_van_ban", 0.95),
    (r"(soạn|viết)\s+(công\s*văn|tờ\s*trình|biên\s*bản|quyết\s*định|thông\s*báo|đơn\s+xin)", "soan_thao_van_ban", 0.95),
    (r"(lập|tạo)\s+(công\s*văn|tờ\s*trình|quyết\s*định|thông\s*báo|đơn\s+xin)", "soan_thao_van_ban", 0.95),
    (r"(tạo|viết|lập|soạn)\s+(báo\s*cáo)", "tao_bao_cao", 0.95),
    (r"(đã\s+nộp|đã\s+có|đã\s+gửi).{0,50}(còn\s+thiếu|thiếu\s+gì|đủ\s+chưa)", "kiem_tra_ho_so", 0.95),
    (r"(ban\s*hành|có\s*hiệu\s*lực|hết\s*hiệu\s*lực)\s+(ngày|năm|khi|từ)\s+nào", "document_metadata", 0.95),
    (r"(ai|cơ\s*quan\s*nào)\s+(ban\s*hành|ký\s*ban\s*hành|phê\s*duyệt)", "document_metadata", 0.93),
]


def _detect_structural(query: str) -> Optional[Tuple[str, float]]:
    """Layer 1: Structural pattern detection (~0 ms)."""
    q = query.lower().strip()
    for pattern, intent, confidence in _STRUCTURAL_RULES:
        if re.search(pattern, q):
            log.debug("Structural match [%s]: pattern='%s'", intent, pattern[:40])
            return intent, confidence
    return None


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
        "Chính sách bảo trợ xã hội nằm trong điều luật nào?",
    ],
    "document_metadata": [
        "Nghị định 36/2019 do cơ quan nào ban hành?",
        "Luật Di sản văn hóa có hiệu lực từ ngày nào?",
        "Ai ký ban hành Thông tư 13/2024?",
        "Nghị định 144 ban hành năm nào?",
        "Quyết định 706 có còn hiệu lực không?",
    ],
    "program_goal": [
        "Mục tiêu chương trình mục tiêu quốc gia nông thôn mới",
        "Đề án phát triển văn hóa 2030 nhằm mục đích gì?",
        "Chỉ tiêu chính của kế hoạch phát triển kinh tế xã hội",
        "Nhiệm vụ chủ yếu của chương trình giảm nghèo bền vững",
        "Mục đích của đề án bảo tồn di sản văn hóa phi vật thể",
    ],
    "document_relation": [
        "Luật nào sửa đổi Luật Di sản văn hóa 2001?",
        "Nghị định 36/2019 thay thế nghị định nào?",
        "Văn bản nào bãi bỏ Thông tư 04/2010?",
        "Luật Đầu tư 2025 sửa đổi bổ sung những luật nào?",
        "Nghị định mới thay thế Nghị định 110/2018?",
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
    ],
    "huong_dan_thu_tuc": [
        "Thủ tục đăng ký kinh doanh gồm mấy bước?",
        "Hồ sơ xin cấp phép xây dựng cần giấy tờ gì?",
        "Quy trình đăng ký hộ khẩu thường trú",
        "Các bước xin cấp giấy chứng nhận quyền sử dụng đất",
        "Làm sao để đăng ký kết hôn?",
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
    "thu_tuc_hanh_chinh": [
        "Thủ tục xin phép tổ chức lễ hội dân gian cấp xã",
        "Hồ sơ đăng ký sinh hoạt tôn giáo tập trung",
        "Thủ tục xin cấp phép tu bổ ngôi đình cổ",
        "Đăng ký hoạt động biểu diễn nghệ thuật tại xã",
        "Hồ sơ công nhận gia đình văn hóa, thôn văn hóa",
    ],
    "hoa_giai_van_dong": [
        "Hai hàng xóm tranh chấp ranh giới đất, cần hòa giải",
        "Vận động người dân thực hiện nếp sống văn minh trong lễ hội",
        "Hòa giải mâu thuẫn tiếng ồn giữa hai gia đình",
        "Tuyên truyền phổ biến pháp luật cho nhân dân",
        "Vận động cộng đồng chấp hành hương ước thôn bản",
    ],
    "bao_ve_xa_hoi": [
        "Trẻ em bị bố đánh đập hàng ngày, cần can thiệp ngay",
        "Người già bị con cái bỏ rơi không chăm sóc",
        "Phụ nữ bị chồng bạo lực gia đình, cần hỗ trợ khẩn cấp",
        "Phát hiện trẻ em lang thang không nơi nương tựa",
        "Người khuyết tật bị ngược đãi, cần bảo vệ",
    ],
    "to_chuc_su_kien_cong": [
        "Tổ chức Đại hội Thể dục thể thao cấp xã",
        "Lên kế hoạch biểu diễn văn nghệ chào mừng ngày lễ",
        "Tổ chức hội thi tìm hiểu pháp luật cho thanh niên",
        "Chuẩn bị lễ hội truyền thống hàng năm của xã",
        "Kế hoạch tổ chức giải bóng chuyền cấp thôn",
    ],
    "bao_ton_phat_trien": [
        "Ngôi chùa cổ bị hư hỏng sau bão, cần trùng tu",
        "Bảo tồn làng nghề truyền thống đan lát",
        "Di tích lịch sử cấp quốc gia xuống cấp nghiêm trọng",
        "Phát huy giá trị di sản văn hóa phi vật thể địa phương",
        "Khôi phục nghề thủ công truyền thống đang mai một",
    ],
    "hoi_dap_chung": [
        "Xin chào",
        "Bạn là ai?",
        "Hệ thống này làm được gì?",
        "Cảm ơn bạn",
        "Tạm biệt",
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
    for intent, protos in INTENT_PROTOTYPES.items():
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
        "Intent prototype index built: %d sentences, %d intents (%.0f ms)",
        len(sentences), len(INTENT_PROTOTYPES), elapsed,
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

    log.info(
        "Semantic match: %s (sim=%.3f, gap=%.3f, conf=%.2f)",
        best_intent, best_sim, gap, confidence,
    )
    return best_intent, confidence


# ══════════════════════════════════════════════════════════════
# LAYER 3: ZERO-SHOT LLM CLASSIFICATION — Tier 2
# ══════════════════════════════════════════════════════════════

INTENT_SEMANTIC_DESCRIPTIONS: Dict[str, str] = {
    "tra_cuu_van_ban": "Tìm kiếm, tra cứu văn bản pháp luật theo chủ đề hoặc lĩnh vực (chưa biết tên/số hiệu cụ thể)",
    "article_query": "Hỏi về nội dung Điều/Khoản cụ thể, hoặc tra cứu theo số hiệu văn bản đã biết",
    "document_metadata": "Hỏi thông tin hành chính của văn bản: ai ban hành, ngày ban hành, còn hiệu lực không",
    "program_goal": "Hỏi về mục tiêu, mục đích, chỉ tiêu của chương trình/đề án/kế hoạch quốc gia",
    "document_relation": "Hỏi quan hệ giữa các văn bản: văn bản nào sửa đổi/thay thế/bãi bỏ văn bản nào",
    "can_cu_phap_ly": "Hỏi căn cứ pháp lý: văn bản nào làm cơ sở pháp lý, dựa vào luật nào",
    "soan_thao_van_ban": "Yêu cầu soạn/viết văn bản hành chính: công văn, tờ trình, biên bản, quyết định",
    "giai_thich_quy_dinh": "Giải thích quy định pháp luật, chính sách, chế độ; hỏi 'là gì', 'hiểu thế nào'",
    "huong_dan_thu_tuc": "Hỏi quy trình/thủ tục/các bước thực hiện; hỏi hồ sơ cần gì, nộp ở đâu",
    "kiem_tra_ho_so": "Kiểm tra hồ sơ đã nộp còn thiếu gì so với yêu cầu",
    "tom_tat_van_ban": "Tóm tắt, khái quát nội dung văn bản pháp luật cụ thể đã biết tên",
    "so_sanh_van_ban": "So sánh nội dung hai văn bản luật, tìm điểm giống/khác",
    "tao_bao_cao": "Tạo/lập báo cáo hành chính, tổng kết, đánh giá",
    "trich_xuat_van_ban": "Liệt kê/trích xuất toàn bộ danh mục từ văn bản: điều cấm, quyền, nghĩa vụ",
    "admin_planning": "Lập kế hoạch quản lý, phân bổ nguồn lực, tổ chức thực hiện cấp địa phương",
    "xu_ly_vi_pham_hanh_chinh": "Xử lý vi phạm hành chính: karaoke ồn, quảng cáo trái phép, kinh doanh không phép",
    "kiem_tra_thanh_tra": "Tổ chức đoàn kiểm tra, thanh tra cơ sở kinh doanh/dịch vụ văn hóa",
    "thu_tuc_hanh_chinh": "Thủ tục hành chính cấp xã: đăng ký lễ hội, cấp phép biểu diễn, tu bổ di tích",
    "hoa_giai_van_dong": "Hòa giải tranh chấp dân sự, vận động nhân dân nếp sống văn minh",
    "bao_ve_xa_hoi": "Bảo vệ nạn nhân bạo lực gia đình, trẻ em bị xâm hại, người yếu thế — khẩn cấp",
    "to_chuc_su_kien_cong": "Tổ chức sự kiện văn hóa thể thao: lễ hội, đại hội TDTT, biểu diễn văn nghệ",
    "bao_ton_phat_trien": "Bảo tồn di sản văn hóa, trùng tu di tích, phát huy giá trị truyền thống",
    "hoi_dap_chung": "Câu hỏi chung, chào hỏi, không thuộc các loại trên",
}

_INTENT_CATALOG: Dict[str, Dict] = {
    "tra_cuu_van_ban": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["tra_cuu_van_ban"],
        "vi_du": ["Văn bản nào quy định về lễ hội dân gian?", "Tìm các nghị định về quảng cáo"],
        "khong_phai": "Nếu đã biết tên luật cụ thể/số điều → article_query hoặc tom_tat_van_ban",
    },
    "article_query": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["article_query"],
        "vi_du": ["Điều 47 Luật Di sản văn hóa quy định gì?", "Khoản 2 Điều 6 Luật Đầu tư 2025"],
        "khong_phai": "Nếu không có số Điều/Khoản cụ thể → tra_cuu_van_ban hoặc trich_xuat_van_ban",
    },
    "document_metadata": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["document_metadata"],
        "vi_du": ["Thông tư 13/2024 do cơ quan nào ban hành?", "Nghị định 36/2019 có còn hiệu lực không?"],
        "khong_phai": "Nếu hỏi nội dung trong văn bản → article_query",
    },
    "program_goal": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["program_goal"],
        "vi_du": ["Mục tiêu chương trình nông thôn mới?", "Đề án phát triển văn hóa 2030 nhằm gì?"],
        "khong_phai": "",
    },
    "document_relation": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["document_relation"],
        "vi_du": ["Luật nào sửa đổi Luật Di sản văn hóa 2001?", "Nghị định 36/2019 thay thế NĐ nào?"],
        "khong_phai": "So sánh nội dung → so_sanh_van_ban",
    },
    "can_cu_phap_ly": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["can_cu_phap_ly"],
        "vi_du": ["Căn cứ pháp lý của kế hoạch này?", "Mâu thuẫn luật thì áp dụng luật nào?"],
        "khong_phai": "",
    },
    "soan_thao_van_ban": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["soan_thao_van_ban"],
        "vi_du": ["Soạn công văn xin gia hạn giấy phép", "Viết tờ trình đề nghị cấp kinh phí"],
        "khong_phai": "Thủ tục → huong_dan_thu_tuc; báo cáo → tao_bao_cao",
    },
    "giai_thich_quy_dinh": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["giai_thich_quy_dinh"],
        "vi_du": ["Chính sách bảo trợ xã hội là gì?", "An sinh xã hội bao gồm gì?"],
        "khong_phai": "Bạo hành khẩn cấp → bao_ve_xa_hoi",
    },
    "huong_dan_thu_tuc": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["huong_dan_thu_tuc"],
        "vi_du": ["Thủ tục đăng ký kinh doanh?", "Hồ sơ xin cấp phép xây dựng?"],
        "khong_phai": "Thủ tục cấp xã → thu_tuc_hanh_chinh",
    },
    "kiem_tra_ho_so": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["kiem_tra_ho_so"],
        "vi_du": ["Đã nộp giấy đề nghị, còn thiếu gì?", "Hồ sơ đã đủ chưa?"],
        "khong_phai": "Chưa có gì, hỏi cần gì → huong_dan_thu_tuc",
    },
    "tom_tat_van_ban": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["tom_tat_van_ban"],
        "vi_du": ["Tóm tắt Luật Đầu tư 2025", "Nội dung chính NĐ 144"],
        "khong_phai": "Mục tiêu chương trình → program_goal",
    },
    "so_sanh_van_ban": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["so_sanh_van_ban"],
        "vi_du": ["Luật ĐT 2020 và 2025 khác gì?", "Điểm mới Luật DSVH sửa đổi?"],
        "khong_phai": "VB nào thay thế VB nào → document_relation",
    },
    "tao_bao_cao": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["tao_bao_cao"],
        "vi_du": ["Lập báo cáo tổng kết VH 2025", "Viết báo cáo phong trào toàn dân"],
        "khong_phai": "Công văn/tờ trình → soan_thao_van_ban",
    },
    "trich_xuat_van_ban": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["trich_xuat_van_ban"],
        "vi_du": ["Các ngành nghề cấm đầu tư", "Liệt kê quyền nhà đầu tư"],
        "khong_phai": "Một điều cụ thể → article_query",
    },
    "admin_planning": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["admin_planning"],
        "vi_du": ["Kế hoạch quản lý di tích 2025", "Phân bổ nhân sự bộ phận VH"],
        "khong_phai": "Sự kiện → to_chuc_su_kien_cong; vi phạm → xu_ly_vi_pham",
    },
    "xu_ly_vi_pham_hanh_chinh": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["xu_ly_vi_pham_hanh_chinh"],
        "vi_du": ["Karaoke ồn quá giờ, xử phạt?", "Biển QC vi phạm, xử lý?"],
        "khong_phai": "Thủ tục cấp phép → thu_tuc_hanh_chinh",
    },
    "kiem_tra_thanh_tra": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["kiem_tra_thanh_tra"],
        "vi_du": ["Kiểm tra đột xuất quán karaoke", "KH thanh tra cơ sở internet"],
        "khong_phai": "Phát hiện vi phạm → xu_ly_vi_pham_hanh_chinh",
    },
    "thu_tuc_hanh_chinh": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["thu_tuc_hanh_chinh"],
        "vi_du": ["Xin phép tổ chức lễ hội xã", "Hồ sơ tu bổ ngôi đình"],
        "khong_phai": "Thủ tục chung → huong_dan_thu_tuc",
    },
    "hoa_giai_van_dong": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["hoa_giai_van_dong"],
        "vi_du": ["Hàng xóm tranh chấp đất, hòa giải?", "Vận động dân chấp hành hương ước"],
        "khong_phai": "",
    },
    "bao_ve_xa_hoi": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["bao_ve_xa_hoi"],
        "vi_du": ["Trẻ em bị bố đánh, can thiệp ngay", "Người già bị bỏ rơi"],
        "khong_phai": "Hỏi chính sách/quyền lợi → giai_thich_quy_dinh",
    },
    "to_chuc_su_kien_cong": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["to_chuc_su_kien_cong"],
        "vi_du": ["Tổ chức Đại hội TDTT xã", "KH biểu diễn văn nghệ ngày lễ"],
        "khong_phai": "Xin phép sự kiện → thu_tuc_hanh_chinh",
    },
    "bao_ton_phat_trien": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["bao_ton_phat_trien"],
        "vi_du": ["Chùa hư hỏng sau bão, trùng tu", "Bảo tồn làng nghề truyền thống"],
        "khong_phai": "Xin phép tu bổ → thu_tuc_hanh_chinh",
    },
    "hoi_dap_chung": {
        "mo_ta": INTENT_SEMANTIC_DESCRIPTIONS["hoi_dap_chung"],
        "vi_du": ["Xin chào", "Bạn là ai?"],
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

## DANH SÁCH INTENT

{catalog_text}

## QUY TẮC PHÂN LOẠI

1. giai_thich_quy_dinh vs bao_ve_xa_hoi:
   - "Chính sách X là gì?" → giai_thich_quy_dinh
   - "Người X đang bị Y, cần can thiệp" → bao_ve_xa_hoi

2. trich_xuat_van_ban vs article_query:
   - "Các/Tất cả ngành nghề cấm" (liệt kê) → trich_xuat_van_ban
   - "Điều 6 quy định gì?" (cụ thể) → article_query

3. thu_tuc_hanh_chinh vs huong_dan_thu_tuc:
   - Đặc thù cấp xã (lễ hội, tôn giáo, di tích) → thu_tuc_hanh_chinh
   - Chung (đăng ký KD, xây dựng) → huong_dan_thu_tuc

4. xu_ly_vi_pham vs kiem_tra_thanh_tra:
   - Đã vi phạm, cần xử lý → xu_ly_vi_pham_hanh_chinh
   - Kế hoạch kiểm tra → kiem_tra_thanh_tra

5. so_sanh_van_ban vs document_relation:
   - So sánh nội dung → so_sanh_van_ban
   - Thay thế/sửa đổi → document_relation

6. admin_planning vs to_chuc_su_kien_cong:
   - Kế hoạch quản lý → admin_planning
   - Tổ chức sự kiện cụ thể → to_chuc_su_kien_cong

7. Nếu câu hỏi hỏi "thẩm quyền" + hành động cụ thể → xem xét intent theo hành động đó.
8. Nếu câu hỏi lồng nhiều ý định → chọn intent PHÙ HỢP NHẤT với ý chính.

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

    intent_match = re.search(r"intent:\s*([a-z_]+)", raw_lower)
    if intent_match:
        candidate = intent_match.group(1).strip()
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
    "bao_ve_xa_hoi": ["bạo lực gia đình", "bảo vệ trẻ em", "người cao tuổi", "người khuyết tật", "nạn nhân"],
    "to_chuc_su_kien_cong": ["lễ hội", "thể dục thể thao", "biểu diễn nghệ thuật", "sự kiện văn hóa"],
    "bao_ton_phat_trien": ["di tích", "di sản", "bảo tồn", "trùng tu", "phát huy giá trị"],
    "giai_thich_quy_dinh": ["chính sách", "quyền lợi", "chế độ", "bảo trợ xã hội", "an sinh"],
    "thu_tuc_hanh_chinh": ["thủ tục", "đăng ký", "cấp phép", "hồ sơ", "sinh hoạt tôn giáo"],
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
    conf = raw_confidence

    word_count = len(query.split())
    if word_count <= 2 and method == "llm":
        conf = max(conf - 0.10, 0.40)
    elif word_count <= 4 and method == "llm":
        conf = max(conf - 0.05, 0.50)

    if intent in COMMUNE_LEVEL_INTENTS and method in ("llm", "semantic"):
        _BOOST_SIGNALS = [
            "địa bàn", "xã", "phường", "thôn", "ông/bà", "tham mưu",
            "ra quân", "vi phạm", "xử lý", "tình huống",
        ]
        q_lower = query.lower()
        if any(sig in q_lower for sig in _BOOST_SIGNALS):
            conf = min(conf + 0.05, 0.95)

    return round(conf, 2)


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
            log.info("LLM classified: %s (conf=%.2f)", intent, confidence)
            return intent, confidence

        log.warning("LLM output parse failed. Raw: %s", raw[:200])
        return "hoi_dap_chung", 0.30

    except Exception as exc:
        log.error("LLM intent detection error: %s", exc)
        return "hoi_dap_chung", 0.30


async def detect_intent(query: str) -> Dict[str, object]:
    """Main 3-tier intent detection pipeline.

    Layer 0 → Guard (empty/noise)
    Layer 1 → Structural (deterministic regex, ~0 ms)
    Layer 2 → Semantic similarity (embedding cosine, ~2 ms)
    Layer 3 → Zero-shot LLM (complex queries, ~1-3 s)

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

    # ── Layer 1: Structural ─────────────────────────────────
    structural_result = _detect_structural(query)
    if structural_result:
        intent, confidence = structural_result
        calibrated = _calibrate_confidence(intent, confidence, query, "structural")
        log.info("Structural intent: %s (conf=%.2f)", intent, calibrated)
        return {"intent": intent, "confidence": calibrated, "method": "structural"}

    # ── Layer 2: Semantic similarity ─────────────────────────
    semantic_result = _detect_semantic(query)
    if semantic_result:
        intent, confidence = semantic_result
        calibrated = _calibrate_confidence(intent, confidence, query, "semantic")
        log.info("Semantic intent: %s (conf=%.2f)", intent, calibrated)
        return {"intent": intent, "confidence": calibrated, "method": "semantic"}

    # ── Layer 3: Zero-shot LLM ───────────────────────────────
    log.info("Calling LLM for intent classification: '%s'", query[:60])
    intent, confidence = await detect_intent_llm(query)
    calibrated = _calibrate_confidence(intent, confidence, query, "llm")

    return {"intent": intent, "confidence": calibrated, "method": "llm"}


# ══════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ══════════════════════════════════════════════════════════════

def detect_intent_rule_based(query: str) -> Tuple[str, float]:
    """DEPRECATED: Kept for backward compatibility.

    Returns structural/semantic result or low-confidence fallback.
    """
    if not query or not query.strip():
        return "hoi_dap_chung", 0.10

    q = query.strip()

    structural = _detect_structural(q)
    if structural:
        return structural

    semantic = _detect_semantic(q)
    if semantic:
        return semantic

    return "hoi_dap_chung", 0.30
