"""
Legal Domain Classifier — classify queries into legal domains BEFORE retrieval.

Maps user queries to one or more legal domain tags so the retriever can
pre-filter vectors in Qdrant by `legal_domain` / `document_type` metadata.

Two classification methods:
  1. Semantic: cosine similarity against domain prototypes (~2 ms)
  2. Keyword: deterministic regex fallback (~0 ms)

Domain tags are injected into the Qdrant payload at ingestion time
(via `classify_document_domain`) and matched at query time
(via `classify_query_domain`).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# LEGAL DOMAINS
# ══════════════════════════════════════════════════════════════

LEGAL_DOMAINS: List[str] = [
    "van_hoa",          # Văn hóa, di sản, lễ hội, biểu diễn nghệ thuật
    "the_thao",         # Thể dục thể thao
    "quang_cao",        # Quảng cáo
    "du_lich",          # Du lịch
    "ton_giao",         # Tôn giáo, tín ngưỡng
    "xu_phat",          # Xử phạt vi phạm hành chính
    "dau_tu",           # Đầu tư, kinh doanh
    "dat_dai",          # Đất đai, xây dựng
    "dan_su",           # Dân sự, hôn nhân gia đình
    "lao_dong",         # Lao động, việc làm
    "giao_duc",         # Giáo dục
    "y_te",             # Y tế, bảo hiểm
    "an_sinh",          # An sinh xã hội, bảo trợ, người cao tuổi, trẻ em
    "moi_truong",       # Môi trường, tài nguyên
    "hanh_chinh",       # Hành chính công, thủ tục HC
    "chung",            # Chung / không xác định
]

# ══════════════════════════════════════════════════════════════
# DOMAIN PROTOTYPES — for semantic classification
# ══════════════════════════════════════════════════════════════

DOMAIN_PROTOTYPES: Dict[str, List[str]] = {
    "van_hoa": [
        "Quản lý hoạt động văn hóa trên địa bàn",
        "Bảo tồn di sản văn hóa phi vật thể",
        "Tổ chức lễ hội truyền thống dân gian",
        "Hoạt động biểu diễn nghệ thuật",
        "Di tích lịch sử văn hóa cấp quốc gia",
        "Nhà văn hóa thôn xóm",
        "Gia đình văn hóa, thôn văn hóa",
    ],
    "the_thao": [
        "Đại hội thể dục thể thao cấp xã",
        "Giải bóng chuyền cấp thôn",
        "Hoạt động thể dục thể thao quần chúng",
        "Câu lạc bộ thể thao cộng đồng",
        "Thi đấu thể thao phong trào",
        "Thẩm quyền quyết định tổ chức giải thể thao quần chúng",
    ],
    "quang_cao": [
        "Biển quảng cáo ngoài trời",
        "Vi phạm quảng cáo kích thước",
        "Quảng cáo trái phép trên địa bàn",
        "Cấp phép quảng cáo",
        "Luật Quảng cáo",
    ],
    "du_lich": [
        "Phát triển du lịch cộng đồng",
        "Điểm du lịch sinh thái",
        "Hướng dẫn viên du lịch",
        "Dịch vụ du lịch lữ hành",
        "Khu du lịch quốc gia",
    ],
    "ton_giao": [
        "Đăng ký sinh hoạt tôn giáo tập trung",
        "Hoạt động tín ngưỡng tại cơ sở thờ tự",
        "Tổ chức tôn giáo trái phép",
        "Quản lý nhà nước về tôn giáo",
        "Luật Tín ngưỡng tôn giáo",
    ],
    "xu_phat": [
        "Xử phạt vi phạm hành chính",
        "Mức phạt tiền cho hành vi vi phạm",
        "Lập biên bản xử phạt",
        "Thẩm quyền xử phạt của chủ tịch UBND xã",
        "Nghị định xử phạt trong lĩnh vực văn hóa",
    ],
    "dau_tu": [
        "Luật Đầu tư kinh doanh",
        "Ngành nghề cấm đầu tư",
        "Đăng ký doanh nghiệp",
        "Giấy phép kinh doanh",
        "Nhà đầu tư nước ngoài",
    ],
    "dat_dai": [
        "Quyền sử dụng đất",
        "Cấp giấy chứng nhận quyền sử dụng đất",
        "Tranh chấp đất đai",
        "Xây dựng nhà ở trên đất nông nghiệp",
        "Luật Đất đai",
    ],
    "dan_su": [
        "Hôn nhân gia đình",
        "Đăng ký kết hôn",
        "Ly hôn thuận tình",
        "Quyền nuôi con sau ly hôn",
        "Thừa kế tài sản",
    ],
    "lao_dong": [
        "Hợp đồng lao động",
        "Bảo hiểm xã hội người lao động",
        "Chế độ nghỉ phép thai sản",
        "Tiền lương tối thiểu vùng",
        "An toàn lao động",
    ],
    "giao_duc": [
        "Giáo dục phổ thông",
        "Tuyển sinh đầu cấp",
        "Chế độ miễn giảm học phí",
        "Giáo dục nghề nghiệp",
        "Xã hội hóa giáo dục",
    ],
    "y_te": [
        "Bảo hiểm y tế",
        "Khám chữa bệnh tuyến xã",
        "An toàn vệ sinh thực phẩm",
        "Phòng chống dịch bệnh",
        "Y tế cơ sở",
    ],
    "an_sinh": [
        "Bảo trợ xã hội cho người cao tuổi",
        "Chính sách hỗ trợ người khuyết tật",
        "Trợ cấp xã hội hàng tháng",
        "Bảo vệ quyền trẻ em",
        "Phòng chống bạo lực gia đình",
        "Chế độ trợ giúp xã hội",
    ],
    "moi_truong": [
        "Bảo vệ môi trường",
        "Xử lý rác thải sinh hoạt",
        "Đánh giá tác động môi trường",
        "Tài nguyên nước",
        "Khai thác khoáng sản",
    ],
    "hanh_chinh": [
        "Thủ tục hành chính một cửa",
        "Cải cách hành chính",
        "Công chức viên chức cấp xã",
        "Quản lý hộ tịch hộ khẩu",
        "Văn bản quy phạm pháp luật",
    ],
    "chung": [
        "Xin chào bạn",
        "Hệ thống này làm được gì?",
        "Câu hỏi tổng quát",
        "Không liên quan đến pháp luật",
    ],
}

# ══════════════════════════════════════════════════════════════
# KEYWORD-BASED DOMAIN DETECTION (fast fallback)
# ══════════════════════════════════════════════════════════════

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "van_hoa": [
        "di sản", "di tích", "văn hóa", "lễ hội", "biểu diễn", "nghệ thuật",
        "nhà văn hóa", "làng nghề", "phi vật thể", "trùng tu", "bảo tồn",
    ],
    "the_thao": [
        "thể dục", "thể thao", "tdtt", "bóng chuyền", "bóng đá",
        "đại hội thể thao", "giải đấu", "giải thể thao", "thể thao quần chúng",
        "tổ chức giải thể thao", "luật thể dục thể thao", "thẩm quyền thể thao",
    ],
    "quang_cao": ["quảng cáo", "biển quảng cáo", "bảng hiệu"],
    "du_lich": ["du lịch", "lữ hành", "hướng dẫn viên", "khu du lịch"],
    "ton_giao": [
        "tôn giáo", "tín ngưỡng", "thờ tự", "chùa", "nhà thờ",
        "sinh hoạt tôn giáo",
    ],
    "xu_phat": [
        "xử phạt", "mức phạt", "biên bản", "phạt tiền", "vi phạm hành chính",
        "thẩm quyền xử phạt", "phạt bao nhiêu",
    ],
    "dau_tu": [
        "đầu tư", "kinh doanh", "doanh nghiệp", "nhà đầu tư",
        "giấy phép kinh doanh", "luật đầu tư",
    ],
    "dat_dai": [
        "đất đai", "quyền sử dụng đất", "sổ đỏ", "giấy chứng nhận đất",
        "xây dựng", "nhà ở",
    ],
    "dan_su": [
        "hôn nhân", "gia đình", "kết hôn", "ly hôn", "thừa kế",
        "nuôi con", "hộ tịch",
    ],
    "lao_dong": [
        "lao động", "tiền lương", "hợp đồng lao động", "bảo hiểm xã hội",
        "thai sản", "an toàn lao động",
    ],
    "giao_duc": ["giáo dục", "học phí", "tuyển sinh", "trường học", "đào tạo"],
    "y_te": [
        "y tế", "bảo hiểm y tế", "khám chữa bệnh", "vệ sinh thực phẩm",
        "dịch bệnh", "phòng chống dịch",
    ],
    "an_sinh": [
        "bảo trợ xã hội", "an sinh", "người cao tuổi", "người khuyết tật",
        "trẻ em", "bạo lực gia đình", "trợ cấp", "người yếu thế",
        "bảo vệ trẻ em", "người già",
    ],
    "moi_truong": [
        "môi trường", "rác thải", "ô nhiễm", "tài nguyên",
        "khoáng sản", "nước thải",
    ],
    "hanh_chinh": [
        "hành chính", "thủ tục", "một cửa", "công chức", "viên chức",
        "cải cách hành chính",
    ],
}

# ── doc_number / document_type → domain mapping ─────────────

_DOCTYPE_TO_DOMAIN: Dict[str, str] = {
    "Luật": None,  # needs content-based classification
    "Nghị định": None,
    "Thông tư": None,
    "Quyết định": None,
    "Chỉ thị": None,
    "Nghị quyết": None,
}

# ══════════════════════════════════════════════════════════════
# SEMANTIC INDEX (built at startup)
# ══════════════════════════════════════════════════════════════

_domain_labels: List[str] = []
_domain_matrix: Optional[np.ndarray] = None


def _build_domain_index() -> None:
    global _domain_labels, _domain_matrix

    from app.pipeline.embedding import embed_texts

    sentences: List[str] = []
    labels: List[str] = []
    for domain, protos in DOMAIN_PROTOTYPES.items():
        for s in protos:
            sentences.append(s)
            labels.append(domain)

    if not sentences:
        return

    t0 = time.monotonic()
    _domain_matrix = embed_texts(sentences)
    _domain_labels = labels
    elapsed = (time.monotonic() - t0) * 1000
    log.info(
        "Domain index built: %d sentences, %d domains (%.0f ms)",
        len(sentences), len(DOMAIN_PROTOTYPES), elapsed,
    )


def warmup_domain_index() -> None:
    """Call at app startup after embedding model is loaded."""
    try:
        _build_domain_index()
    except Exception as exc:
        log.error("Failed to build domain index: %s", exc)


def _semantic_classify(text: str, top_n: int = 2) -> List[Tuple[str, float]]:
    """Return top-N domains by cosine similarity."""
    if _domain_matrix is None or not _domain_labels:
        return []

    from app.pipeline.embedding import embed_texts

    q_vec = embed_texts([text])
    if q_vec.size == 0:
        return []

    sims = (q_vec @ _domain_matrix.T).flatten()

    domain_scores: Dict[str, float] = {}
    for i, sim in enumerate(sims):
        label = _domain_labels[i]
        if label not in domain_scores or sim > domain_scores[label]:
            domain_scores[label] = float(sim)

    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_domains[:top_n]


def _keyword_classify(text: str) -> List[Tuple[str, float]]:
    """Keyword-based domain detection (deterministic fallback)."""
    q = text.lower()
    results: List[Tuple[str, float]] = []

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in q)
        if matches > 0:
            conf = min(0.5 + matches * 0.15, 0.90)
            results.append((domain, conf))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

def classify_query_domain(query: str, top_n: int = 2) -> List[Dict]:
    """Classify a user query into legal domains.

    Returns list of {"domain": str, "confidence": float, "method": str}
    ordered by confidence descending.
    """
    if not query or not query.strip():
        return [{"domain": "chung", "confidence": 0.1, "method": "guard"}]

    results: List[Dict] = []

    # 1) Semantic classification
    semantic = _semantic_classify(query, top_n=top_n + 1)
    for domain, sim in semantic:
        if domain == "chung":
            continue
        if sim >= 0.55:
            results.append({
                "domain": domain,
                "confidence": round(sim, 3),
                "method": "semantic",
            })

    # 2) Keyword boost / fallback
    keyword_hits = _keyword_classify(query)
    existing_domains = {r["domain"] for r in results}

    for domain, conf in keyword_hits:
        if domain in existing_domains:
            for r in results:
                if r["domain"] == domain:
                    r["confidence"] = round(min(r["confidence"] + 0.1, 0.95), 3)
                    break
        else:
            results.append({
                "domain": domain,
                "confidence": round(conf, 3),
                "method": "keyword",
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)

    if not results:
        return [{"domain": "chung", "confidence": 0.3, "method": "fallback"}]

    return results[:top_n]


def classify_document_domain(title: str, content_snippet: str = "") -> str:
    """Classify a legal document into a single primary domain.

    Used during ingestion to tag the Qdrant payload with `legal_domain`.
    """
    text = f"{title} {content_snippet[:500]}".strip()
    if not text:
        return "chung"

    keyword_hits = _keyword_classify(text)
    if keyword_hits and keyword_hits[0][1] >= 0.60:
        return keyword_hits[0][0]

    semantic = _semantic_classify(text, top_n=1)
    if semantic and semantic[0][1] >= 0.50:
        return semantic[0][0]

    return "chung"


def get_domain_filter_values(query: str) -> Optional[List[str]]:
    """Return domain values to filter in Qdrant, or None for no filtering.

    Only filters when classification is confident enough (>= 0.60).
    Returns None (no filter) for ambiguous or general queries.
    """
    domains = classify_query_domain(query, top_n=3)

    confident = [d for d in domains if d["confidence"] >= 0.60 and d["domain"] != "chung"]
    if not confident:
        return None

    return [d["domain"] for d in confident[:2]]
