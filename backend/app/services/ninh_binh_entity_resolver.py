"""Ninh Bình Entity Resolver – administrative hierarchy + fuzzy entity detection.

Cơ cấu hành chính mới (sau sáp nhập 2025): chỉ 2 cấp:
  Tỉnh Ninh Bình → Xã / Phường (không còn cấp Huyện)

Tên huyện cũ (Gia Viễn, Nho Quan, …) được giữ làm "khu vực" (area)
để hỗ trợ tìm kiếm chính xác hơn.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

PROVINCE = "Ninh Bình"

# ── All communes / wards – trực thuộc tỉnh ────────────────
# area_hint = tên khu vực cũ (huyện cũ) để hỗ trợ tìm kiếm

COMMUNES: List[Dict[str, str]] = [
    # ── Khu vực TP Ninh Bình cũ ────────
    {"name": "Hoa Lư", "type": "phuong", "area": "Ninh Bình"},
    {"name": "Tây Hoa Lư", "type": "phuong", "area": "Ninh Bình"},
    {"name": "Đông Hoa Lư", "type": "phuong", "area": "Ninh Bình"},
    {"name": "Nam Hoa Lư", "type": "phuong", "area": "Ninh Bình"},
    {"name": "Thành Nam", "type": "phuong", "area": "Ninh Bình"},
    {"name": "Yên Sơn", "type": "phuong", "area": "Ninh Bình"},
    {"name": "Trung Sơn", "type": "phuong", "area": "Ninh Bình"},
    {"name": "Hồng Quang", "type": "phuong", "area": "Ninh Bình"},
    {"name": "Trường Thi", "type": "phuong", "area": "Ninh Bình"},
    {"name": "Ninh Giang", "type": "xa", "area": "Ninh Bình"},
    {"name": "Ninh Nhất", "type": "xa", "area": "Ninh Bình"},
    {"name": "Ninh Tiến", "type": "xa", "area": "Ninh Bình"},
    {"name": "Ninh Phúc", "type": "xa", "area": "Ninh Bình"},
    {"name": "Ninh Sơn", "type": "xa", "area": "Ninh Bình"},
    {"name": "Ninh Phong", "type": "xa", "area": "Ninh Bình"},
    # ── Khu vực Tam Điệp cũ ───────────
    {"name": "Tam Điệp", "type": "phuong", "area": "Tam Điệp"},
    {"name": "Quang Sơn", "type": "xa", "area": "Tam Điệp"},
    {"name": "Đông Sơn", "type": "xa", "area": "Tam Điệp"},
    # ── Khu vực Gia Viễn cũ ───────────
    {"name": "Gia Sinh", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Hưng", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Phong", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Tường", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Vân", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Trấn", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Lâm", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Phương", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Tân", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Trung", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Xuân", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Lập", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Minh", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Tiến", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Thắng", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Thanh", "type": "xa", "area": "Gia Viễn"},
    {"name": "Gia Hòa", "type": "xa", "area": "Gia Viễn"},
    {"name": "Liên Sơn", "type": "xa", "area": "Gia Viễn"},
    # ── Khu vực Nho Quan cũ ───────────
    {"name": "Cúc Phương", "type": "xa", "area": "Nho Quan"},
    {"name": "Phú Sơn", "type": "xa", "area": "Nho Quan"},
    {"name": "Thanh Sơn", "type": "xa", "area": "Nho Quan"},
    {"name": "Đồng Thịnh", "type": "xa", "area": "Nho Quan"},
    {"name": "Phú Long", "type": "xa", "area": "Nho Quan"},
    {"name": "Quỳnh Lưu", "type": "xa", "area": "Nho Quan"},
    {"name": "Quảng Lạc", "type": "xa", "area": "Nho Quan"},
    {"name": "Xích Thổ", "type": "xa", "area": "Nho Quan"},
    {"name": "Gia Sơn", "type": "xa", "area": "Nho Quan"},
    {"name": "Thạch Bình", "type": "xa", "area": "Nho Quan"},
    {"name": "Sơn Hà", "type": "xa", "area": "Nho Quan"},
    {"name": "Sơn Thành", "type": "xa", "area": "Nho Quan"},
    {"name": "Văn Phú", "type": "xa", "area": "Nho Quan"},
    {"name": "Văn Phong", "type": "xa", "area": "Nho Quan"},
    {"name": "Kỳ Phú", "type": "xa", "area": "Nho Quan"},
    {"name": "Yên Quang", "type": "xa", "area": "Nho Quan"},
    # ── Khu vực Yên Khánh cũ ──────────
    {"name": "Khánh Nhạc", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Thiện", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Hội", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Trung", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Cường", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh An", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Công", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Hòa", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Lợi", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Mậu", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Phú", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Thành", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Thủy", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Tiên", "type": "xa", "area": "Yên Khánh"},
    {"name": "Khánh Vân", "type": "xa", "area": "Yên Khánh"},
    # ── Khu vực Yên Mô cũ ─────────────
    {"name": "Yên Từ", "type": "xa", "area": "Yên Mô"},
    {"name": "Yên Mạc", "type": "xa", "area": "Yên Mô"},
    {"name": "Yên Đồng", "type": "xa", "area": "Yên Mô"},
    {"name": "Yên Hưng", "type": "xa", "area": "Yên Mô"},
    {"name": "Yên Lâm", "type": "xa", "area": "Yên Mô"},
    {"name": "Yên Nhân", "type": "xa", "area": "Yên Mô"},
    {"name": "Yên Phú", "type": "xa", "area": "Yên Mô"},
    {"name": "Yên Phong", "type": "xa", "area": "Yên Mô"},
    {"name": "Yên Thái", "type": "xa", "area": "Yên Mô"},
    {"name": "Yên Thắng", "type": "xa", "area": "Yên Mô"},
    {"name": "Yên Thành", "type": "xa", "area": "Yên Mô"},
    {"name": "Mai Sơn", "type": "xa", "area": "Yên Mô"},
    # ── Khu vực Kim Sơn cũ ─────────────
    {"name": "Phát Diệm", "type": "xa", "area": "Kim Sơn"},
    {"name": "Lai Thành", "type": "xa", "area": "Kim Sơn"},
    {"name": "Kim Đông", "type": "xa", "area": "Kim Sơn"},
    {"name": "Định Hóa", "type": "xa", "area": "Kim Sơn"},
    {"name": "Bình Minh", "type": "xa", "area": "Kim Sơn"},
    {"name": "Hồng Phong", "type": "xa", "area": "Kim Sơn"},
    {"name": "Chất Bình", "type": "xa", "area": "Kim Sơn"},
    {"name": "Như Hòa", "type": "xa", "area": "Kim Sơn"},
    {"name": "Cồn Thoi", "type": "xa", "area": "Kim Sơn"},
    {"name": "Kim Chính", "type": "xa", "area": "Kim Sơn"},
    {"name": "Kim Hải", "type": "xa", "area": "Kim Sơn"},
    {"name": "Kim Mỹ", "type": "xa", "area": "Kim Sơn"},
    {"name": "Kim Tân", "type": "xa", "area": "Kim Sơn"},
    {"name": "Kim Trung", "type": "xa", "area": "Kim Sơn"},
    {"name": "Lưu Phương", "type": "xa", "area": "Kim Sơn"},
    {"name": "Thượng Kiệm", "type": "xa", "area": "Kim Sơn"},
    {"name": "Xuân Thiện", "type": "xa", "area": "Kim Sơn"},
    {"name": "Yên Lộc", "type": "xa", "area": "Kim Sơn"},
]

# Tên khu vực cũ (huyện/TP cũ) – không còn là đơn vị hành chính
# nhưng vẫn cần nhận diện khi người dùng nhắc tới
AREAS: Dict[str, List[str]] = {
    "Ninh Bình": ["tp ninh bình", "ninh bình city", "thành phố ninh bình"],
    "Tam Điệp": ["tp tam điệp", "tam điệp", "tam diep"],
    "Gia Viễn": ["gia viễn", "gia vien"],
    "Nho Quan": ["nho quan"],
    "Yên Khánh": ["yên khánh", "yen khanh"],
    "Yên Mô": ["yên mô", "yen mo"],
    "Kim Sơn": ["kim sơn", "kim son"],
}

# Danh lam / địa danh du lịch nổi tiếng (không phải đơn vị hành chính)
LANDMARKS: Dict[str, str] = {
    "Tràng An": "Ninh Bình",
    "Tam Cốc": "Ninh Bình",
    "Bái Đính": "Ninh Bình",
    "Bích Động": "Ninh Bình",
    "Vân Long": "Gia Viễn",
    "Cố đô Hoa Lư": "Ninh Bình",
    "Cúc Phương": "Nho Quan",
    "Tam Chúc": "Ninh Bình",
    "Thung Nham": "Ninh Bình",
    "Nhà thờ Phát Diệm": "Kim Sơn",
    "Hang Múa": "Ninh Bình",
}

# ── Lookup indices (built once at import) ──────────────────

_commune_index: Dict[str, Dict[str, str]] = {}   # norm_name → {"name", "type", "area"}
_area_aliases: Dict[str, str] = {}                 # norm_alias → area_name
_all_names: Dict[str, Tuple[str, str, str]] = {}   # norm → (real_name, area_hint, kind)


def _strip_diacritics(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _build_indices() -> None:
    if _commune_index:
        return
    for c in COMMUNES:
        n = _strip_diacritics(c["name"])
        _commune_index[n] = c
        kind = "ward" if c["type"] == "phuong" else "commune"
        _all_names[n] = (c["name"], c["area"], kind)

    for area_name, aliases in AREAS.items():
        a_norm = _strip_diacritics(area_name)
        _area_aliases[a_norm] = area_name
        # Don't add "Ninh Bình" to general index – it conflicts with the province name
        if a_norm != _strip_diacritics(PROVINCE):
            _all_names[a_norm] = (area_name, "", "area")
        for alias in aliases:
            alias_norm = _strip_diacritics(alias)
            _area_aliases[alias_norm] = area_name
            if _strip_diacritics(PROVINCE) not in alias_norm:
                _all_names[alias_norm] = (area_name, "", "area")

    for landmark, near_area in LANDMARKS.items():
        l_norm = _strip_diacritics(landmark)
        _all_names[l_norm] = (landmark, near_area, "landmark")


_build_indices()


# ── Dataclass ──────────────────────────────────────────────

@dataclass
class ResolvedEntity:
    name: str
    level: str          # "province" | "commune" | "ward" | "area" | "landmark" | "unknown"
    area: str = ""      # khu vực cũ (tên huyện cũ) để hỗ trợ tìm kiếm
    province: str = PROVINCE
    confidence: float = 1.0
    search_queries: List[str] = field(default_factory=list)


# ── Regex patterns ─────────────────────────────────────────

_ADMIN_PREFIX_RE = re.compile(
    r"\b(tỉnh|tinh|thành\s*phố|thanh\s*pho|tp"
    r"|huyện|huyen"    # vẫn nhận diện "huyện" dù không còn cấp hành chính
    r"|xã|xa|phường|phuong|thị\s*trấn|thi\s*tran)\s+",
    re.IGNORECASE,
)

_LEVEL_MAP = {
    "tỉnh": "province", "tinh": "province",
    "thành phố": "area", "thanh pho": "area", "tp": "area",
    "huyện": "area", "huyen": "area",
    "xã": "commune", "xa": "commune",
    "phường": "ward", "phuong": "ward",
    "thị trấn": "ward", "thi tran": "ward",
}


def _extract_entity_candidates(query: str) -> List[Tuple[str, str]]:
    """Extract (name, hinted_level) pairs from query text."""
    candidates: List[Tuple[str, str]] = []
    for m in _ADMIN_PREFIX_RE.finditer(query):
        prefix_raw = m.group(1).strip().lower()
        prefix_norm = _strip_diacritics(prefix_raw)
        level = "unknown"
        for k, v in _LEVEL_MAP.items():
            if _strip_diacritics(k) == prefix_norm or prefix_norm.startswith(_strip_diacritics(k)):
                level = v
                break

        rest = query[m.end():].strip()
        name_match = re.match(
            r"([A-ZÀ-Ỹa-zà-ỹ\s]+?)"
            r"(?:\s+(?:có|co|là|la|ở|o|thuộc|thuoc|bao|diện|dien|dân|dan"
            r"|mới|moi|sau|nào|nao|gì|gi|nằm|nam|phát|phat|được|duoc)"
            r"|[,?.!]|$)",
            rest,
        )
        if name_match:
            name = name_match.group(1).strip()
            if len(name) >= 2:
                candidates.append((name, level))
    return candidates


# ── Core resolution ────────────────────────────────────────

def resolve_entity(query: str) -> ResolvedEntity:
    """Resolve the primary location entity from a user query.

    Hierarchy after 2025 merger: Tỉnh → Xã/Phường (no district level).
    Old district names are recognized as "area" for search context.
    """
    candidates = _extract_entity_candidates(query)
    q_norm = _strip_diacritics(query)

    # Best known-name substring match – prefer specific (commune/ward) over broad (area)
    _KIND_PRIORITY = {"commune": 3, "ward": 3, "landmark": 2, "area": 1}
    best_match: Optional[Tuple[str, str, str, int, int]] = None  # + priority
    for norm_name, (real_name, area_hint, kind) in _all_names.items():
        if norm_name in q_norm and len(norm_name) >= 3:
            pri = _KIND_PRIORITY.get(kind, 0)
            score = pri * 100 + len(norm_name)
            if best_match is None or score > best_match[4]:
                best_match = (real_name, area_hint, kind, len(norm_name), score)

    # Pass 1: prefix candidates that resolve to known entities
    for name, hinted_level in candidates:
        name_norm = _strip_diacritics(name)
        if name_norm in _commune_index:
            info = _commune_index[name_norm]
            kind = "ward" if info["type"] == "phuong" else "commune"
            return _build_result(info["name"], kind, info["area"], 0.95, query)
        if name_norm in _area_aliases:
            area_name = _area_aliases[name_norm]
            return _build_result(area_name, "area", "", 0.90, query)
        for n, (real, _, _) in _all_names.items():
            if name_norm == n:
                return _build_result(real, hinted_level or "unknown", "", 0.85, query)

    # Pass 2: known-name substring match
    if best_match:
        name, area_hint, kind, _len, _score = best_match
        return _build_result(name, kind, area_hint, 0.8, query)

    # Pass 3: prefix candidates with unknown entities
    for name, hinted_level in candidates:
        if len(name) >= 3:
            return _build_result(name, hinted_level, "", 0.5, query)

    # Fallback: province level
    return ResolvedEntity(
        name=PROVINCE, level="province", confidence=0.4,
        search_queries=_generate_search_queries(PROVINCE, "province", "", query),
    )


def _build_result(
    name: str, level: str, area: str, confidence: float, query: str,
) -> ResolvedEntity:
    return ResolvedEntity(
        name=name, level=level, area=area, confidence=confidence,
        search_queries=_generate_search_queries(name, level, area, query),
    )


# ── Search query generation ────────────────────────────────

def _generate_search_queries(
    name: str, level: str, area: str, original_query: str,
) -> List[str]:
    queries: List[str] = []
    loc = f"{name} tỉnh Ninh Bình"
    if area:
        loc = f"{name} khu vực {area} tỉnh Ninh Bình"

    level_vi = {
        "commune": "xã", "ward": "phường", "area": "khu vực", "province": "tỉnh",
    }.get(level, "")
    full_name = f"{level_vi} {name}".strip() if level_vi else name

    queries.append(f"{original_query} {loc}")

    topics = _detect_topics(original_query)
    if topics:
        for topic in topics[:3]:
            queries.append(f"{full_name} {loc} {topic}")
    else:
        queries.append(f"{full_name} {loc} thông tin hành chính kinh tế")
        queries.append(f"{full_name} {loc} du lịch văn hóa")

    name_ascii = _strip_diacritics(name)
    queries.append(f"{name_ascii} Ninh Binh Vietnam")
    return queries[:5]


_TOPIC_PATTERNS = {
    "kinh tế phát triển": re.compile(r"kinh\s*tế|phát\s*triển|sản\s*xuất|nông\s*nghiệp|công\s*nghiệp|development", re.I),
    "du lịch tham quan": re.compile(r"du\s*lịch|tham\s*quan|đặc\s*sản|lễ\s*hội|di\s*tích|danh\s*lam|tourism", re.I),
    "dân số diện tích": re.compile(r"dân\s*số|diện\s*tích|bao\s*nhiêu|population|area", re.I),
    "hạ tầng giao thông": re.compile(r"hạ\s*tầng|giao\s*thông|trường|bệnh\s*viện|đường|infrastructure", re.I),
    "hành chính sáp nhập": re.compile(r"hành\s*chính|sáp\s*nhập|thuộc|ở\s+đâu|administrative", re.I),
    "tin tức mới nhất": re.compile(r"tin\s*tức|mới\s*nhất|cập\s*nhật|gần\s*đây|news|latest", re.I),
}


def _detect_topics(query: str) -> List[str]:
    topics = []
    for topic, pattern in _TOPIC_PATTERNS.items():
        if pattern.search(query):
            topics.append(topic)
    return topics


# ── Display ────────────────────────────────────────────────

def get_entity_display_name(entity: ResolvedEntity) -> str:
    """Format: 'Xã Gia Sinh, khu vực Gia Viễn, tỉnh Ninh Bình'."""
    parts = []
    level_vi = {
        "commune": "Xã", "ward": "Phường", "area": "Khu vực", "province": "Tỉnh",
    }
    name = entity.name

    name_stripped = re.sub(
        r"^(Tỉnh|Thành phố|Huyện|Xã|Phường|Thị trấn|Khu vực)\s+",
        "", name, flags=re.IGNORECASE,
    ).strip() or name

    if entity.level in level_vi:
        parts.append(f"{level_vi[entity.level]} {name_stripped}")
    else:
        parts.append(name)
    if entity.area:
        parts.append(f"khu vực {entity.area}")
    parts.append(f"tỉnh {entity.province}")
    return ", ".join(parts)
