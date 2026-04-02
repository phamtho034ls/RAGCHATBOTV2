"""Đọc cấu hình intent từ YAML: structural fallback, routing (query_intent), prototype bổ sung.

Đường dẫn: ``INTENT_PATTERNS_YAML`` (.env) hoặc mặc định ``app/intent_patterns/routing.yaml``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "intent_patterns" / "routing.yaml"


def _resolved_yaml_path(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    try:
        from app.config import INTENT_PATTERNS_YAML

        return Path(INTENT_PATTERNS_YAML)
    except Exception:
        return _DEFAULT_YAML

# (intent_id, confidence, tuple of compiled patterns) — đã sắp theo priority giảm dần
_structural_rules: List[Tuple[str, float, Tuple[re.Pattern[str], ...]]] = []
_routing_compiled: Dict[str, List[re.Pattern[str]]] = {}
_flag_overrides: List[Dict[str, object]] = []
_prototype_sentences: Dict[str, List[str]] = {}
_loaded = False


def _builtin_structural() -> List[dict]:
    """Giữ đồng bộ với routing.yaml mặc định — dùng khi không có file."""
    return [
        {"intent_id": "article_query", "priority": 100, "confidence": 0.97, "patterns": [
            r"điều\s+\d+\s+(luật|nghị\s*định|thông\s*tư|quyết\s*định|pháp\s*lệnh|chỉ\s*thị)",
        ]},
        {"intent_id": "article_query", "priority": 99, "confidence": 0.97, "patterns": [
            r"khoản\s+\d+\s+điều\s+\d+",
        ]},
        {"intent_id": "article_query", "priority": 98, "confidence": 0.95, "patterns": [
            r"^\s*điều\s+\d+[\s,\.]*$",
        ]},
        {"intent_id": "article_query", "priority": 97, "confidence": 0.95, "patterns": [
            r"\b\d{1,3}/\d{4}/[A-ZĐa-zđ\-]+\b",
        ]},
        {"intent_id": "article_query", "priority": 96, "confidence": 0.93, "patterns": [
            r"\b\d{1,3}/[A-ZĐa-zđ]{2,}-[A-ZĐa-zđ]{2,}\b",
        ]},
        {"intent_id": "tom_tat_van_ban", "priority": 90, "confidence": 0.95, "patterns": [
            r"(tóm\s*tắt|tổng\s*hợp|khái\s*quát)\s+(nghị\s*định|luật|thông\s*tư|quyết\s*định|chỉ\s*thị)",
        ]},
        {"intent_id": "so_sanh_van_ban", "priority": 89, "confidence": 0.95, "patterns": [
            r"so\s*sánh\s+.{0,30}(luật|nghị\s*định|thông\s*tư).{0,30}(và|với|so\s*với)",
        ]},
        {"intent_id": "soan_thao_van_ban", "priority": 88, "confidence": 0.95, "patterns": [
            r"(soạn|viết)\s+(công\s*văn|tờ\s*trình|biên\s*bản|quyết\s*định|thông\s*báo|đơn\s*xin)",
            r"(lập|tạo)\s+(công\s*văn|tờ\s*trình|quyết\s*định|thông\s*báo|đơn\s*xin)",
        ]},
        {"intent_id": "tao_bao_cao", "priority": 87, "confidence": 0.95, "patterns": [
            r"(tạo|viết|lập|soạn)\s+(báo\s*cáo)",
        ]},
        {"intent_id": "kiem_tra_ho_so", "priority": 86, "confidence": 0.95, "patterns": [
            r"(đã\s+nộp|đã\s+có|đã\s+gửi).{0,50}(còn\s+thiếu|thiếu\s+gì|đủ\s+chưa)",
        ]},
        {"intent_id": "huong_dan_thu_tuc", "priority": 86, "confidence": 0.93, "patterns": [
            r"(hướng\s*dẫn|thủ\s*tục).{0,60}(hồ\s*sơ|mẫu|nộp|ubnd|một\s*cửa)",
            r"\bubnd\b|ủy\s*ban\s*nhân\s*dân",
        ]},
        {"intent_id": "document_meta_relation", "priority": 85, "confidence": 0.95, "patterns": [
            r"(ban\s*hành|có\s*hiệu\s*lực|hết\s*hiệu\s*lực)\s+(ngày|năm|khi|từ)\s+nào",
        ]},
        {"intent_id": "document_meta_relation", "priority": 84, "confidence": 0.93, "patterns": [
            r"(ai|cơ\s*quan\s*nào)\s+(ban\s*hành|ký\s*ban\s*hành|phê\s*duyệt)",
            r"(sửa\s*đổi|thay\s*thế|bãi\s*bỏ).{0,40}(văn\s*bản|luật|nghị\s*định)",
        ]},
    ]


def _compile_routing(raw: Dict[str, List[str]]) -> Dict[str, List[re.Pattern[str]]]:
    out: Dict[str, List[re.Pattern[str]]] = {}
    for name, pats in (raw or {}).items():
        if not isinstance(pats, list):
            continue
        compiled: List[re.Pattern[str]] = []
        for p in pats:
            if isinstance(p, str) and p.strip():
                try:
                    compiled.append(re.compile(p, re.IGNORECASE))
                except re.error as exc:
                    log.warning("Skip invalid routing pattern [%s]: %s — %s", name, p[:60], exc)
        if compiled:
            out[name] = compiled
    return out


def load_intent_pattern_config(
    path: Optional[Path] = None,
    *,
    force: bool = False,
) -> None:
    """Load YAML một lần; gọi từ startup hoặc test (force=True)."""
    global _structural_rules, _routing_compiled, _flag_overrides, _prototype_sentences, _loaded

    if _loaded and not force:
        return

    yaml_path = _resolved_yaml_path(path)

    structural_raw: List[dict] = []
    routing_raw: Dict[str, List[str]] = {}
    proto_raw: Dict[str, List[str]] = {}
    flag_raw: List[dict] = []

    try:
        import yaml  # type: ignore
    except ImportError:
        log.warning("PyYAML chưa cài — dùng structural/routing built-in.")
        yaml = None  # type: ignore

    if yaml is not None and yaml_path.is_file():
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            structural_raw = list(data.get("structural_rules") or [])
            rr = data.get("routing") or {}
            routing_raw = {k: list(v) for k, v in rr.items() if isinstance(v, list)}
            pr = data.get("prototype_sentences") or {}
            proto_raw = {k: list(v) for k, v in pr.items() if isinstance(v, list)}
            flag_raw = list(data.get("flag_overrides") or [])
            log.info("Loaded intent patterns from %s", yaml_path)
        except Exception as exc:
            log.error("Failed to load %s: %s — built-in fallback", yaml_path, exc)
            structural_raw = []
            routing_raw = {}
            proto_raw = {}
            flag_raw = []

    if not structural_raw:
        structural_raw = _builtin_structural()

    # Sắp: priority giảm dầu, giữ thứ tự file khi priority bằng nhau
    indexed = list(enumerate(structural_raw))
    indexed.sort(
        key=lambda it: (-int(it[1].get("priority", 0)), it[0]),
    )

    built_struct: List[Tuple[str, float, Tuple[re.Pattern[str], ...]]] = []
    for _, row in indexed:
        intent = str(row.get("intent_id") or "").strip()
        if not intent:
            continue
        conf = float(row.get("confidence", 0.9))
        pats_in: List[str] = []
        for p in row.get("patterns") or []:
            if isinstance(p, str) and p.strip():
                pats_in.append(p.strip())
        if not pats_in:
            continue
        compiled_row: List[re.Pattern[str]] = []
        for p in pats_in:
            try:
                compiled_row.append(re.compile(p, re.IGNORECASE))
            except re.error as exc:
                log.warning("Skip invalid structural pattern for %s: %s — %s", intent, p[:60], exc)
        if compiled_row:
            built_struct.append((intent, conf, tuple(compiled_row)))

    _structural_rules = built_struct
    _routing_compiled = _compile_routing(routing_raw) if routing_raw else _default_routing_compiled()
    _flag_overrides = _compile_flag_overrides(flag_raw)
    _prototype_sentences = {k: [str(s) for s in v if s] for k, v in proto_raw.items()}
    _loaded = True


def _compile_flag_overrides(raw: List[dict]) -> List[Dict[str, object]]:
    compiled: List[Dict[str, object]] = []
    for row in raw or []:
        if not isinstance(row, dict):
            continue
        patterns = [str(p).strip() for p in (row.get("patterns") or []) if str(p).strip()]
        if not patterns:
            continue
        set_flags = row.get("set_flags") or {}
        if not isinstance(set_flags, dict):
            continue
        cpatterns: List[re.Pattern[str]] = []
        for p in patterns:
            try:
                cpatterns.append(re.compile(p, re.IGNORECASE))
            except re.error as exc:
                log.warning("Skip invalid flag override pattern: %s — %s", p[:80], exc)
        if not cpatterns:
            continue
        compiled.append(
            {
                "name": str(row.get("name") or "").strip(),
                "patterns": tuple(cpatterns),
                "set_flags": {
                    "is_legal_lookup": bool(set_flags.get("is_legal_lookup", False)),
                    "use_multi_article": bool(set_flags.get("use_multi_article", False)),
                    "needs_expansion": bool(set_flags.get("needs_expansion", False)),
                },
            }
        )
    return compiled


def _default_routing_compiled() -> Dict[str, List[re.Pattern[str]]]:
    """Routing mặc định (trùng routing.yaml) nếu YAML không có khóa routing."""
    raw: Dict[str, List[str]] = {
        "multi_doc_synthesis": [
            r"tổng\s+hợp", r"so\s+sánh", r"đối\s+chiếu", r"nêu\s+điểm\s+khác",
            r"giữa\s+.+\s+và\s+", r"và\s+các\s+nghị\s+định", r"nghị\s+định\s+hướng\s+dẫn",
            r"nằm\s+ở\s+những", r"những\s+luật", r"luật/nghị", r"luật\s*/\s*nghị",
            r"thông\s+tư\s+nào", r"nghị\s+định\s+nào",
            r"các\s+nghị\s+định.{0,120}hiện\s+hành", r"theo\s+các\s+văn\s+bản",
            r"văn\s+bản\s+pháp\s+luật.{0,160}điều\s+chỉnh",
            r"các\s+văn\s+bản\s+nào\s+quy\s+định", r"những\s+văn\s+bản\s+nào\s+quy\s+định",
            r"văn\s+bản\s+nào\s+quy\s+định", r"quy\s+định.{0,100}nằm\s+ở\s+những",
        ],
        "checklist_documents": [
            r"liệt kê", r"danh sách", r"các văn bản", r"những văn bản", r"thống kê",
            r"bao nhiêu văn bản", r"cho biết các", r"kể tên", r"check\s*list", r"checklist",
            r"văn bản liên quan", r"văn bản nào", r"có những văn bản",
        ],
        "substantive_expansion": [
            r"mức\s+(xử\s+)?phạt", r"mức\s+tiền\s+phạt", r"tiền\s+phạt",
            r"hình\s+thức\s+xử\s+phạt", r"chế\s+tài",
            r"hành\s+vi\s+.*(cấm|nghiêm\s+cấm|vi\s+phạm)", r"nghiêm\s+cấm",
            r"\bcác\s+bước\b", r"quy\s+trình", r"trình\s+tự", r"thủ\s+tục",
            r"\bhồ\s+sơ\b", r"giấy\s+tờ\s*(cần|có|phải)?", r"thành\s+phần\s+hồ\s+sơ",
            r"nêu\s+đầy\s+đủ", r"trình\s+bày\s+chi\s+tiết", r"hướng\s+dẫn\s+chi\s+tiết",
            r"biện\s+pháp\s+phòng\s+ngừa", r"tiếp\s+nhận", r"xác\s+minh",
            r"can\s+thiệp", r"tin\s+báo", r"xâm\s+hại", r"bạo\s+hành",
            r"phối\s+hợp\s+liên\s+ngành", r"tu\s+bổ|trùng\s+tu",
            r"đăng\s+ký\s+.*(tổ\s+chức\s+)?lễ\s+hội",
            r"kiểm\s+tra\s+.*cơ\s+sở\s+kinh\s+doanh", r"vai\s+trò\s+của\s+ubnd",
        ],
        "consultation_advisory": [
            r"\btham\s*mưu\b", r"\btư\s*vấn\b", r"\bphối\s*hợp\b", r"ông/bà",
            r"\bông\s+bà\b", r"\bcán\s+bộ\b", r"\bubnd\b", r"ủy\s*ban\s*nhân\s*dân",
            r"\bđảng\s*ủy\b", r"\bhđnd\b", r"hội\s*đồng\s*nhân\s*dân",
            r"kế\s*hoạch", r"nghị\s*quyết", r"xử\s*lý\s*tình\s*huống",
            r"xin\s*chỉ\s*đạo", r"nên\s+làm\s+gì", r"làm\s+thế\s+nào\s+để",
            r"đảm\s*bảo\s+an\s+ninh", r"an\s+ninh\s+trật\s+tự", r"truyền\s+dạy",
            r"bảo\s+tồn\s+di\s+sản", r"địa\s+phương\s+có", r"tham\s*mưu\s+cho",
            r"ban\s+hành\s+nghị\s*quyết",
        ],
        "multi_article_boost_substantive": [
            r"mức\s+(xử\s+)?phạt|hành\s+vi\s+.*cấm|nghiêm\s+cấm|biện\s+pháp\s+phòng\s+ngừa",
        ],
    }
    return _compile_routing(raw)


def structural_match(query_lower: str) -> Optional[Tuple[str, float]]:
    """Khớp structural (đã lower)."""
    load_intent_pattern_config()
    q = (query_lower or "").strip()
    if not q:
        return None
    for intent, conf, patterns in _structural_rules:
        if any(p.search(q) for p in patterns):
            return intent, conf
    return None


def routing_group_matches(query: str, group: str) -> bool:
    """True nếu một pattern trong nhóm ``group`` khớp ``query``."""
    load_intent_pattern_config()
    q = (query or "").lower()
    if not q:
        return False
    for p in _routing_compiled.get(group, ()):
        if p.search(q):
            return True
    return False


def get_prototype_sentences_extra() -> Dict[str, List[str]]:
    """Câu prototype thêm từ YAML (merge vào INTENT_PROTOTYPES)."""
    load_intent_pattern_config()
    return dict(_prototype_sentences)


def get_flag_override_set_flags(query: str) -> Optional[Dict[str, bool]]:
    """Trả set_flags của override đầu tiên khớp query; None nếu không khớp."""
    load_intent_pattern_config()
    q = (query or "").lower()
    if not q:
        return None
    for row in _flag_overrides:
        pats = row.get("patterns") or ()
        if any(p.search(q) for p in pats):  # type: ignore[attr-defined]
            return dict(row.get("set_flags") or {})
    return None
