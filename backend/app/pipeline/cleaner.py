"""Input text preprocessing for Vietnamese legal documents."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

OCR_REPLACEMENTS: List[tuple[str, str]] = [
    (r"ĐỔ#", "ĐỔI"),
    (r"\bQUYẾTĐỊNH\b", "QUYẾT ĐỊNH"),
    (r"\bNGHỊĐỊNH\b", "NGHỊ ĐỊNH"),
    (r"\bTHÔNGTƯ\b", "THÔNG TƯ"),
    (r"\bBỘTRƯỞNG\b", "BỘ TRƯỞNG"),
    (r"\bBỘVĂN\b", "BỘ VĂN"),
    (r"CHƯ>", "CHƯƠNG"),
    # OCR variants for "Điều"
    (r"Đ6ều", "Điều"),
    (r"Điề6", "Điều"),
    (r"Điề-", "Điều"),
    (r"ĐIỀ-", "ĐIỀU"),
    (r"\bD[i1]ều\b", "Điều"),
    (r"\b[ĐÐ]i[eê]u\b", "Điều"),
    (r"\bChuong\b", "Chương"),
]

_HEADER_FOOTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*Trang\s+\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Page\s+\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$"),
    re.compile(r"^\s*CỘNG\s+HÒA\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM\s*$", re.IGNORECASE),
    re.compile(r"^\s*Độc\s+lập\s*-\s*Tự\s+do\s*-\s*Hạnh\s+phúc\s*$", re.IGNORECASE),
    re.compile(r"^\s*_{3,}\s*$"),
    re.compile(r"^\s*\.{4,}\s*$"),
)

_SIGNATURE_HINTS = ("nơi nhận", "tm.", "kt.", "tuq.", "tl.", "ký tên", "đã ký")

DIEU_HEADER_RE = re.compile(r"(?im)(^|\n)\s*(Điều)\s+(\d+)\s*(?:[.\):\-])?\s*(.*)$")


def normalize_text(s: str) -> str:
    """Normalize Unicode/OCR issues and strip escape/control noise."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", s)
    s = re.sub(r"\\[abtnfrv]", " ", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    for pat, rep in OCR_REPLACEMENTS:
        s = re.sub(pat, rep, s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_block(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s or "")
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def remove_headers_footers(text: str) -> str:
    """Remove recurring page headers, footers and signature stubs."""
    if not text:
        return ""
    kept: list[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            kept.append("")
            continue
        if any(pattern.match(candidate) for pattern in _HEADER_FOOTER_PATTERNS):
            continue
        lower_line = candidate.lower()
        if any(hint in lower_line for hint in _SIGNATURE_HINTS) and len(candidate) < 140:
            continue
        kept.append(line)
    return "\n".join(kept)


def extract_quyet_dinh_intro_block(text_raw: str) -> Optional[str]:
    """Extract block from 'QUYẾT ĐỊNH:' until before first 'Điều n'."""
    t = normalize_text(text_raw)
    m = re.search(r"(?:^|\n)\s*QUYẾT\s+ĐỊNH\s*:\s*", t, flags=re.IGNORECASE)
    if not m:
        return None
    after = t[m.end() :]
    m2 = re.search(r"(?:^|\n)\s*Điều\s+\d+\b", after, flags=re.IGNORECASE)
    end = m.end() + (m2.start() if m2 else len(after))
    return clean_block(t[m.start() : end])


def extract_dieu_blocks(text_raw: str, max_dieu: int = 500) -> List[Dict]:
    """Extract all Điều blocks and keep clean boundaries."""
    t = normalize_text(text_raw)
    matches = [m for m in DIEU_HEADER_RE.finditer(t) if 1 <= int(m.group(3)) <= max_dieu]
    if not matches:
        return []

    result: List[Dict] = []
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start(0) if idx + 1 < len(matches) else len(t)
        block_text = t[m.start(0) : end].strip()
        result.append(
            {
                "dieu": int(m.group(3)),
                "title": (m.group(4) or "").strip(),
                "content": clean_block(block_text),
            }
        )
    return result


def preprocess_input_text(raw_text: str) -> str:
    """Preprocess legal input text before structure detection."""
    t = normalize_text(raw_text)
    t = remove_headers_footers(t)
    t = clean_block(t)

    # If OCR/noise heavy docs still contain clean Điều markers, rebuild text by Điều blocks.
    dieu_blocks = extract_dieu_blocks(t)
    if dieu_blocks:
        intro = extract_quyet_dinh_intro_block(t)
        body = "\n\n".join(block["content"] for block in dieu_blocks)
        if intro:
            return clean_block(f"{intro}\n\n{body}")
        return clean_block(body)
    return t


def join_chunks(chunks: List[Dict]) -> str:
    """Join OCR chunks sorted by page/chunk order."""
    chunks_sorted = sorted(chunks, key=lambda x: (x.get("page", 10**9), x.get("chunk_id", 10**9)))
    return "\n".join(c.get("text", "") for c in chunks_sorted)


def extract_qd_and_dieu_from_file(file_obj: Dict) -> Dict:
    """Utility used when input source is pre-split chunk list."""
    full_text_raw = join_chunks(file_obj.get("chunks", []))
    return {
        "name": file_obj.get("name"),
        "quyet_dinh_intro": extract_quyet_dinh_intro_block(full_text_raw),
        "dieu_list": extract_dieu_blocks(full_text_raw),
    }


# Backward-compatible names used by existing modules
def normalize_unicode(text: str) -> str:
    return normalize_text(text)


def remove_control_characters(text: str) -> str:
    return normalize_text(text)


def normalize_whitespace(text: str) -> str:
    return clean_block(text)


def clean_text(raw: str) -> str:
    return preprocess_input_text(raw)
