"""Document ingestion pipeline orchestrator.

Full pipeline:
  DOCX → Text extraction → Cleaning → Structure detection
  → Chunking → Embedding → PostgreSQL metadata → Qdrant vectors

This module ties all pipeline stages together.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.parser.docx_parser import read_docx
from app.pipeline.cleaner import clean_text
from app.pipeline.chunk_generator import generate_clause_chunks
from app.pipeline.db_writer import (
    insert_articles,
    insert_chapters,
    insert_chunks,
    insert_clauses,
    insert_document,
    insert_sections_from_articles,
)
from app.pipeline.embedding import embed_texts
from app.pipeline.legal_parser import build_article_tree
from app.pipeline.vector_store import upsert_vectors
from app.services.domain_classifier import classify_document_domain

log = logging.getLogger(__name__)

# ── Document-number extraction from filename & text body ───

_DOC_NUM_FILENAME_UNDERSCORE = re.compile(
    r"^\s*(\d+)_(\d{4})_([A-Za-z][A-Za-z0-9\-]*)(?:_\d+)*\s*$"
)
_DOC_NUM_FILENAME_SLASH = re.compile(r"(\d+/\d{4}/[A-Za-z0-9\-]+)")

# Vietnamese-style filenames: "Nghị định 382021NĐ-CP ngày ..."
# Extracts: 382021NĐ-CP → 38/2021/NĐ-CP, or 542QĐ-UBND → 542/QĐ-UBND
_DOC_NUM_FILENAME_VIET = re.compile(
    r"(\d{1,4})"               # number: 38, 542, 706
    r"(\d{4})?"                # optional year: 2021 (4 digits)
    r"([A-ZĐ]{1,4})"          # code: NĐ, QĐ, TT, CT, NQ
    r"[-–]"                    # separator
    r"([A-Z0-9Đ]{1,20})"      # suffix: CP, UBND, BVHTTDL
    r"(?:\s|$|[^A-Z0-9])"     # boundary
)

_DOC_NUM_TEXT_BODY = re.compile(r"""
    (?im)
    \b(?P<prefix>Luật\s*số|Số)\s*:\s*
    (?P<so>
        \d{1,4}
        (?:\s*/\s*\d{4})?
        \s*/\s*
        [A-ZĐ]{1,6}
        (?:\s*-\s*[A-Z0-9Đ]{1,20})*
        (?:\s*\d{1,2})?
    )
""", re.VERBOSE)

_ISSUER_PATTERN = re.compile(
    r"^(BỘ|CHÍNH PHỦ|QUỐC HỘI|ỦY BAN|ỦY\s+BAN|UBND|HỘI ĐỒNG|TÒA ÁN|VIỆN KIỂM SÁT)\b",
    re.IGNORECASE,
)

# Giá trị placeholder từ Swagger / form (giống document_router_v2)
_METADATA_PLACEHOLDERS = frozenset(
    {"", "string", "null", "none", "undefined", "n/a", "na", "-"}
)


def _clean_metadata_field(value: Optional[str]) -> str:
    if value is None:
        return ""
    s = value.strip()
    if not s or s.lower() in _METADATA_PLACEHOLDERS:
        return ""
    return s

_DOCTYPES_FROM_CODE = {
    "QH": "Luật",
    "NQ": "Nghị quyết",
    "ND": "Nghị định",
    "NĐ": "Nghị định",
    "TT": "Thông tư",
    "QD": "Quyết định",
    "QĐ": "Quyết định",
    "CT": "Chỉ thị",
    "PL": "Pháp lệnh",
}

_ISSUED_DATE_PATTERN = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE,
)
_EFFECTIVE_DATE_PATTERN = re.compile(
    r"(?:có hiệu lực(?: thi hành)?(?: kể)?(?: từ)?|hiệu lực từ)\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE,
)


def _extract_doc_number(filename: str, text: str = "") -> str:
    """Extract legal document number (số hiệu văn bản).

    Priority:
    1. Filename underscore: 13_2025_TT-BVHTTDL → 13/2025/TT-BVHTTDL
    2. Filename slash: 13/2025/TT-BVHTTDL
    3. Filename Vietnamese: Nghị định 382021NĐ-CP ngày... → 38/2021/NĐ-CP
    4. Text body: "Số: 13/2025/TT-BVHTTDL" or "Luật số: 81/QH15"
    5. Fallback: filename stem (truncated to 100 chars)
    """
    stem = Path(filename).stem

    m = _DOC_NUM_FILENAME_UNDERSCORE.match(stem.strip())
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"

    m2 = _DOC_NUM_FILENAME_SLASH.search(stem)
    if m2:
        return m2.group(1)

    m3 = _DOC_NUM_FILENAME_VIET.search(stem)
    if m3:
        num, year, code, suffix = m3.group(1), m3.group(2), m3.group(3), m3.group(4)
        if year:
            return f"{num}/{year}/{code}-{suffix}"
        return f"{num}/{code}-{suffix}"

    m4 = _DOC_NUM_TEXT_BODY.search(text[:3000])
    if m4:
        raw = m4.group("so")
        return re.sub(r"\s+", "", raw)

    return stem[:100]


def extract_all_doc_numbers(text: str) -> list[str]:
    """Extract all unique legal document numbers from full text.

    Returns deduplicated list preserving first-occurrence order.
    Handles: "Số: 13/2025/TT-BVHTTDL", "Luật số: 81/QH15", etc.
    """
    seen: set[str] = set()
    result: list[str] = []
    for m in _DOC_NUM_TEXT_BODY.finditer(text):
        raw = m.group("so")
        normalized = re.sub(r"\s+", "", raw)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _extract_document_type(filename: str, text: str, detected_type: str, doc_number: str = "") -> str:
    # Ưu tiên code trong số hiệu vì ổn định hơn tìm theo từ khóa toàn văn.
    num = (doc_number or "").upper()
    if "/" in num:
        tail = num.split("/")[-1]
        code_part = re.split(r"[-\s]", tail)[0]
        for code, doc_type in _DOCTYPES_FROM_CODE.items():
            if code_part.startswith(code):
                return doc_type

    stem = Path(filename).stem.upper()
    parts = stem.split("_")
    if len(parts) >= 3:
        code_part = parts[2]
        for code, doc_type in _DOCTYPES_FROM_CODE.items():
            if code in code_part:
                return doc_type

    first_lines = [ln.strip().upper() for ln in text.splitlines()[:60] if ln.strip()]
    for line in first_lines:
        compact = re.sub(r"[\s\-–—]+", " ", line).strip()
        if compact == "CHI THI" or compact == "CHỈ THỊ":
            return "Chỉ thị"
        if compact == "THONG TU" or compact == "THÔNG TƯ":
            return "Thông tư"
        if compact == "NGHI DINH" or compact == "NGHỊ ĐỊNH":
            return "Nghị định"
        if compact == "NGHI QUYET" or compact == "NGHỊ QUYẾT":
            return "Nghị quyết"
        if compact == "QUYET DINH" or compact == "QUYẾT ĐỊNH":
            return "Quyết định"
        if compact == "LUAT" or compact == "LUẬT":
            return "Luật"

    first_1500 = text[:1500].upper()
    if " CHỈ THỊ" in f" {first_1500}":
        return "Chỉ thị"
    if "NGHỊ ĐỊNH" in first_1500:
        return "Nghị định"
    if "THÔNG TƯ" in first_1500:
        return "Thông tư"
    if "NGHỊ QUYẾT" in first_1500:
        return "Nghị quyết"
    if "QUYẾT ĐỊNH" in first_1500:
        return "Quyết định"
    if " LUẬT" in f" {first_1500}":
        return "Luật"
    return detected_type or "Văn bản"


def _is_header_noise_line(line: str) -> bool:
    """Dòng header không phải trích yếu (số văn bản, quốc hiệu, địa điểm–ngày…)."""
    part = line.split("|")[0].strip()
    if not part:
        return True
    u = part.upper()
    if u.startswith("SỐ") or u.startswith("SỐ:") or u.startswith("SO:"):
        return True
    if "CỘNG HÒA" in u:
        return True
    if "ĐỘC LẬP" in u and "TỰ DO" in u:
        return True
    if "ỦY BAN NHÂN DÂN" in u and "VỀ VIỆC" not in u:
        return True
    if _ISSUER_PATTERN.search(part) and "VỀ VIỆC" not in part.upper():
        return True
    if _ISSUED_DATE_PATTERN.search(part) and len(part) < 120:
        return True
    if u in ("CHỈ THỊ", "LUẬT", "NGHỊ ĐỊNH", "THÔNG TƯ", "NGHỊ QUYẾT", "QUYẾT ĐỊNH"):
        return True
    return False


def _line_equals_doc_type_label(line: str, doc_type: str) -> bool:
    if not doc_type:
        return False
    left = re.sub(r"\s+", " ", line.split("|")[0].strip()).upper()
    dt = re.sub(r"\s+", " ", doc_type.strip()).upper()
    return left == dt


def _looks_like_clause_or_article_line(line: str) -> bool:
    """Tránh nhầm '1. Phạt tiền...' (khoản Điều 1) với trích yếu."""
    s = line.split("|")[0].strip()
    if re.match(r"^\d{1,3}[\.\)]\s+\S", s):
        return True
    if re.match(r"(?i)^điều\s+\d+", s):
        return True
    if re.match(r"(?i)^chương\s+[ivxlcdm\d]+", s):
        return True
    return False


def _extract_trich_yeu(text: str, doc_type: str) -> str:
    """Trích yếu: 'VỀ VIỆC …', 'V/v …', hoặc dòng chủ đề ngay dưới nhãn loại văn bản (LUẬT …)."""
    lines = [ln.strip() for ln in text.splitlines()[:120] if ln.strip()]

    for line in lines:
        core = line.split("|")[0].strip()
        if not core or _is_header_noise_line(core):
            continue
        if re.match(r"(?i)^v[v／/]\s*\.\s*", core) or re.match(r"(?i)^v/v\b", core):
            return core
        if re.match(r"(?i)^về\s+việc\b", core):
            return core

    for idx, line in enumerate(lines):
        core = line.split("|")[0].strip()
        if _line_equals_doc_type_label(core, doc_type):
            j = idx + 1
            while j < len(lines):
                nxt = lines[j].split("|")[0].strip()
                j += 1
                if not nxt or _is_header_noise_line(nxt):
                    continue
                if _looks_like_clause_or_article_line(nxt):
                    continue
                if len(nxt) >= 2:
                    return nxt
            break

    for line in lines:
        core = line.split("|")[0].strip()
        u = core.upper()
        if "VỀ VIỆC" in u and len(core) >= 12:
            return core

    return ""


def _compact_doc_number_for_title(doc_number: str) -> str:
    """06/CT-UBND -> 06CT-UBND; 09/2017/QH14 -> 092017QH14."""
    if not doc_number:
        return ""
    s = re.sub(r"\s+", "", doc_number.strip())
    return s.replace("/", "")


def _subject_tail_for_title(subject: str) -> str:
    """Bỏ tiền tố V/v, Về việc; đưa về chữ thường cho phần cuối title."""
    s = subject.strip()
    s = re.sub(r"(?i)^v[v／/]\s*\.\s*", "", s)
    s = re.sub(r"(?i)^v/v\s+", "", s)
    s = re.sub(r"(?i)^về\s+việc\s+", "", s)
    return s.strip().lower()


def _build_canonical_title(
    doc_type: str,
    doc_number: str,
    issued: Optional[date],
    trich_yeu: str,
) -> Optional[str]:
    """VD: 'Chỉ thị 06CT-UBND ngày 22052025 tăng cường ...'."""
    compact = _compact_doc_number_for_title(doc_number)
    if not compact or not doc_type:
        return None
    parts: list[str] = [doc_type.strip(), compact]
    if issued:
        parts.append(f"ngày {issued.day:02d}{issued.month:02d}{issued.year}")
    tail = _subject_tail_for_title(trich_yeu)
    if tail:
        parts.append(tail)
    return " ".join(parts)


# ── Main pipeline ─────────────────────────────────────────

async def ingest_document(
    file_path: str | Path,
    db: AsyncSession,
    issuer: str = "",
    issued_date: Optional[date] = None,
    effective_date: Optional[date] = None,
    title_override: Optional[str] = None,
) -> Dict:
    """Run the full ingestion pipeline for a single DOCX file.

    Returns a summary dict with counts of articles, clauses, chunks.
    """
    file_path = Path(file_path)
    filename = file_path.name
    issuer = _clean_metadata_field(issuer)
    if title_override is not None:
        to = _clean_metadata_field(title_override)
        title_override = to or None

    log.info("▶ [START] Ingesting '%s'...", filename)

    # ── Stage 1: Parse DOCX ──────────────────────────────
    raw_text = read_docx(file_path)
    log.info(
        "  [1/9 PARSE] '%s' → %d chars, %d lines",
        filename, len(raw_text), raw_text.count("\n") + 1,
    )

    # ── Stage 2: Clean text + extract doc_number ─────────
    cleaned = clean_text(raw_text)
    doc_number = _extract_doc_number(filename, cleaned)
    log.info(
        "  [2/9 CLEAN] '%s' → %d chars (delta %+d), doc_number='%s'",
        filename, len(cleaned), len(cleaned) - len(raw_text), doc_number,
    )

    # ── Stage 3: Detect legal structure ──────────────────
    structure = build_article_tree(cleaned)
    doc_type = _extract_document_type(filename, cleaned, structure.document_type, doc_number)
    issued_date_value = issued_date or _extract_issued_date(cleaned)
    effective_date_value = effective_date or _extract_effective_date(cleaned)
    trich_yeu = _extract_trich_yeu(cleaned, doc_type)
    issuer_value = issuer or trich_yeu
    doc_title = title_override
    if not doc_title:
        doc_title = _build_canonical_title(doc_type, doc_number, issued_date_value, trich_yeu)
    if not doc_title:
        doc_title = _title_from_filename(filename) or _extract_title(cleaned, doc_type)

    legal_domain = classify_document_domain(doc_title, cleaned[:1000])
    total_clauses_in_tree = sum(len(a.clauses) for a in structure.articles) if structure.articles else 0
    log.info(
        "  [3/9 STRUCTURE] '%s' → type='%s', domain='%s', articles=%d, clauses_in_tree=%d, "
        "title='%.60s', issuer='%.40s', issued=%s, effective=%s",
        filename, doc_type, legal_domain, len(structure.articles), total_clauses_in_tree,
        doc_title, issuer_value,
        issued_date_value or "N/A", effective_date_value or "N/A",
    )

    if not structure.articles:
        log.error("  [3/9 STRUCTURE] FAILED – no articles detected in '%s'", filename)
        raise ValueError(
            "Không phát hiện được Điều trong văn bản. "
            "Vui lòng kiểm tra lại định dạng file Word hoặc nội dung OCR."
        )

    # ── Stage 4: Save document to PostgreSQL ─────────────
    if len(doc_number) > 255:
        log.warning("  doc_number truncated from %d to 255 chars", len(doc_number))
        doc_number = doc_number[:255]

    doc = await insert_document(
        db,
        doc_number=doc_number,
        title=doc_title,
        document_type=doc_type,
        issuer=issuer_value,
        issued_date=issued_date_value,
        effective_date=effective_date_value,
        file_path=str(file_path),
    )
    document_id = doc.id
    log.info("  [4/9 DB-DOC] '%s' → document_id=%d", filename, document_id)

    # ── Stage 5: Save chapters, sections, articles & clauses ─
    chapter_id_map = await insert_chapters(db, document_id=document_id, chapters=structure.chapters)
    section_id_map = await insert_sections_from_articles(
        db, chapter_id_map=chapter_id_map, articles=structure.articles
    )
    article_id_map = await insert_articles(
        db,
        document_id=document_id,
        articles=structure.articles,
        chapter_id_map=chapter_id_map,
        section_id_map=section_id_map,
    )
    clause_id_map = await insert_clauses(
        db,
        articles=structure.articles,
        article_id_map=article_id_map,
    )
    log.info(
        "  [5/9 DB-ARTICLES] '%s' → articles_saved=%d, clauses_saved=%d",
        filename, len(article_id_map), len(clause_id_map),
    )

    # ── Stage 6: Chunk text ──────────────────────────────
    all_chunks = generate_clause_chunks(
        structure.articles,
        law_title=doc_title,
        doc_number=doc_number,
        document_type=doc_type,
    )

    if not all_chunks:
        log.error("  [6/9 CHUNK] FAILED – no chunks generated for '%s'", filename)
        raise ValueError(
            "Không phát hiện được cấu trúc Điều/Khoản hợp lệ từ văn bản. "
            "Đã hủy ingest để tránh lưu dữ liệu không đầy đủ."
        )

    chunk_total_chars = sum(len(c.text) for c in all_chunks)
    avg_chunk_len = chunk_total_chars // len(all_chunks) if all_chunks else 0
    log.info(
        "  [6/9 CHUNK] '%s' → chunks=%d, total_chars=%d, avg_chunk=%d chars",
        filename, len(all_chunks), chunk_total_chars, avg_chunk_len,
    )

    # ── Stage 7: Generate embeddings ─────────────────────
    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(texts)
    log.info(
        "  [7/9 EMBED] '%s' → vectors=%d, dim=%d",
        filename, len(embeddings), embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
    )

    # ── Stage 8: Upsert to Qdrant ────────────────────────
    payloads = []
    for chunk in all_chunks:
        art_id = article_id_map.get(chunk.article_number) if chunk.article_number else None

        clause_no_for_lookup = chunk.clause_number or ""
        if ".sub." in clause_no_for_lookup:
            clause_no_for_lookup = clause_no_for_lookup.split(".sub.")[0]

        clause_key = None
        if chunk.article_number and clause_no_for_lookup:
            clause_no = clause_no_for_lookup.strip()
            if clause_no.lower().startswith("điểm "):
                clause_key = f"{chunk.article_number}:{clause_no}"
            else:
                clause_key = f"{chunk.article_number}:Khoản {clause_no}"
        cl_id = clause_id_map.get(clause_key) if clause_key else None

        payloads.append({
            "document_id": document_id,
            "article_id": art_id,
            "clause_id": cl_id,
            "law_name": chunk.document_title or doc_title,
            "document_title": chunk.document_title or doc_title,
            "doc_number": doc_number,
            "document_type": doc_type,
            "legal_domain": legal_domain,
            "article_number": chunk.article_number,
            "article_title": chunk.article_title,
            "clause_number": chunk.clause_number,
            "year": chunk.year,
            "chapter": chunk.chapter,
            "section": getattr(chunk, "section", "") or "",
            "chunk_type": getattr(chunk, "chunk_type", "clause"),
            "text_chunk": chunk.text,
        })

    vector_ids = upsert_vectors(embeddings.tolist(), payloads)
    log.info(
        "  [8/9 QDRANT] '%s' → upserted=%d vectors",
        filename, len(vector_ids),
    )

    # ── Stage 9: Save vector_chunks to PostgreSQL ────────
    await insert_chunks(
        db,
        document_id=document_id,
        chunks=all_chunks,
        vector_ids=vector_ids,
        article_id_map=article_id_map,
        clause_id_map=clause_id_map,
    )
    log.info(
        "  [9/9 DB-CHUNKS] '%s' → vector_chunks_saved=%d",
        filename, len(vector_ids),
    )

    summary = {
        "document_id": document_id,
        "doc_number": doc_number,
        "title": doc_title,
        "document_type": doc_type,
        "articles": len(structure.articles),
        "clauses": len(clause_id_map),
        "chunks": len(all_chunks),
    }
    log.info(
        "✔ [DONE] '%s' → doc_id=%d, doc_number='%s', type='%s', "
        "articles=%d, clauses=%d, chunks=%d",
        filename, document_id, doc_number, doc_type,
        len(structure.articles), len(clause_id_map), len(all_chunks),
    )
    return summary


# ── Helpers ───────────────────────────────────────────────

def _title_from_filename(filename: str) -> str:
    """Use filename stem as document title, cleaning up underscores/hyphens."""
    stem = Path(filename).stem.strip()
    if not stem:
        return ""
    cleaned = stem.replace("_", " ").replace("-", " – ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _extract_title(text: str, doc_type: str) -> str:
    """Best-effort title extraction from the first few lines."""
    lines = [line.strip() for line in text.split("\n")[:25] if line.strip()]

    def _is_non_title_header(line: str) -> bool:
        upper = line.upper()
        if _ISSUER_PATTERN.search(line):
            return True
        if upper.startswith("SỐ") or upper.startswith("SO"):
            return True
        if upper.startswith("CỘNG HÒA") or upper.startswith("ĐỘC LẬP"):
            return True
        return False

    filtered_lines = [line for line in lines if not _is_non_title_header(line)]

    # 1) If line is exactly document type, title is often right below it.
    for idx, line in enumerate(filtered_lines):
        if line.lower() == doc_type.lower() and idx + 1 < len(filtered_lines):
            nxt = filtered_lines[idx + 1]
            if len(nxt) > 10:
                return nxt[:300]

    # 1) Prefer full heading that already includes document type.
    for line in filtered_lines:
        if len(line) > 20 and doc_type.lower() in line.lower():
            return line

    # 2) Common legal heading line in uppercase.
    for line in filtered_lines:
        if len(line) >= 12 and line == line.upper() and len(line.split()) >= 3:
            return line[:300]

    # 3) Fallback: first substantial non-empty line.
    for line in filtered_lines:
        if len(line) > 10:
            return line[:300]

    return "Untitled"


def _extract_issued_date(text: str) -> Optional[date]:
    """Extract issued date from header line like 'ngày 01 tháng 01 năm 2024'."""
    m = _ISSUED_DATE_PATTERN.search(text[:5000])
    if not m:
        return None
    return _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))


def _extract_effective_date(text: str) -> Optional[date]:
    """Extract effective date from body phrase 'có hiệu lực ... ngày ...'."""
    m = _EFFECTIVE_DATE_PATTERN.search(text[:12000])
    if not m:
        return None
    return _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return datetime(year=year, month=month, day=day).date()
    except ValueError:
        return None
