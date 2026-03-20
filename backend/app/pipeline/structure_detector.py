"""Vietnamese legal structure detector using robust line-based parsing."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)

_CHAPTER_PATTERN = re.compile(r"^\s*Chương\s+([IVXLCDM]+|\d+)\b[\.\-–:]*\s*(.*)$", re.IGNORECASE)
_MUC_PATTERN = re.compile(r"^\s*Mục\s+(\d+[A-Za-z]?)\s*[\.\-–:]?\s*(.*)$", re.IGNORECASE)
_ARTICLE_PATTERN = re.compile(r"^\s*(?:Điều|Dieu)\s*(\d+[A-Za-z]?)\s*[\.\-–:]?\s*(.*)$", re.IGNORECASE)
_CLAUSE_PATTERN = re.compile(r"^\s*(?:Khoản\s+)?(\d+[A-Za-z]?)\s*[\.\)]\s*(.*)$", re.IGNORECASE)
_POINT_PATTERN = re.compile(r"^\s*([a-zđ])\s*[\)\.]\s*(.*)$", re.IGNORECASE)
_SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(Chương\s*trình|Nội\s*dung|Mục\s*tiêu|Phần)\b\s*[:\-]?\s*(.*)$",
    re.IGNORECASE,
)
_NUMBERED_HEADING_PATTERN = re.compile(r"^\s*(\d{1,3})\s*[\.\)]\s*(.+)$")

_APPENDIX_START_PATTERN = re.compile(
    r"^\s*(Phụ\s*lục|Mẫu\s*(?:số|biểu)?|Biểu\s*mẫu|Bảng)\b",
    re.IGNORECASE,
)
_TABLE_LIKE_LINE_PATTERN = re.compile(r"^\s*[\wÀ-ỹ0-9\(\)\-/\.:]+(?:\s{2,}|\t+).+")

_DOCTYPE_PATTERNS = {
    "Luật": re.compile(r"\bLuật\b", re.IGNORECASE),
    "Nghị định": re.compile(r"\bNghị\s+định\b", re.IGNORECASE),
    "Thông tư": re.compile(r"\bThông\s+tư\b", re.IGNORECASE),
    "Nghị quyết": re.compile(r"\bNghị\s+quyết\b", re.IGNORECASE),
    "Quyết định": re.compile(r"\bQuyết\s+định\b", re.IGNORECASE),
}


@dataclass
class PointItem:
    letter: str
    content: str


@dataclass
class ClauseItem:
    number: str
    content: str
    points: List[PointItem] = field(default_factory=list)


@dataclass
class ArticleItem:
    number: str
    title: str
    full_label: str
    content: str
    clauses: List[ClauseItem] = field(default_factory=list)
    chapter: str = ""
    section: str = ""  # Mục X - title (e.g. "Mục 1 - Tên mục")


@dataclass
class ChapterItem:
    chapter_number: str
    title: str
    articles: List[ArticleItem] = field(default_factory=list)


@dataclass
class DocumentStructure:
    document_type: str
    preamble: str
    chapters: List[ChapterItem]
    articles: List[ArticleItem]
    excluded_sections: List[str] = field(default_factory=list)


def detect_chapters(text: str) -> List[ChapterItem]:
    """Detect chapter headings from line starts."""
    chapters: List[ChapterItem] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _CHAPTER_PATTERN.match(line)
        if not m:
            continue
        chapters.append(ChapterItem(chapter_number=m.group(1).upper(), title=(m.group(2) or "").strip()))
    return chapters


def detect_articles(text: str) -> List[ArticleItem]:
    """Detect articles with strict boundaries to avoid article merge."""
    lines = (text or "").splitlines()
    if not lines:
        return []

    article_blocks: List[tuple[str, str, List[str], str, str]] = []
    current_chapter = ""
    current_section = ""
    current: Optional[dict] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current is not None:
                current["lines"].append("")
            continue

        chapter_match = _CHAPTER_PATTERN.match(line)
        if chapter_match:
            chapter_no = chapter_match.group(1).upper()
            chapter_title = (chapter_match.group(2) or "").strip()
            current_chapter = f"Chương {chapter_no}" if not chapter_title else f"Chương {chapter_no} - {chapter_title}"
            current_section = ""
            if current is not None:
                current["lines"].append(line)
            continue

        muc_match = _MUC_PATTERN.match(line)
        if muc_match:
            muc_no = muc_match.group(1).strip()
            muc_title = (muc_match.group(2) or "").strip()
            current_section = f"Mục {muc_no}" if not muc_title else f"Mục {muc_no} - {muc_title}"
            if current is not None:
                current["lines"].append(line)
            continue

        article_match = _ARTICLE_PATTERN.match(line)
        if article_match:
            if current is not None:
                article_blocks.append(
                    (
                        current["number"],
                        current["title"],
                        current["lines"],
                        current["chapter"],
                        current["section"],
                    )
                )
            current = {
                "number": article_match.group(1).strip(),
                "title": (article_match.group(2) or "").strip(),
                "lines": [],
                "chapter": current_chapter,
                "section": current_section,
            }
            continue

        if current is not None:
            current["lines"].append(line)

    if current is not None:
        article_blocks.append((current["number"], current["title"], current["lines"], current["chapter"], current["section"]))

    articles: List[ArticleItem] = []
    for number, title, body_lines, chapter, section in article_blocks:
        resolved_title, body_lines = _resolve_article_title(title, body_lines)
        body = "\n".join(body_lines).strip()
        clauses = detect_clauses(body)
        articles.append(
            ArticleItem(
                number=number,
                title=resolved_title,
                full_label=f"Điều {number}",
                content=body,
                clauses=clauses,
                chapter=chapter,
                section=section,
            )
        )
    if articles:
        return articles
    return _detect_outline_articles(lines)


def detect_clauses(article_text: str) -> List[ClauseItem]:
    """Detect clause boundaries by line starts to avoid cutting mid-clause."""
    lines = (article_text or "").splitlines()
    clauses: List[ClauseItem] = []
    current_number: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        if current_number is None:
            return
        full_clause = "\n".join(current_lines).strip()
        if not full_clause:
            return
        clauses.append(ClauseItem(number=current_number, content=full_clause, points=detect_points(full_clause)))

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_number is not None:
                current_lines.append("")
            continue

        match = _CLAUSE_PATTERN.match(line)
        if match:
            flush()
            current_number = match.group(1).strip()
            current_lines = [line]
            continue

        if current_number is not None:
            current_lines.append(line)

    flush()
    return clauses


_INLINE_POINT_RE = re.compile(r"(?<=[;.:!])\s+([a-zđ])\s*\)\s+", re.IGNORECASE)


def _normalize_inline_points(text: str) -> str:
    """Split inline points like 'text; a) content b) content' onto separate lines."""
    return _INLINE_POINT_RE.sub(r"\n\1) ", text)


def detect_points(clause_text: str) -> List[PointItem]:
    """Detect points a), b), c) inside one clause (handles inline points too)."""
    normalized = _normalize_inline_points(clause_text or "")
    lines = normalized.splitlines()
    points: List[PointItem] = []
    current_letter: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        if current_letter is None:
            return
        content = "\n".join(current_lines).strip()
        if content:
            points.append(PointItem(letter=current_letter, content=content))

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_letter is not None:
                current_lines.append("")
            continue
        match = _POINT_PATTERN.match(line)
        if match:
            flush()
            current_letter = match.group(1).lower()
            current_lines = [line]
            continue
        if current_letter is not None:
            current_lines.append(line)

    flush()
    return points


def detect_document_type(text: str) -> str:
    """Detect legal document type from leading content."""
    first_1500 = text[:1500]
    for label, pattern in _DOCTYPE_PATTERNS.items():
        if pattern.search(first_1500):
            return label
    return "Văn bản"


def detect_structure(text: str) -> DocumentStructure:
    """Build a hierarchical structure and separate non-embedding sections."""
    main_text, excluded_sections = _split_excluded_sections(text)
    doc_type = detect_document_type(main_text)
    chapters = detect_chapters(main_text)
    articles = detect_articles(main_text)

    preamble_lines: List[str] = []
    for line in main_text.splitlines():
        if _ARTICLE_PATTERN.match(line.strip()):
            break
        preamble_lines.append(line)
    preamble = "\n".join(preamble_lines).strip()

    chapter_map = {chapter.chapter_number: chapter for chapter in chapters}
    for article in articles:
        chapter_num = _extract_chapter_number(article.chapter)
        if chapter_num:
            chapter_map.setdefault(chapter_num, ChapterItem(chapter_number=chapter_num, title=""))
            chapter_map[chapter_num].articles.append(article)
    if not chapter_map:
        chapters = [ChapterItem(chapter_number="", title="", articles=articles)]
    else:
        chapters = list(chapter_map.values())

    log.info(
        "Detected structure: type=%s chapters=%d articles=%d clauses=%d excluded=%d",
        doc_type,
        len(chapters),
        len(articles),
        sum(len(a.clauses) for a in articles),
        len(excluded_sections),
    )
    return DocumentStructure(
        document_type=doc_type,
        preamble=preamble,
        chapters=chapters,
        articles=articles,
        excluded_sections=excluded_sections,
    )


def _detect_outline_articles(lines: List[str]) -> List[ArticleItem]:
    """Fallback parser for documents without explicit 'Điều' markers.

    Strategy:
    - Split by chapter/section headings when available.
    - If absent, split by top-level numbered headings (1., 2., ...).
    - Build synthetic Article objects so ingestion still captures full text.
    """
    if not lines:
        return []

    chapter_label = ""
    section_label = ""
    synthetic_blocks: list[tuple[str, str, list[str], str, str]] = []
    current: Optional[dict] = None
    auto_number = 1

    def flush_current() -> None:
        if current is None:
            return
        body_lines = current["lines"]
        title = current["title"].strip()
        if not title and body_lines:
            probe = body_lines[0].strip()
            if probe and len(probe) <= 220:
                title = probe
        body = [ln for ln in body_lines]
        synthetic_blocks.append((str(current["number"]), title, body, current["chapter"], current.get("section", "")))

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current is not None:
                current["lines"].append("")
            continue

        chapter_match = _CHAPTER_PATTERN.match(line)
        if chapter_match:
            flush_current()
            chapter_no = chapter_match.group(1).upper()
            chapter_title = (chapter_match.group(2) or "").strip()
            chapter_label = f"Chương {chapter_no}" if not chapter_title else f"Chương {chapter_no} - {chapter_title}"
            section_label = ""
            current = {
                "number": auto_number,
                "title": chapter_label,
                "lines": [],
                "chapter": chapter_label,
                "section": "",
            }
            auto_number += 1
            continue

        muc_match = _MUC_PATTERN.match(line)
        if muc_match:
            flush_current()
            muc_no = muc_match.group(1).strip()
            muc_title = (muc_match.group(2) or "").strip()
            section_label = f"Mục {muc_no}" if not muc_title else f"Mục {muc_no} - {muc_title}"
            current = {
                "number": auto_number,
                "title": section_label,
                "lines": [],
                "chapter": chapter_label,
                "section": section_label,
            }
            auto_number += 1
            continue

        section_match = _SECTION_HEADING_PATTERN.match(line)
        if section_match:
            flush_current()
            section_name = section_match.group(1).strip()
            section_tail = (section_match.group(2) or "").strip()
            section_title = section_name if not section_tail else f"{section_name}: {section_tail}"
            section_label = section_title
            current = {
                "number": auto_number,
                "title": section_title,
                "lines": [],
                "chapter": chapter_label,
                "section": section_label,
            }
            auto_number += 1
            continue

        numbered = _NUMBERED_HEADING_PATTERN.match(line)
        if numbered and (current is None or len(current["lines"]) > 2):
            flush_current()
            current = {
                "number": auto_number,
                "title": line[:220],
                "lines": [line],
                "chapter": chapter_label,
                "section": section_label,
            }
            auto_number += 1
            continue

        if current is None:
            current = {
                "number": auto_number,
                "title": "",
                "lines": [],
                "chapter": chapter_label,
                "section": section_label,
            }
            auto_number += 1
        current["lines"].append(line)

    flush_current()

    articles: List[ArticleItem] = []
    for _number, title, body_lines, chapter, section in synthetic_blocks:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        clauses = detect_clauses(body)
        number = str(len(articles) + 1)
        articles.append(
            ArticleItem(
                number=number,
                title=title,
                full_label=f"Mục {number}",
                content=body,
                clauses=clauses,
                chapter=chapter,
                section=section,
            )
        )

    if articles:
        log.warning(
            "Fallback outline parser activated: synthesized %d pseudo-articles.",
            len(articles),
        )
    return articles


def build_legal_tree(structure: DocumentStructure, document_name: str = "") -> dict:
    """Convert dataclass structure into JSON-serializable legal tree."""
    return {
        "document": document_name,
        "document_type": structure.document_type,
        "chapters": [
            {
                "chapter_number": chapter.chapter_number,
                "title": chapter.title,
                "articles": [
                    {
                        "article_number": article.number,
                        "title": article.title,
                        "clauses": [
                            {
                                "clause_number": clause.number,
                                "points": [
                                    {"point_letter": point.letter, "content": point.content}
                                    for point in clause.points
                                ],
                            }
                            for clause in article.clauses
                        ],
                    }
                    for article in chapter.articles
                ],
            }
            for chapter in structure.chapters
        ],
    }


def _extract_chapter_number(chapter_text: str) -> Optional[str]:
    if not chapter_text:
        return None
    match = re.search(r"Chương\s+([IVXLCDM]+|\d+)", chapter_text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def _split_excluded_sections(text: str) -> tuple[str, List[str]]:
    """Split document into embeddable legal body and excluded appendix-like blocks."""
    lines = (text or "").splitlines()
    if not lines:
        return "", []

    main_lines: list[str] = []
    excluded: list[str] = []
    in_excluded = False
    current_block: list[str] = []

    for line in lines:
        striped = line.strip()
        if _APPENDIX_START_PATTERN.match(striped):
            if current_block:
                excluded.append("\n".join(current_block).strip())
                current_block = []
            in_excluded = True

        table_like = bool(_TABLE_LIKE_LINE_PATTERN.match(line) and len(line.split()) > 6)
        if in_excluded or table_like:
            current_block.append(line)
            continue

        main_lines.append(line)

    if current_block:
        excluded.append("\n".join(current_block).strip())

    return "\n".join(main_lines).strip(), [item for item in excluded if item]


def _resolve_article_title(initial_title: str, body_lines: List[str]) -> tuple[str, List[str]]:
    """Resolve missing titles when title is on the next line."""
    if initial_title.strip():
        return initial_title.strip(), body_lines
    if not body_lines:
        return "", body_lines

    first = body_lines[0].strip()
    if not first:
        return "", body_lines
    if _CLAUSE_PATTERN.match(first) or _POINT_PATTERN.match(first):
        return "", body_lines

    # Heading-like line immediately after "Điều X." should be title.
    if len(first) <= 220:
        return first, body_lines[1:]
    return "", body_lines
