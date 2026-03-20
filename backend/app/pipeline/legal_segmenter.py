"""Utilities for splitting legal text by structural boundary markers."""

from __future__ import annotations

import re
from typing import List

_CHAPTER_LINE = re.compile(r"^\s*Chương\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
_ARTICLE_LINE = re.compile(r"^\s*(?:Điều|Dieu)\s+\d+[A-Za-z]?\b", re.IGNORECASE)
_CLAUSE_LINE = re.compile(r"^\s*(?:Khoản\s+)?\d+[A-Za-z]?\s*[\.\)]\s*.+$")
_POINT_LINE = re.compile(r"^\s*[a-zđ]\s*[\)\.]\s*.+$", re.IGNORECASE)
_NUMBERED_ITEM_LINE = re.compile(r"^\s*\d{1,3}\s*[\.\)]\s*.+$")


def is_hard_boundary(line: str) -> bool:
    """Hard boundaries split article/clauses aggressively."""
    text = (line or "").strip()
    if not text:
        return False
    return bool(_CHAPTER_LINE.match(text) or _ARTICLE_LINE.match(text))


def is_soft_boundary(line: str, *, split_points: bool = True) -> bool:
    """Soft boundaries split chunks inside article/clauses."""
    text = (line or "").strip()
    if not text:
        return False
    if is_hard_boundary(text):
        return True
    if _CLAUSE_LINE.match(text):
        return True
    if split_points and (_POINT_LINE.match(text) or _NUMBERED_ITEM_LINE.match(text)):
        return True
    return False


def trim_article_content(text: str) -> str:
    """Trim article content before accidental next Chapter/Article block."""
    lines = (text or "").splitlines()
    if not lines:
        return ""

    kept: List[str] = []
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if idx > 0 and is_hard_boundary(line):
            break
        kept.append(raw)
    return "\n".join(kept).strip()


def trim_clause_content(text: str) -> str:
    """Trim clause before accidental next Chapter/Article block."""
    lines = (text or "").splitlines()
    if not lines:
        return ""

    kept: List[str] = []
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if idx > 0 and is_hard_boundary(line):
            break
        kept.append(raw)
    return "\n".join(kept).strip()


def split_for_chunking(text: str, *, split_points: bool = True) -> List[str]:
    """Split text into smaller segments whenever a new legal marker appears."""
    lines = (text or "").splitlines()
    if not lines:
        return []

    segments: List[List[str]] = []
    current: List[str] = []

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if idx > 0 and is_soft_boundary(line, split_points=split_points) and current:
            segments.append(current)
            current = [raw]
            continue
        current.append(raw)

    if current:
        segments.append(current)

    return ["\n".join(seg).strip() for seg in segments if "\n".join(seg).strip()]

