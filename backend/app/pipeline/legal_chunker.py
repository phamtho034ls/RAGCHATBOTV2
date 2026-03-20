"""Chunk generator for Vietnamese legal documents (Document -> Article -> Clause)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import List, Optional

from app.pipeline.legal_segmenter import split_for_chunking, trim_article_content, trim_clause_content
from app.pipeline.structure_detector import ArticleItem, ClauseItem, detect_points

log = logging.getLogger(__name__)


TOKEN_CHUNK_THRESHOLD = 512
TOKEN_CHUNK_SIZE = 512
TOKEN_CHUNK_OVERLAP = 80


@dataclass
class Chunk:
    text: str
    document_title: str = ""
    document_type: str = ""
    doc_number: str = ""
    year: Optional[int] = None
    chapter: str = ""
    section: str = ""  # Mục (e.g. "Mục 1 - Tên mục")
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    clause_number: Optional[str] = None
    chunk_type: str = "clause"


def create_chunk_text(
    *,
    document_title: str,
    article_number: str,
    clause_number: str,
    content: str,
) -> str:
    """Build chunk text in required legal format."""
    return "\n".join(
        [
            (document_title or "").strip(),
            f"Điều {article_number}".strip(),
            clause_number.strip(),
            content.strip(),
        ]
    ).strip()


def chunk_by_clause(
    articles: List[ArticleItem],
    *,
    document_title: str,
    doc_number: str = "",
    document_type: str = "",
) -> List[Chunk]:
    """Three-tier strategy: article chunk + clause chunks + 512-token sub-chunks.

    For each article:
      1. One "article" chunk containing the full article text
      2. One "clause" chunk per clause/point
      3. Overlapping "token_sub" chunks for any content > 512 words
    """
    chunks: List[Chunk] = []
    for article in articles:
        article_content = trim_article_content(article.content)

        # ── Tier 1: Article-level chunk ──
        article_text = _build_article_chunk_text(
            document_title=document_title,
            article=article,
            content=article_content,
        )
        chunks.append(Chunk(
            text=article_text,
            document_title=document_title,
            document_type=document_type,
            doc_number=doc_number,
            year=_extract_year(doc_number),
            chapter=article.chapter,
            section=getattr(article, "section", "") or "",
            article_number=article.number,
            article_title=article.title,
            clause_number=None,
            chunk_type="article",
        ))

        # ── Tier 2: Clause-level chunks ──
        if article.clauses:
            for clause in article.clauses:
                base_content = trim_clause_content(clause.content)
                parts = split_for_chunking(base_content, split_points=True) or [base_content]
                for idx, part in enumerate(parts, start=1):
                    clause_no = clause.number if idx == 1 else f"{clause.number}.{idx}"
                    chunks.append(
                        _make_clause_chunk(
                            article=article,
                            clause=ClauseItem(number=clause_no, content=part, points=[]),
                            document_title=document_title,
                            document_type=document_type,
                            doc_number=doc_number,
                        )
                    )
        else:
            points = detect_points(article_content)
            if points:
                for point in points:
                    point_clause = ClauseItem(
                        number=f"Điểm {point.letter}",
                        content=point.content,
                        points=[],
                    )
                    chunks.append(
                        _make_clause_chunk(
                            article=article,
                            clause=point_clause,
                            document_title=document_title,
                            document_type=document_type,
                            doc_number=doc_number,
                        )
                    )
            else:
                parts = split_for_chunking(article_content, split_points=True) or [article_content]
                for idx, part in enumerate(parts, start=1):
                    clause_no = "1" if idx == 1 else f"1.{idx}"
                    fallback_clause = ClauseItem(number=clause_no, content=part, points=[])
                    chunks.append(
                        _make_clause_chunk(
                            article=article,
                            clause=fallback_clause,
                            document_title=document_title,
                            document_type=document_type,
                            doc_number=doc_number,
                        )
                    )

    # ── Tier 3: 512-token sub-chunks for large content ──
    chunks = _add_token_sub_chunks(chunks)
    log.info("Generated %d legal chunks from %d articles.", len(chunks), len(articles))
    return chunks


def _build_article_chunk_text(
    *,
    document_title: str,
    article: "ArticleItem",
    content: str,
) -> str:
    """Build a single chunk containing the full article text with header."""
    parts = []
    if document_title:
        parts.append(document_title.strip())
    label = f"Điều {article.number}"
    if article.title:
        label += f". {article.title}"
    parts.append(label)
    parts.append(content.strip())
    return "\n".join(parts).strip()


def _add_token_sub_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Add overlapping token-based sub-chunks for large clause chunks.

    Keeps the original clause-level chunks for full-context retrieval,
    and adds smaller sub-chunks that fit within the embedding model's
    token window (typically 512 tokens) for precise semantic matching.
    """
    result: List[Chunk] = []
    for chunk in chunks:
        result.append(chunk)
        words = chunk.text.split()
        if len(words) <= TOKEN_CHUNK_THRESHOLD:
            continue

        prefix_parts: List[str] = []
        if chunk.document_title:
            prefix_parts.append(chunk.document_title)
        if chunk.article_number:
            label = f"Điều {chunk.article_number}"
            if chunk.article_title:
                label += f". {chunk.article_title}"
            prefix_parts.append(label)
        prefix = "\n".join(prefix_parts)

        start = 0
        sub_idx = 1
        while start < len(words):
            end = min(start + TOKEN_CHUNK_SIZE, len(words))
            sub_text = " ".join(words[start:end])
            full_text = f"{prefix}\n{sub_text}" if prefix else sub_text

            parent_clause = chunk.clause_number or "1"
            result.append(Chunk(
                text=full_text,
                document_title=chunk.document_title,
                document_type=chunk.document_type,
                doc_number=chunk.doc_number,
                year=chunk.year,
                chapter=chunk.chapter,
                section=getattr(chunk, "section", "") or "",
                article_number=chunk.article_number,
                article_title=chunk.article_title,
                clause_number=f"{parent_clause}.sub.{sub_idx}",
                chunk_type="token_sub",
            ))

            if end >= len(words):
                break
            start += TOKEN_CHUNK_SIZE - TOKEN_CHUNK_OVERLAP
            sub_idx += 1

    added = len(result) - len(chunks)
    if added > 0:
        log.info("Added %d token sub-chunks from %d original chunks.", added, len(chunks))
    return result


def chunk_by_clause_primary(
    articles: List[ArticleItem],
    *,
    law_title: str,
    doc_number: str = "",
    document_type: str = "",
) -> List[Chunk]:
    """Backward-compatible alias for existing pipeline."""
    return chunk_by_clause(
        articles,
        document_title=law_title,
        doc_number=doc_number,
        document_type=document_type,
    )


def format_clause_chunk_text(
    *,
    law_title: str,
    article_number: str,
    article_title: str,
    clause_number: str,
    clause_content: str,
) -> str:
    """Backward-compatible alias kept for legacy calls."""
    _ = article_title
    return create_chunk_text(
        document_title=law_title,
        article_number=article_number,
        clause_number=f"Khoản {clause_number}",
        content=clause_content,
    )


def _make_clause_chunk(
    *,
    article: ArticleItem,
    clause: ClauseItem,
    document_title: str,
    document_type: str,
    doc_number: str,
) -> Chunk:
    clause_label = clause.number if clause.number.lower().startswith("điểm ") else f"Khoản {clause.number}"
    text = create_chunk_text(
        document_title=document_title,
        article_number=article.number,
        clause_number=clause_label,
        content=clause.content,
    )
    return Chunk(
        text=text,
        document_title=document_title,
        document_type=document_type,
        doc_number=doc_number,
        year=_extract_year(doc_number),
        chapter=article.chapter,
        section=getattr(article, "section", "") or "",
        article_number=article.number,
        article_title=article.title,
        clause_number=clause.number,
    )


def _extract_year(doc_number: str) -> Optional[int]:
    if not doc_number:
        return None
    match = re.search(r"/(\d{4})/", doc_number)
    if not match:
        return None
    return int(match.group(1))


def attach_context_prefix(
    *,
    doc_number: str,
    document_type: str,
    chapter: str,
    article_number: Optional[str],
    article_title: Optional[str],
    clause_number: Optional[str],
) -> str:
    """Compatibility helper for previous chunker interface."""
    parts: List[str] = []
    if document_type and doc_number:
        parts.append(f"{document_type} {doc_number}")
    elif doc_number:
        parts.append(doc_number)
    if chapter:
        parts.append(chapter)
    if article_number:
        parts.append(f"Điều {article_number}" if not article_title else f"Điều {article_number} - {article_title}")
    if clause_number:
        parts.append(f"Khoản {clause_number}")
    return "\n".join(parts).strip()


def chunk_by_clause_if_needed(
    article: ArticleItem,
    *,
    document_title: str,
    doc_number: str,
    document_type: str,
) -> List[Chunk]:
    """Compatibility wrapper: chunk one article by clause."""
    return chunk_by_clause(
        [article],
        document_title=document_title,
        doc_number=doc_number,
        document_type=document_type,
    )


def chunk_by_article(
    articles: List[ArticleItem],
    *,
    document_title: str = "",
    doc_number: str = "",
    document_type: str = "",
) -> List[Chunk]:
    """Compatibility wrapper: now delegates to clause-first chunking."""
    return chunk_by_clause(
        articles,
        document_title=document_title,
        doc_number=doc_number,
        document_type=document_type,
    )


def chunk_articles(
    articles: List[ArticleItem],
    document_title: str = "",
    doc_number: str = "",
    document_type: str = "",
) -> List[Chunk]:
    """Compatibility wrapper used by legacy imports."""
    return chunk_by_article(
        articles,
        document_title=document_title,
        doc_number=doc_number,
        document_type=document_type,
    )


def chunk_preamble(
    preamble: str,
    *,
    document_title: str = "",
    doc_number: str = "",
    document_type: str = "",
) -> List[Chunk]:
    """Compatibility wrapper: preamble is intentionally not chunked."""
    _ = (preamble, document_title, doc_number, document_type)
    return []
