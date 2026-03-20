"""Database write helpers for legal ingestion pipeline."""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Article, Chapter, Clause, Document, Section, VectorChunk
from app.pipeline.legal_chunker import Chunk
from app.pipeline.legal_segmenter import split_for_chunking, trim_article_content, trim_clause_content
from app.pipeline.structure_detector import ArticleItem, ChapterItem, detect_points


def _extract_chapter_number(chapter_text: str) -> Optional[str]:
    if not chapter_text:
        return None
    m = re.search(r"Chương\s+([IVXLCDM]+|\d+)", chapter_text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _extract_section_number(section_text: str) -> Optional[str]:
    if not section_text:
        return None
    m = re.search(r"Mục\s+(\d+[A-Za-z]?)", section_text, re.IGNORECASE)
    return m.group(1).strip() if m else None


async def insert_document(
    db: AsyncSession,
    *,
    doc_number: str,
    title: str,
    document_type: str,
    issuer: str,
    issued_date: Optional[date],
    effective_date: Optional[date],
    file_path: str,
) -> Document:
    document = Document(
        doc_number=doc_number,
        title=title,
        document_type=document_type,
        issuer=issuer,
        issued_date=issued_date,
        effective_date=effective_date,
        file_path=file_path,
    )
    db.add(document)
    await db.flush()
    return document


async def insert_chapters(
    db: AsyncSession,
    *,
    document_id: int,
    chapters: List[ChapterItem],
) -> Dict[str, int]:
    """Insert chapter rows. Returns chapter_number -> chapter.id."""
    chapter_id_map: Dict[str, int] = {}
    for order, ch in enumerate(chapters):
        row = Chapter(
            document_id=document_id,
            chapter_number=ch.chapter_number,
            title=ch.title or "",
            sort_order=order,
        )
        db.add(row)
        await db.flush()
        chapter_id_map[ch.chapter_number] = row.id
    return chapter_id_map


async def insert_sections_from_articles(
    db: AsyncSession,
    *,
    chapter_id_map: Dict[str, int],
    articles: List[ArticleItem],
) -> Dict[Tuple[int, str], int]:
    """Create section (Mục) rows from article.section. Returns (chapter_id, section_number) -> section.id."""
    section_id_map: Dict[Tuple[int, str], int] = {}
    seen: set = set()
    for item in articles:
        section_text = getattr(item, "section", "") or ""
        section_num = _extract_section_number(section_text)
        if not section_num:
            continue
        ch_num = _extract_chapter_number(getattr(item, "chapter", "") or "")
        if not ch_num or ch_num not in chapter_id_map:
            continue
        ch_id = chapter_id_map[ch_num]
        key = (ch_id, section_num)
        if key in seen:
            continue
        seen.add(key)
        title = section_text.replace(f"Mục {section_num}", "").strip(" -")
        row = Section(
            chapter_id=ch_id,
            section_number=section_num,
            title=title or "",
            sort_order=len(seen),
        )
        db.add(row)
        await db.flush()
        section_id_map[key] = row.id
    return section_id_map


async def insert_articles(
    db: AsyncSession,
    *,
    document_id: int,
    articles: List[ArticleItem],
    chapter_id_map: Optional[Dict[str, int]] = None,
    section_id_map: Optional[Dict[Tuple[int, str], int]] = None,
) -> Dict[str, int]:
    article_id_map: Dict[str, int] = {}
    chapter_id_map = chapter_id_map or {}
    section_id_map = section_id_map or {}

    for item in articles:
        article_content = trim_article_content(item.content)
        ch_num = _extract_chapter_number(getattr(item, "chapter", "") or "")
        sec_num = _extract_section_number(getattr(item, "section", "") or "")
        chapter_id = chapter_id_map.get(ch_num) if ch_num else None
        section_id = None
        if chapter_id and sec_num:
            section_id = section_id_map.get((chapter_id, sec_num))

        article = Article(
            document_id=document_id,
            chapter_id=chapter_id,
            section_id=section_id,
            article_number=f"Điều {item.number}",
            title=item.title,
            content=article_content,
        )
        db.add(article)
        await db.flush()
        article_id_map[item.number] = article.id
    return article_id_map


async def insert_clauses(
    db: AsyncSession,
    *,
    articles: List[ArticleItem],
    article_id_map: Dict[str, int],
) -> Dict[str, int]:
    """Insert clause rows (Khoản + Điểm) and return id mapping for chunk linkage."""
    clause_id_map: Dict[str, int] = {}
    for article in articles:
        article_id = article_id_map[article.number]
        if article.clauses:
            for clause in article.clauses:
                base_content = trim_clause_content(clause.content)
                pieces = split_for_chunking(base_content, split_points=True) or [base_content]
                for idx, piece in enumerate(pieces, start=1):
                    normalized_no = f"{clause.number}" if idx == 1 else f"{clause.number}.{idx}"
                    clause_row = Clause(
                        article_id=article_id,
                        clause_number=f"Khoản {normalized_no}",
                        content=piece,
                    )
                    db.add(clause_row)
                    await db.flush()
                    clause_id_map[f"{article.number}:Khoản {normalized_no}"] = clause_row.id

                # Persist points as separate clause rows as requested.
                for point in detect_points(base_content):
                    point_row = Clause(
                        article_id=article_id,
                        clause_number=f"Điểm {point.letter}",
                        content=point.content,
                    )
                    db.add(point_row)
                    await db.flush()
                    clause_id_map[f"{article.number}:Điểm {point.letter}"] = point_row.id
        else:
            base_content = trim_article_content(article.content)
            pieces = split_for_chunking(base_content, split_points=True) or [base_content]
            first_piece = pieces[0]
            fallback = Clause(
                article_id=article_id,
                clause_number="Khoản 1",
                content=first_piece,
            )
            db.add(fallback)
            await db.flush()
            clause_id_map[f"{article.number}:Khoản 1"] = fallback.id

            for idx, piece in enumerate(pieces[1:], start=2):
                seg_row = Clause(
                    article_id=article_id,
                    clause_number=f"Khoản 1.{idx}",
                    content=piece,
                )
                db.add(seg_row)
                await db.flush()
                clause_id_map[f"{article.number}:Khoản 1.{idx}"] = seg_row.id

            # Also persist lettered points (a, b, c, ...) when no numeric clause exists.
            for point in detect_points(base_content):
                point_row = Clause(
                    article_id=article_id,
                    clause_number=f"Điểm {point.letter}",
                    content=point.content,
                )
                db.add(point_row)
                await db.flush()
                clause_id_map[f"{article.number}:Điểm {point.letter}"] = point_row.id
    return clause_id_map


async def insert_chunks(
    db: AsyncSession,
    *,
    document_id: int,
    chunks: List[Chunk],
    vector_ids: List[str],
    article_id_map: Dict[str, int],
    clause_id_map: Dict[str, int],
) -> None:
    for chunk, vector_id in zip(chunks, vector_ids):
        article_id = article_id_map.get(chunk.article_number or "")
        clause_id = None
        if chunk.article_number and chunk.clause_number:
            # For token sub-chunks, look up parent clause
            clause_number = chunk.clause_number.strip()
            if ".sub." in clause_number:
                clause_number = clause_number.split(".sub.")[0]
            if clause_number.lower().startswith("điểm "):
                clause_id = clause_id_map.get(f"{chunk.article_number}:{clause_number}")
            else:
                clause_id = clause_id_map.get(f"{chunk.article_number}:Khoản {clause_number}")

        row = VectorChunk(
            document_id=document_id,
            article_id=article_id,
            clause_id=clause_id,
            vector_id=vector_id,
            chunk_text=chunk.text,
            chunk_type=getattr(chunk, "chunk_type", "clause"),
        )
        db.add(row)
    await db.flush()

