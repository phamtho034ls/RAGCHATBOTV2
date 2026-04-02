"""
Document Summarizer – tóm tắt văn bản pháp luật.

Hai chế độ:
  1. list_document_articles: truy DB lấy danh sách điều luật (nhanh, chính xác)
  2. summarize_document: RAG retrieval → LLM tóm tắt (fallback)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from app.config import SUMMARIZE_PROMPT, COPILOT_SYSTEM_PROMPT
from app.database.models import Document, Article
from app.database.session import _session_factory
from app.services.llm_client import generate
from app.services.retrieval import search_all, format_sources, rewrite_query

log = logging.getLogger(__name__)


def _select_document_for_lookup():
    """SELECT không lấy ``law_intents`` — DB chưa alembic upgrade vẫn khớp schema cũ."""
    return select(Document).options(
        load_only(
            Document.id,
            Document.doc_number,
            Document.title,
            Document.document_type,
            Document.issuer,
            Document.issued_date,
            Document.effective_date,
            Document.file_path,
            Document.created_at,
        )
    )


_DOC_TYPE_MAP = {
    "luật": "Luật",
    "nghị định": "Nghị định",
    "nghị quyết": "Nghị quyết",
    "thông tư": "Thông tư",
    "quyết định": "Quyết định",
}

_DOC_REF_RE = re.compile(
    r"(\d+/\d{4}/[A-ZĐa-zđ\-]+)"
    r"|(\d+[/_]\d{4}[/_][A-Za-z][A-Za-z0-9\-]*)",
)

_DOC_TITLE_RE = re.compile(
    r"(?:luật|nghị\s*định|thông\s*tư|quyết\s*định|nghị\s*quyết)"
    r"\s+(?:số\s+)?(?:\d+[/_ ]\d{4}[/_ ][A-Za-zĐđ\-]*\s+)?(?:về\s+)?"
    r"(.{3,60}?)(?:\s+(?:bao gồm|gồm|có|quy định|nội dung|gì|nào|những))",
    re.IGNORECASE,
)


def _strip_diacritics_simple(text: str) -> str:
    import unicodedata
    text = text.replace("Đ", "D").replace("đ", "d")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


async def _find_document(query: str) -> Optional[Document]:
    """Find the best matching document from the user query.

    Strategies (in order):
      1. Explicit doc_number (e.g. 45/2024/QH15)
      2. Title keyword match (e.g. "luật báo chí")
    """
    async with _session_factory() as db:
        # Strategy 1: doc_number reference
        m = _DOC_REF_RE.search(query)
        if m:
            raw_ref = (m.group(1) or m.group(2)).replace("_", "/")
            normalized = _strip_diacritics_simple(raw_ref).upper()
            stmt = _select_document_for_lookup().where(
                func.upper(Document.doc_number).like(f"%{normalized}%")
            ).limit(1)
            row = (await db.execute(stmt)).scalar()
            if row:
                return row
            num_prefix = re.match(r"(\d+/\d{4})", raw_ref)
            if num_prefix:
                stmt = _select_document_for_lookup().where(
                    Document.doc_number.like(f"{num_prefix.group(1)}%")
                ).limit(1)
                row = (await db.execute(stmt)).scalar()
                if row:
                    return row

        # Strategy 2: title keyword match
        q_lower = query.lower()
        title_keywords = []
        for doc_type_kw in _DOC_TYPE_MAP:
            if doc_type_kw in q_lower:
                title_keywords.append(_DOC_TYPE_MAP[doc_type_kw])
                remaining = q_lower.split(doc_type_kw, 1)[1].strip()
                remaining = re.sub(
                    r"^(số\s+)?\d+[/_ ]\d{4}[/_ ]\S+\s*", "", remaining
                ).strip()
                remaining = re.sub(
                    r"\b(bao gồm|gồm|có|quy định|nội dung|những|gì|nào|"
                    r"về|trong|các|điều|luật|khoản|mục)\b",
                    "", remaining,
                ).strip()
                if remaining and len(remaining) >= 2:
                    title_keywords.extend(remaining.split())
                break

        if title_keywords:
            stmt = _select_document_for_lookup()
            for kw in title_keywords[:5]:
                if len(kw) >= 2:
                    stmt = stmt.where(Document.title.ilike(f"%{kw}%"))
            stmt = stmt.limit(1)
            row = (await db.execute(stmt)).scalar()
            if row:
                return row

            if len(title_keywords) > 1:
                stmt = _select_document_for_lookup().where(
                    Document.title.ilike(f"%{title_keywords[0]}%")
                )
                for kw in title_keywords[1:3]:
                    if len(kw) >= 2:
                        stmt = stmt.where(Document.title.ilike(f"%{kw}%"))
                stmt = stmt.limit(1)
                row = (await db.execute(stmt)).scalar()
                if row:
                    return row

    return None


async def list_document_articles(query: str) -> Dict[str, Any]:
    """List all article titles in a document matched from the query.

    Returns structured output with document info and article list.
    """
    doc = await _find_document(query)
    if not doc:
        return {
            "summary": "Không tìm thấy văn bản phù hợp trong cơ sở dữ liệu.",
            "sources": [],
            "confidence_score": 0.0,
        }

    async with _session_factory() as db:
        stmt = (
            select(Article.article_number, Article.title)
            .where(Article.document_id == doc.id)
            .order_by(Article.id)
        )
        rows = (await db.execute(stmt)).all()

    if not rows:
        return {
            "summary": f"Văn bản **{doc.title}** ({doc.doc_number}) không có điều luật nào trong DB.",
            "sources": [],
            "confidence_score": 0.5,
        }

    lines = [f"**{doc.title}**"]
    if doc.doc_number:
        lines[0] += f" (Số hiệu: {doc.doc_number})"
    if doc.document_type:
        lines.append(f"Loại: {doc.document_type}")
    if doc.issuer:
        lines.append(f"Trích yếu: {doc.issuer}")
    lines.append(f"Tổng số điều: {len(rows)}")
    lines.append("")
    lines.append("**Danh sách các điều:**\n")

    def _article_num_display(raw: str) -> str:
        s = (raw or "").strip()
        s = re.sub(r"(?i)^điều\s*", "", s).strip()
        return s

    for art_num, art_title in rows:
        num = _article_num_display(art_num or "")
        title = (art_title or "").strip()
        if re.match(r"(?i)^điều\s*\d", title):
            title = re.sub(r"(?i)^điều\s*\d+[a-z]?\s*[.:–-]\s*", "", title).strip()
        if num and title:
            lines.append(f"- **Điều {num}.** {title}")
        elif num:
            lines.append(f"- **Điều {num}**")
        elif title:
            lines.append(f"- {title}")

    sources = [{
        "doc_number": doc.doc_number or "",
        "document_title": doc.title or "",
        "document_type": doc.document_type or "",
    }]

    return {
        "summary": "\n".join(lines),
        "sources": sources,
        "confidence_score": 0.95,
    }


async def summarize_document(
    query: str,
    temperature: float = 0.3,
    top_k: int = 10,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Tóm tắt văn bản pháp luật dựa trên query (LLM-based fallback)."""
    rewritten = await rewrite_query(query)
    results = await search_all(rewritten, top_k=top_k)

    if not results:
        return {
            "summary": "Không tìm thấy văn bản liên quan trong cơ sở dữ liệu.",
            "sources": [],
        }

    document_text = "\n\n---\n\n".join(
        f"[Phần {i+1}]\n{doc['text']}" for i, doc in enumerate(results)
    )

    prompt = SUMMARIZE_PROMPT.format(document_text=document_text)
    summary = await generate(
        prompt,
        system=COPILOT_SYSTEM_PROMPT,
        temperature=temperature,
    )

    sources = format_sources(results)
    return {"summary": summary, "sources": sources}


async def summarize_matched_document(
    query: str,
    temperature: float = 0.25,
    max_chars: int = 22000,
) -> Dict[str, Any]:
    """Summarize the whole matched legal document (article coverage first)."""
    doc = await _find_document(query)
    if not doc:
        return {
            "summary": "Không tìm thấy văn bản phù hợp trong cơ sở dữ liệu.",
            "sources": [],
            "confidence_score": 0.0,
        }

    async with _session_factory() as db:
        stmt = (
            select(Article.article_number, Article.title, Article.content)
            .where(Article.document_id == doc.id)
            .order_by(Article.id)
        )
        rows = (await db.execute(stmt)).all()

    if not rows:
        return {
            "summary": f"Không có điều khoản để tóm tắt cho văn bản {doc.doc_number or doc.title}.",
            "sources": [{
                "doc_number": doc.doc_number or "",
                "document_title": doc.title or "",
                "document_type": doc.document_type or "",
            }],
            "confidence_score": 0.4,
        }

    parts: List[str] = []
    char_count = 0
    for art_no, art_title, art_content in rows:
        seg = f"[{art_no or ''} - {art_title or ''}]\n{(art_content or '').strip()}\n"
        if char_count + len(seg) > max_chars:
            break
        parts.append(seg)
        char_count += len(seg)

    document_text = "\n".join(parts).strip()
    if not document_text:
        return {
            "summary": "Không đủ nội dung văn bản để tóm tắt.",
            "sources": [{
                "doc_number": doc.doc_number or "",
                "document_title": doc.title or "",
                "document_type": doc.document_type or "",
            }],
            "confidence_score": 0.3,
        }

    prompt = (
        f"TÓM TẮT VĂN BẢN PHÁP LUẬT SAU THEO DẠNG CÓ CẤU TRÚC:\n"
        f"- Phạm vi điều chỉnh\n"
        f"- Đối tượng áp dụng\n"
        f"- Nhóm quy định chính\n"
        f"- Điểm cần lưu ý khi thực thi\n"
        f"- Điều/Khoản trọng yếu và nội dung quy định tương ứng\n\n"
        f"Văn bản: {doc.title or ''} ({doc.doc_number or ''})\n\n"
        f"Nội dung trích từ các điều:\n{document_text}\n\n"
        f"YÊU CẦU BẮT BUỘC:\n"
        f"- Không chỉ liệt kê số điều; phải nêu nội dung quy định tương ứng.\n"
        f"- Không bịa đặt nội dung ngoài phần trích."
    )
    summary = await generate(
        prompt=prompt,
        system=COPILOT_SYSTEM_PROMPT,
        temperature=min(max(temperature, 0.0), 0.35),
    )

    return {
        "summary": summary,
        "sources": [{
            "doc_number": doc.doc_number or "",
            "document_title": doc.title or "",
            "document_type": doc.document_type or "",
        }],
        "confidence_score": 0.9,
    }
