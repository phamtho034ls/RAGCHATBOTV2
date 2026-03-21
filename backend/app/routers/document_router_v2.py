"""Document API router – upload, list, and manage documents.

Endpoints:
  POST /api/upload
  GET  /api/documents
  GET  /api/documents/{id}
  GET  /api/datasets          (frontend-compatible alias)
  DELETE /api/datasets/{id}   (delete document + vectors)
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import STORAGE_DIR
from app.database.models import Article, Clause, Document, VectorChunk
from app.database.session import get_db
from app.models.schemas_v2 import (
    DocumentDetailResponse,
    DocumentInfo,
    DocumentListResponse,
    UploadResponse,
)
from app.pipeline.ingestor import ingest_document
from app.pipeline.vector_store import delete_by_document_id
from app.services.intent_detector import auto_index_from_document
from app.cache.redis_cache import invalidate_all_query_cache

log = logging.getLogger(__name__)


async def _invalidate_cache_after_upload(document_id: int) -> None:
    try:
        n = await invalidate_all_query_cache()
        log.info("Cache invalidated after document upload: doc_id=%s (%d keys)", document_id, n)
    except Exception as exc:
        log.error("Cache invalidation failed after upload doc_id=%s: %s", document_id, exc)
router = APIRouter(prefix="/api", tags=["documents"])

ALLOWED_EXTENSIONS = (".doc", ".docx")


_MAX_FILENAME_LEN = 150  # Windows MAX_PATH = 260; reserve ~110 for dir prefix


# Swagger UI often pre-fills optional form fields with the literal "string".
_DATE_FORM_IGNORE = frozenset(
    {"", "string", "null", "none", "undefined", "n/a", "na", "-"}
)


def _parse_optional_date_form(value: Optional[str], field: str) -> Optional[date]:
    """Parse YYYY-MM-DD from multipart form; empty / Swagger placeholders → None."""
    if value is None:
        return None
    s = value.strip()
    if not s or s.lower() in _DATE_FORM_IGNORE:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError as e:
        msg = f"{field} không hợp lệ (dùng YYYY-MM-DD): {s!r}"
        log.warning("POST /api/upload → 400 | %s", msg)
        raise HTTPException(400, msg) from e


def _safe_filename(raw_filename: str) -> str:
    """Strip folder prefixes and truncate long filenames.

    When using webkitdirectory, browsers send relative paths like
    'Tài Liệu/Luật Di sản văn hóa 2024.doc'. We only need the basename.
    Filenames exceeding _MAX_FILENAME_LEN are truncated to avoid MAX_PATH on Windows.
    """
    if not raw_filename:
        return "unknown"
    name = PurePosixPath(raw_filename).name or PureWindowsPath(raw_filename).name
    name = name or raw_filename

    if len(name) > _MAX_FILENAME_LEN:
        stem = Path(name).stem
        ext = Path(name).suffix
        name = stem[: _MAX_FILENAME_LEN - len(ext)] + ext

    return name


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    issuer: str = Form(default=""),
    issued_date: Optional[str] = Form(default=None),
    effective_date: Optional[str] = Form(default=None),
    title: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Upload and ingest a legal DOCX document."""
    issued_parsed = _parse_optional_date_form(issued_date, "issued_date")
    effective_parsed = _parse_optional_date_form(effective_date, "effective_date")
    safe_name = _safe_filename(file.filename or "")
    if not safe_name.lower().endswith(ALLOWED_EXTENSIONS):
        log.warning(
            "POST /api/upload → 400 | Chỉ hỗ trợ .doc/.docx | filename=%r",
            file.filename,
        )
        raise HTTPException(400, "Chỉ hỗ trợ file .doc và .docx")

    upload_id = uuid.uuid4().hex[:12]
    upload_dir = STORAGE_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / safe_name

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        result = await ingest_document(
            file_path=file_path,
            db=db,
            issuer=issuer,
            issued_date=issued_parsed,
            effective_date=effective_parsed,
            title_override=title,
        )
        doc_id = result.get("document_id")
        if doc_id:
            try:
                added = await auto_index_from_document(int(doc_id))
                if added:
                    log.info("Auto-indexed %d prototypes from doc_id=%s", added, doc_id)
            except Exception as e_idx:
                log.warning("Auto-index failed for doc_id=%s: %s", doc_id, e_idx)
            background_tasks.add_task(_invalidate_cache_after_upload, int(doc_id))
        return UploadResponse(**result)
    except ValueError as e:
        # Clean up file — document không đúng định dạng văn bản pháp luật (Điều/Khoản).
        shutil.rmtree(upload_dir, ignore_errors=True)
        log.warning(
            "POST /api/upload → 400 | file=%r | %s",
            file.filename,
            e,
        )
        raise HTTPException(400, str(e))
    except Exception as e:
        # Clean up on failure
        shutil.rmtree(upload_dir, ignore_errors=True)
        log.error("Ingestion failed for '%s': %s", file.filename, e)
        raise HTTPException(500, f"Ingestion failed: {e}")


@router.post("/upload-folder")
async def upload_folder(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    issuer: str = Form(default=""),
    db: AsyncSession = Depends(get_db),
):
    """Upload and ingest multiple legal DOC/DOCX files in one request."""
    if not files:
        raise HTTPException(400, "Không có file nào được tải lên")

    total_files = len(files)
    log.info("═══ FOLDER UPLOAD START: %d files ═══", total_files)

    results: list[dict] = []
    success_count = 0
    skipped_count = 0

    for idx, file in enumerate(files, 1):
        safe_name = _safe_filename(file.filename or "")
        file_name = file.filename or safe_name
        if not safe_name.lower().endswith(ALLOWED_EXTENSIONS):
            skipped_count += 1
            log.warning(
                "  [%d/%d] SKIP '%s' – not .doc/.docx",
                idx, total_files, safe_name,
            )
            results.append(
                {
                    "file_name": file_name,
                    "dataset_id": None,
                    "total_chunks": 0,
                    "total_chars": 0,
                    "success": False,
                    "error": "Chỉ hỗ trợ file .doc và .docx",
                }
            )
            continue

        upload_id = uuid.uuid4().hex[:12]
        upload_dir = STORAGE_DIR / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / safe_name

        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            log.info(
                "  [%d/%d] SAVED '%s' → %d bytes",
                idx, total_files, safe_name, len(content),
            )

            ingest_result = await ingest_document(
                file_path=file_path,
                db=db,
                issuer=issuer,
            )
            await db.commit()

            doc_id = ingest_result.get("document_id")
            if doc_id:
                try:
                    await auto_index_from_document(int(doc_id))
                except Exception as e_idx:
                    log.warning("Auto-index failed for doc_id=%s: %s", doc_id, e_idx)

            chunks = int(ingest_result.get("chunks", 0))
            results.append(
                {
                    "file_name": file_name,
                    "dataset_id": str(ingest_result.get("document_id")),
                    "total_chunks": chunks,
                    "total_chars": 0,
                    "success": True,
                    "error": None,
                }
            )
            success_count += 1
            if ingest_result.get("document_id"):
                background_tasks.add_task(
                    _invalidate_cache_after_upload, int(ingest_result["document_id"])
                )
            log.info(
                "  [%d/%d] OK '%s' → doc_id=%s, chunks=%d",
                idx, total_files, safe_name,
                ingest_result.get("document_id"), chunks,
            )
        except Exception as e:
            await db.rollback()
            shutil.rmtree(upload_dir, ignore_errors=True)
            log.error(
                "  [%d/%d] FAIL '%s' → %s",
                idx, total_files, safe_name, e,
            )
            results.append(
                {
                    "file_name": file_name,
                    "dataset_id": None,
                    "total_chunks": 0,
                    "total_chars": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    fail_count = total_files - success_count - skipped_count
    log.info(
        "═══ FOLDER UPLOAD DONE: %d total, %d success, %d failed, %d skipped ═══",
        total_files, success_count, fail_count, skipped_count,
    )
    return {
        "total_files": total_files,
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results,
        "message": (
            f"Đã xử lý {total_files} file: {success_count} thành công, {fail_count} thất bại"
        ),
    }


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    document_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all ingested documents with optional filtering."""
    stmt = select(Document).order_by(Document.created_at.desc())

    if document_type:
        stmt = stmt.where(Document.document_type == document_type)

    # Count total
    count_stmt = select(func.count()).select_from(Document)
    if document_type:
        count_stmt = count_stmt.where(Document.document_type == document_type)
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = stmt.offset(skip).limit(limit)
    result = await db.execute(stmt)
    docs = result.scalars().all()

    return DocumentListResponse(
        documents=[DocumentInfo.model_validate(d) for d in docs],
        total=total,
    )


@router.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
async def get_document(doc_id: int, db: AsyncSession = Depends(get_db)):
    """Get document details by ID."""
    stmt = select(Document).where(Document.id == doc_id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(404, "Document not found")

    # Count articles and chunks
    articles_count = (await db.execute(
        select(func.count()).select_from(Article).where(Article.document_id == doc_id)
    )).scalar() or 0

    chunks_count = (await db.execute(
        select(func.count()).select_from(VectorChunk).where(VectorChunk.document_id == doc_id)
    )).scalar() or 0

    doc_dict = {
        "id": doc.id,
        "doc_number": doc.doc_number,
        "title": doc.title,
        "document_type": doc.document_type,
        "issuer": doc.issuer,
        "issued_date": doc.issued_date,
        "effective_date": doc.effective_date,
        "file_path": doc.file_path,
        "created_at": doc.created_at,
        "articles_count": articles_count,
        "chunks_count": chunks_count,
    }
    return DocumentDetailResponse(**doc_dict)


# ── Datasets endpoints (frontend-compatible) ──────────────

@router.get("/datasets")
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """List all documents as datasets (frontend-compatible format)."""
    try:
        stmt = select(Document).order_by(Document.created_at.desc())
        result = await db.execute(stmt)
        docs = result.scalars().all()
    except asyncio.CancelledError:
        return {"datasets": []}

    datasets = []
    for doc in docs:
        file_name = Path(doc.file_path).name if doc.file_path else doc.doc_number
        datasets.append({
            "dataset_id": doc.id,
            "file_name": file_name,
            "doc_number": doc.doc_number,
            "title": doc.title,
            "document_type": doc.document_type,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        })

    return {"datasets": datasets}


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a document and its associated vectors, articles, clauses, chunks."""
    stmt = select(Document).where(Document.id == dataset_id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(404, "Dataset not found")

    # Delete vectors from Qdrant
    try:
        delete_by_document_id(dataset_id)
    except Exception as e:
        log.warning("Failed to delete Qdrant vectors for doc %d: %s", dataset_id, e)

    # Delete file from storage
    if doc.file_path:
        upload_dir = Path(doc.file_path).parent
        shutil.rmtree(upload_dir, ignore_errors=True)

    # Delete from PostgreSQL (CASCADE handles articles, clauses, vector_chunks)
    await db.execute(sa_delete(Document).where(Document.id == dataset_id))

    return {"status": "ok", "deleted_id": dataset_id}
