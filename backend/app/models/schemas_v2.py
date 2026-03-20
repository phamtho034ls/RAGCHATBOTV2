"""Pydantic request/response schemas for the upgraded legal chatbot API."""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ── Chat ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: Optional[str] = Field(default=None, min_length=1, max_length=5000)
    question: Optional[str] = Field(default=None, min_length=1, max_length=5000)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    doc_number: Optional[str] = None
    conversation_id: Optional[str] = None

    @model_validator(mode="after")
    def resolve_query(self):
        """Accept both ``question`` (frontend) and ``query`` (v2 native)."""
        if self.query is None and self.question is not None:
            self.query = self.question
        if self.query is None:
            raise ValueError("Either 'query' or 'question' must be provided")
        return self


class SourceInfo(BaseModel):
    citation: str = ""
    document_title: str = ""
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    snippet: str = ""
    document_id: Optional[int] = None
    article_id: Optional[int] = None
    clause_id: Optional[int] = None
    doc_number: str = ""
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []
    confidence_score: float = 0.0


# ── Upload ────────────────────────────────────────────────

class UploadRequest(BaseModel):
    issuer: str = ""
    issued_date: Optional[date] = None
    effective_date: Optional[date] = None
    title: Optional[str] = None


class UploadResponse(BaseModel):
    document_id: int
    doc_number: str
    title: str
    document_type: str
    articles: int
    clauses: int
    chunks: int


# ── Documents ─────────────────────────────────────────────

class DocumentInfo(BaseModel):
    id: int
    doc_number: Optional[str] = None
    title: Optional[str] = None
    document_type: Optional[str] = None
    issuer: Optional[str] = None
    issued_date: Optional[date] = None
    effective_date: Optional[date] = None
    file_path: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DocumentDetailResponse(DocumentInfo):
    articles_count: int = 0
    chunks_count: int = 0


# ── Health ────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    postgres: bool = False
    qdrant: bool = False
    redis: bool = False
    embedding_model: str = ""
    reranker_model: str = ""
