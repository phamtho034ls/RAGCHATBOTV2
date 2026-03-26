"""SQLAlchemy ORM models for the legal chatbot database.

Normalized schema optimized for Vietnamese legal documents at scale:
- documents: top-level legal document metadata
- chapters: Chương within a document
- sections: Mục within a chapter (optional level)
- articles: individual articles (Điều) within a document/chapter/section
- clauses: subsections (Khoản/Điểm) within an article
- vector_chunks: text chunks linked to their source for RAG retrieval
- chat_logs: full interaction audit log
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# ── Documents ──────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_number = Column(String(255), index=True, comment="e.g. 13/2025/TT-BVHTTDL")
    title = Column(Text)
    document_type = Column(String(50), index=True, comment="Luật, Nghị định, Thông tư, ...")
    issuer = Column(  # Theo pipeline ingest: trích yếu (vd. dòng "VỀ VIỆC ..."), không phải tên cơ quan
        Text,
        comment="Trích yếu / chủ đề văn bản (vd. VỀ VIỆC ...); do ingest tự trích",
    )
    issued_date = Column(Date)
    effective_date = Column(Date)
    file_path = Column(Text)
    law_intents = Column(
        JSON,
        nullable=True,
        comment="Multi-label legal topic tags (same vocabulary as LEGAL_DOMAINS / query domain classifier)",
    )
    created_at = Column(DateTime, server_default=func.now())

    # relationships
    chapters = relationship("Chapter", back_populates="document", cascade="all, delete-orphan")
    articles = relationship("Article", back_populates="document", cascade="all, delete-orphan")
    vector_chunks = relationship("VectorChunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_doc_number_type", "doc_number", "document_type"),
        Index("idx_doc_issuer", "issuer"),
        Index("idx_doc_effective_date", "effective_date"),
    )


# ── Chapters (Chương) ───────────────────────────────────────

class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chapter_number = Column(String(20), nullable=False, comment="e.g. 'I', 'II', '1'")
    title = Column(Text)
    sort_order = Column(Integer, default=0)

    # relationships
    document = relationship("Document", back_populates="chapters")
    sections = relationship("Section", back_populates="chapter", cascade="all, delete-orphan")
    articles = relationship("Article", back_populates="chapter", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_chapter_doc_order", "document_id", "sort_order"),
    )


# ── Sections (Mục) ──────────────────────────────────────────

class Section(Base):
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False, index=True)
    section_number = Column(String(20), nullable=False, comment="e.g. '1', '2'")
    title = Column(Text)
    sort_order = Column(Integer, default=0)

    # relationships
    chapter = relationship("Chapter", back_populates="sections")
    articles = relationship("Article", back_populates="section", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_section_chapter_order", "chapter_id", "sort_order"),
    )


# ── Articles ───────────────────────────────────────────────

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id", ondelete="SET NULL"), nullable=True, index=True)
    section_id = Column(Integer, ForeignKey("sections.id", ondelete="SET NULL"), nullable=True, index=True)
    article_number = Column(String(20), comment="e.g. 'Điều 5'")
    title = Column(Text)
    content = Column(Text, nullable=False)

    # relationships
    document = relationship("Document", back_populates="articles")
    chapter = relationship("Chapter", back_populates="articles")
    section = relationship("Section", back_populates="articles")
    clauses = relationship("Clause", back_populates="article", cascade="all, delete-orphan")
    vector_chunks = relationship("VectorChunk", back_populates="article", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_article_doc_num", "document_id", "article_number"),
        Index("idx_article_chapter", "chapter_id"),
        Index("idx_article_section", "section_id"),
    )


# ── Clauses ────────────────────────────────────────────────

class Clause(Base):
    __tablename__ = "clauses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True)
    clause_number = Column(String(20), comment="e.g. 'Khoản 3' or 'Điểm a'")
    content = Column(Text, nullable=False)

    # relationships
    article = relationship("Article", back_populates="clauses")
    vector_chunks = relationship("VectorChunk", back_populates="clause", cascade="all, delete-orphan")


# ── Vector Chunks ──────────────────────────────────────────

class VectorChunk(Base):
    __tablename__ = "vector_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="SET NULL"), nullable=True)
    clause_id = Column(Integer, ForeignKey("clauses.id", ondelete="SET NULL"), nullable=True)
    vector_id = Column(String(64), unique=True, nullable=False, comment="Qdrant point ID (UUID)")
    chunk_text = Column(Text, nullable=False)
    chunk_type = Column(String(20), server_default="clause", comment="article | clause | token_sub")

    # relationships
    document = relationship("Document", back_populates="vector_chunks")
    article = relationship("Article", back_populates="vector_chunks")
    clause = relationship("Clause", back_populates="vector_chunks")

    __table_args__ = (
        Index("idx_vchunk_vector_id", "vector_id"),
        Index("idx_vchunk_doc_article", "document_id", "article_id"),
        Index("idx_vchunk_type", "chunk_type"),
    )


# ── Chat Logs ─────────────────────────────────────────────

class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_query = Column(Text, nullable=False)
    chatbot_answer = Column(Text)
    documents_used = Column(Text, comment="JSON array of doc_numbers used")
    confidence_score = Column(Float)
    latency_ms = Column(Float, comment="Response time in milliseconds")
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_chatlog_created", "created_at"),
    )


# ── Chat conversations (persistent, /api/chat + CRUD) ─────

class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    id = Column(String(32), primary_key=True, comment="Client-facing conversation id (hex)")
    title = Column(String(512), nullable=False)
    context_json = Column(Text, nullable=True, comment="JSON: last_topic, document_history, …")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    messages = relationship(
        "ChatMessage",
        back_populates="conversation",
        order_by="ChatMessage.id",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("idx_chat_conv_updated", "updated_at"),)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        String(32),
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    conversation = relationship("ChatConversation", back_populates="messages")

    __table_args__ = (Index("idx_chat_msg_conv_id", "conversation_id", "id"),)
