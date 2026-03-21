"""Async SQLAlchemy session management for PostgreSQL.

Provides:
- ``async_engine``: the single async engine
- ``get_db()``: async generator yielding ``AsyncSession`` (for FastAPI Depends)
- ``init_db()``: run on startup to create all tables
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import POSTGRES_URL
from app.database.models import Base

log = logging.getLogger(__name__)

# ── Engine (lazy-created once) ─────────────────────────────

async_engine = create_async_engine(
    POSTGRES_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

_session_factory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Public helpers ─────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency – yields an async session, auto-commits on success."""
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except asyncio.CancelledError:
            await session.rollback()
            raise
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Standalone async context manager for use outside FastAPI Depends."""
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except asyncio.CancelledError:
            await session.rollback()
            raise
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables if they don't exist (run at app startup)."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "ALTER TABLE documents "
                "ALTER COLUMN doc_number TYPE VARCHAR(255)"
            )
        )
        # Bảng cũ có thể tạo title/issuer/file_path VARCHAR(255) — canonical title + path dài cần TEXT
        for col in ("title", "issuer", "file_path"):
            await conn.execute(
                text(f"ALTER TABLE documents ALTER COLUMN {col} TYPE TEXT")
            )
        await conn.execute(
            text(
                "ALTER TABLE vector_chunks "
                "ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(20) DEFAULT 'clause'"
            )
        )
        # Chương + Mục: add FK columns to articles if upgrading from old schema
        await conn.execute(
            text(
                "ALTER TABLE articles "
                "ADD COLUMN IF NOT EXISTS chapter_id INTEGER REFERENCES chapters(id) ON DELETE SET NULL"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE articles "
                "ADD COLUMN IF NOT EXISTS section_id INTEGER REFERENCES sections(id) ON DELETE SET NULL"
            )
        )
    log.info("PostgreSQL tables ensured.")


async def close_db() -> None:
    """Dispose the connection pool (run at app shutdown)."""
    await async_engine.dispose()
    log.info("PostgreSQL connection pool disposed.")
