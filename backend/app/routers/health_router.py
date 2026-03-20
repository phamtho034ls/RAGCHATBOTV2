"""Health check router – GET /api/health."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.config import EMBEDDING_MODEL, RERANKER_MODEL
from app.models.schemas_v2 import HealthResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check – reports status of all dependencies."""
    pg_ok = await _check_postgres()
    qdrant_ok = _check_qdrant()
    redis_ok = await _check_redis()

    status = "ok" if (pg_ok and qdrant_ok) else "degraded"

    return HealthResponse(
        status=status,
        postgres=pg_ok,
        qdrant=qdrant_ok,
        redis=redis_ok,
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
    )


async def _check_postgres() -> bool:
    try:
        from sqlalchemy import text
        from app.database.session import async_engine

        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.warning("PostgreSQL health check failed: %s", e)
        return False


def _check_qdrant() -> bool:
    try:
        from app.pipeline.vector_store import get_collection_info
        info = get_collection_info()
        return info.get("status") == "green"
    except Exception as e:
        log.warning("Qdrant health check failed: %s", e)
        return False


async def _check_redis() -> bool:
    try:
        from app.cache.redis_cache import _get_redis
        redis = await _get_redis()
        if redis:
            await redis.ping()
            return True
        return False
    except Exception as e:
        log.warning("Redis health check failed: %s", e)
        return False
