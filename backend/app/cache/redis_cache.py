"""Redis caching layer for frequently asked questions.

Architecture:
  User → API → Redis Cache → Hybrid Retrieval → LLM

Cache keys are derived from normalized query text.
TTL is configurable (default: 1 hour).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, Optional

from app.config import REDIS_URL, REDIS_CACHE_TTL

log = logging.getLogger(__name__)

_redis = None


async def _get_redis():
    """Lazy-initialize async Redis client."""
    global _redis
    if _redis is None:
        try:
            from redis.asyncio import from_url
            _redis = from_url(REDIS_URL, decode_responses=True)
            await _redis.ping()
            log.info("Connected to Redis at %s", REDIS_URL)
        except Exception as e:
            log.warning("Redis unavailable (%s). Caching disabled.", e)
            _redis = False  # sentinel: don't retry
    return _redis if _redis else None


def _cache_key(query: str) -> str:
    """Generate a deterministic cache key from a query."""
    normalized = query.strip().lower()
    h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"legal_chat:{h}"


async def get_cached_answer(query: str) -> Optional[Dict]:
    """Look up a cached answer for the given query."""
    redis = await _get_redis()
    if not redis:
        return None

    try:
        key = _cache_key(query)
        data = await redis.get(key)
        if data:
            return json.loads(data)
    except Exception as e:
        log.warning("Redis GET error: %s", e)
    return None


async def cache_answer(query: str, result: Dict) -> None:
    """Cache a query-answer pair with TTL."""
    redis = await _get_redis()
    if not redis:
        return

    try:
        key = _cache_key(query)
        await redis.setex(key, REDIS_CACHE_TTL, json.dumps(result, ensure_ascii=False))
    except Exception as e:
        log.warning("Redis SET error: %s", e)


async def invalidate_cache(query: str) -> None:
    """Remove a specific cached answer."""
    redis = await _get_redis()
    if not redis:
        return

    try:
        key = _cache_key(query)
        await redis.delete(key)
    except Exception as e:
        log.warning("Redis DELETE error: %s", e)


async def flush_cache() -> None:
    """Clear all cached answers."""
    await invalidate_all_query_cache()


async def invalidate_all_query_cache() -> int:
    """Xóa toàn bộ query cache (prefix legal_chat:). Giữ embedding cache (in-process)."""
    redis = await _get_redis()
    if not redis:
        return 0
    try:
        keys = []
        async for key in redis.scan_iter("legal_chat:*"):
            keys.append(key)
        if keys:
            await redis.delete(*keys)
            log.info("Invalidated %d query cache entries (legal_chat:*).", len(keys))
        return len(keys)
    except Exception as e:
        log.warning("Redis invalidate_all_query_cache error: %s", e)
        return 0


async def invalidate_cache_for_document(document_id: int) -> int:
    """Best-effort: xóa cache entries có document_id trong payload JSON (nếu có)."""
    redis = await _get_redis()
    if not redis:
        return 0
    removed = 0
    try:
        async for key in redis.scan_iter("legal_chat:*"):
            try:
                raw = await redis.get(key)
                if not raw:
                    continue
                import json as _json

                data = _json.loads(raw)
                sources = data.get("sources") or []
                for s in sources:
                    if s.get("document_id") == document_id:
                        await redis.delete(key)
                        removed += 1
                        break
            except Exception:
                continue
        if removed:
            log.info("invalidate_cache_for_document: removed %d keys for doc_id=%s", removed, document_id)
    except Exception as e:
        log.warning("invalidate_cache_for_document failed: %s", e)
    return removed
