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
    redis = await _get_redis()
    if not redis:
        return

    try:
        keys = []
        async for key in redis.scan_iter("legal_chat:*"):
            keys.append(key)
        if keys:
            await redis.delete(*keys)
            log.info("Flushed %d cached answers.", len(keys))
    except Exception as e:
        log.warning("Redis FLUSH error: %s", e)
