"""Cache package – Redis caching layer."""

from app.cache.redis_cache import (
    get_cached_answer,
    cache_answer,
    invalidate_cache,
    flush_cache,
    invalidate_all_query_cache,
    invalidate_cache_for_document,
)

__all__ = [
    "get_cached_answer",
    "cache_answer",
    "invalidate_cache",
    "flush_cache",
    "invalidate_all_query_cache",
    "invalidate_cache_for_document",
]
