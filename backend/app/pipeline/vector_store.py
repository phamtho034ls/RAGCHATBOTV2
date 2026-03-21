"""Qdrant vector database integration.

Collection: ``law_documents``

Each point stores:
- vector: embedding from sentence-transformer
- payload: document_id, article_id, clause_id, text_chunk, doc_number
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import (
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_RECREATE_ON_DIM_MISMATCH,
)

# Tránh import vòng: chỉ gọi sau khi embedding model đã warmup
def _vector_size() -> int:
    try:
        from app.pipeline.embedding import get_embedding_dimension

        return int(get_embedding_dimension())
    except Exception:
        return int(EMBEDDING_DIM)

log = logging.getLogger(__name__)

# ── Singleton client ───────────────────────────────────────

_client: Optional[QdrantClient] = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        kwargs: Dict = {"host": QDRANT_HOST, "port": QDRANT_PORT, "timeout": 30}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        _client = QdrantClient(**kwargs)
        log.info("Connected to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
    return _client


# ── Collection management ─────────────────────────────────

def _collection_vector_dim(client: QdrantClient, collection_name: str) -> Optional[int]:
    """Đọc kích thước vector đã cấu hình trong Qdrant (single vector hoặc named)."""
    try:
        info = client.get_collection(collection_name=collection_name)
    except Exception:
        return None
    params = getattr(info.config, "params", None)
    if params is None:
        return None
    vc = params.vectors
    if vc is None:
        return None
    if hasattr(vc, "size"):
        return int(vc.size)
    if isinstance(vc, dict):
        for v in vc.values():
            if v is not None and hasattr(v, "size"):
                return int(v.size)
    return None


def ensure_collection() -> None:
    """Create the ``law_documents`` collection if it doesn't exist; align dim with embedding model."""
    client = _get_client()
    collections = [c.name for c in client.get_collections().collections]
    dim = _vector_size()
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
        )
        log.info("Created Qdrant collection '%s' (dim=%d).", QDRANT_COLLECTION, dim)
        return

    existing = _collection_vector_dim(client, QDRANT_COLLECTION)
    if existing is not None and existing != dim:
        log.warning(
            "Qdrant collection '%s' đang dùng vector size=%d nhưng model cần %d.",
            QDRANT_COLLECTION,
            existing,
            dim,
        )
        if not QDRANT_RECREATE_ON_DIM_MISMATCH:
            raise RuntimeError(
                f"Qdrant '{QDRANT_COLLECTION}': vector dim {existing} != {dim} (model). "
                "Xóa collection trên Qdrant hoặc bật QDRANT_RECREATE_ON_DIM_MISMATCH=true "
                "(sẽ xóa toàn bộ point trong collection — chạy scripts/reembed_all.py để nạp lại)."
            )
        log.warning(
            "Đang xóa và tạo lại collection '%s' — toàn bộ vector cũ mất. "
            "Sau đó chạy: python scripts/reembed_all.py",
            QDRANT_COLLECTION,
        )
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
        )
        log.info("Recreated Qdrant collection '%s' (dim=%d).", QDRANT_COLLECTION, dim)
        return

    log.info(
        "Qdrant collection '%s' OK (dim=%d, theo model/config).",
        QDRANT_COLLECTION,
        dim,
    )


# ── Insert ────────────────────────────────────────────────

def upsert_vectors(
    vectors: List[List[float]],
    payloads: List[Dict],
    batch_size: int = 100,
) -> List[str]:
    """Batch upsert vectors into Qdrant. Returns list of point IDs."""
    client = _get_client()
    point_ids: List[str] = []

    for start in range(0, len(vectors), batch_size):
        batch_vectors = vectors[start : start + batch_size]
        batch_payloads = payloads[start : start + batch_size]

        points = []
        for vec, payload in zip(batch_vectors, batch_payloads):
            pid = str(uuid.uuid4())
            point_ids.append(pid)
            points.append(
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
        )

    log.info("Upserted %d vectors into '%s'.", len(vectors), QDRANT_COLLECTION)
    return point_ids


def upsert_vectors_with_ids(
    vectors: List[List[float]],
    payloads: List[Dict],
    point_ids: List[str],
    batch_size: int = 100,
) -> None:
    """Upsert vào Qdrant với ID cố định (re-embedding / batch cập nhật)."""
    if len(vectors) != len(payloads) or len(vectors) != len(point_ids):
        raise ValueError("vectors, payloads, point_ids must have same length")
    client = _get_client()
    for start in range(0, len(vectors), batch_size):
        batch_v = vectors[start : start + batch_size]
        batch_p = payloads[start : start + batch_size]
        batch_ids = point_ids[start : start + batch_size]
        points = [
            PointStruct(id=pid, vector=vec, payload=payload)
            for vec, payload, pid in zip(batch_v, batch_p, batch_ids)
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    log.info("Upserted %d vectors (fixed IDs) into '%s'.", len(vectors), QDRANT_COLLECTION)


# ── Search ────────────────────────────────────────────────

def search_vectors(
    query_vector: List[float],
    top_k: int = 20,
    doc_number: Optional[str] = None,
    document_id: Optional[int] = None,
    legal_domains: Optional[List[str]] = None,
    document_type: Optional[str] = None,
) -> List[Dict]:
    """Semantic search in Qdrant with optional metadata filtering.

    Args:
        legal_domains: Filter by legal domain tags (OR logic: match any).
        document_type:  Filter by document type (e.g. "Luật", "Nghị định").
    """
    client = _get_client()

    query_filter = None
    conditions = []
    if doc_number:
        conditions.append(FieldCondition(key="doc_number", match=MatchValue(value=doc_number)))
    if document_id is not None:
        conditions.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    if legal_domains:
        conditions.append(FieldCondition(key="legal_domain", match=MatchAny(any=legal_domains)))
    if document_type:
        conditions.append(FieldCondition(key="document_type", match=MatchValue(value=document_type)))
    if conditions:
        query_filter = Filter(must=conditions)

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            **hit.payload,
        }
        for hit in results.points
    ]


# ── Delete ────────────────────────────────────────────────

def delete_by_document_id(document_id: int) -> None:
    """Delete all vectors belonging to a specific document."""
    client = _get_client()
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
        ),
    )
    log.info("Deleted vectors for document_id=%d from '%s'.", document_id, QDRANT_COLLECTION)


def get_collection_info() -> Dict:
    """Return collection stats."""
    client = _get_client()
    info = client.get_collection(collection_name=QDRANT_COLLECTION)
    return {
        "name": QDRANT_COLLECTION,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": info.status.value,
    }
