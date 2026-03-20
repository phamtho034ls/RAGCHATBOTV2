"""
Search Router – API endpoint cho Knowledge Search.

Endpoints:
    POST /api/search  — Tìm kiếm vector database, trả về top documents
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import SearchRequest, SearchResponse, SourceChunk
from app.services.retrieval import search_all, has_any_dataset

router = APIRouter(tags=["search"])


@router.post("/api/search", response_model=SearchResponse)
async def search_documents(req: SearchRequest):
    """Tìm kiếm văn bản trong vector database.

    Pipeline: Qdrant + PostgreSQL → Hybrid Retrieval → Cross-encoder Reranking → Top-K

    Trả về danh sách documents kèm điểm relevance.
    """
    if not req.query.strip():
        raise HTTPException(400, "Câu truy vấn không được để trống")

    if not await has_any_dataset():
        raise HTTPException(404, "Chưa có tài liệu nào được tải lên")

    filters = req.filters.to_dict() if req.filters else None
    results = await search_all(req.query, filters=filters, top_k=req.top_k)

    sources = [
        SourceChunk(
            content=doc["text"],
            score=round(doc.get("rerank_score", doc.get("score", 0)), 4),
            dataset_id=doc.get("dataset_id", ""),
            metadata=doc.get("metadata", {}),
        )
        for doc in results
    ]

    return SearchResponse(results=sources, total=len(sources))
