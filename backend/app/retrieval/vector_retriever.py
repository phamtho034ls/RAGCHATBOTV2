"""Vector retriever – semantic search via Qdrant.

Queries the Qdrant ``law_documents`` collection and returns
top-K passages ranked by cosine similarity.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.pipeline.embedding import embed_query
from app.pipeline.vector_store import search_vectors

log = logging.getLogger(__name__)


def vector_search(
    query: str,
    top_k: int = 20,
    doc_number: Optional[str] = None,
    document_id: Optional[int] = None,
    legal_domains: Optional[List[str]] = None,
    document_type: Optional[str] = None,
) -> List[Dict]:
    """Run semantic vector search with optional metadata filtering.

    Returns list of dicts with keys:
        id, score, text_chunk, document_id, article_id, clause_id, doc_number
    """
    query_vec = embed_query(query)
    results = search_vectors(
        query_vector=query_vec,
        top_k=top_k,
        doc_number=doc_number,
        document_id=document_id,
        legal_domains=legal_domains,
        document_type=document_type,
    )
    log.debug(
        "Vector search returned %d results for: '%.60s...' (domains=%s)",
        len(results), query, legal_domains or "all",
    )
    return results
