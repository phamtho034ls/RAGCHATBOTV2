"""Clause-first chunk generation for Vietnamese legal documents."""

from __future__ import annotations

from typing import List

from app.pipeline.legal_chunker import Chunk, chunk_by_clause
from app.pipeline.structure_detector import ArticleItem


def generate_clause_chunks(
    articles: List[ArticleItem],
    *,
    law_title: str,
    doc_number: str,
    document_type: str,
) -> List[Chunk]:
    """Generate chunks where each primary unit is one legal clause."""
    return chunk_by_clause(
        articles,
        document_title=law_title,
        doc_number=doc_number,
        document_type=document_type,
    )

