#!/usr/bin/env python3
"""Batch re-embed mọi vector_chunks → Qdrant (không chạy server).

Chạy sau khi đổi embedding model / chiều vector:
  cd backend && python scripts/reembed_all.py

Yêu cầu: Qdrant collection đã đúng EMBEDDING_DIM, PostgreSQL có vector_chunks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("reembed")


async def main() -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    from sqlalchemy import select

    from app.database.models import VectorChunk, Document, Article, Clause
    from app.database.session import get_db_context
    from app.pipeline.embedding import embed_texts, warmup as warmup_embeddings
    from app.pipeline.vector_store import upsert_vectors_with_ids

    warmup_embeddings()

    batch_size = 32
    offset = 0
    total_updated = 0

    async with get_db_context() as db:
        from sqlalchemy import func

        cnt = (await db.execute(select(func.count()).select_from(VectorChunk))).scalar() or 0
        log.info("Total chunks: %d", cnt)

        while True:
            stmt = (
                select(
                    VectorChunk.id,
                    VectorChunk.vector_id,
                    VectorChunk.chunk_text,
                    VectorChunk.document_id,
                    VectorChunk.article_id,
                    VectorChunk.clause_id,
                    VectorChunk.chunk_type,
                    Document.doc_number,
                    Document.title.label("document_title"),
                    Document.document_type,
                    Article.article_number,
                    Article.title.label("article_title"),
                    Clause.clause_number,
                )
                .join(Document, VectorChunk.document_id == Document.id)
                .outerjoin(Article, VectorChunk.article_id == Article.id)
                .outerjoin(Clause, VectorChunk.clause_id == Clause.id)
                .order_by(VectorChunk.id)
                .offset(offset)
                .limit(batch_size * 4)
            )
            rows = (await db.execute(stmt)).all()
            if not rows:
                break

            texts = [r.chunk_text or "" for r in rows]
            iterator = range(0, len(texts), batch_size)
            if tqdm:
                iterator = tqdm(list(iterator), desc="Re-embed batches", unit="batch")

            for start in iterator:
                batch_rows = rows[start : start + batch_size]
                if not batch_rows:
                    continue
                tbatch = [r.chunk_text or "" for r in batch_rows]
                emb = embed_texts(tbatch, batch_size=batch_size)
                vecs = emb.tolist()
                payloads = []
                ids = []
                for r in batch_rows:
                    ids.append(str(r.vector_id))
                    payloads.append(
                        {
                            "document_id": r.document_id,
                            "article_id": r.article_id,
                            "clause_id": r.clause_id,
                            "law_name": r.document_title or "",
                            "document_title": r.document_title or "",
                            "doc_number": r.doc_number or "",
                            "document_type": r.document_type or "",
                            "article_number": (r.article_number or "") if r.article_number else "",
                            "article_title": r.article_title or "",
                            "clause_number": (r.clause_number or "") if r.clause_number else "",
                            "legal_domain": "",
                            "year": "",
                            "chapter": "",
                            "section": "",
                            "chunk_type": (r.chunk_type or "clause") if hasattr(r, "chunk_type") else "clause",
                            "text_chunk": r.chunk_text or "",
                        }
                    )
                upsert_vectors_with_ids(vecs, payloads, ids)
                total_updated += len(vecs)

            offset += len(rows)
            log.info("Progress: %d / %d chunks", offset, cnt)

    log.info("Done. Re-embedded %d chunk vectors.", total_updated)


if __name__ == "__main__":
    asyncio.run(main())
