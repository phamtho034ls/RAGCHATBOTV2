"""FastAPI entry-point – Legal Chatbot API v2.

Production-ready architecture:
  PostgreSQL + Qdrant + Redis + Hybrid Retrieval + Reranking
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.session import init_db as init_postgres, close_db as close_postgres
from app.pipeline.vector_store import ensure_collection
from app.pipeline.embedding import warmup as warmup_embeddings
from app.retrieval.reranker import warmup as warmup_reranker
from app.services.intent_detector import warmup_intent_index
from app.services.intent_classifier import warmup_intent_index as warmup_rag_intent_index
from app.services.domain_classifier import warmup_domain_index

# ── v2 Routers ────────────────────────────────────────────
from app.routers.chat_router import router as chat_router
from app.routers.document_router_v2 import router as document_router
from app.routers.health_router import router as health_router

# ── Legacy Routers (refactored to v2 infrastructure) ──────
from app.routers.copilot_router import router as copilot_router
from app.routers.procedure_router import router as procedure_router
from app.routers.document_router import router as legacy_document_router
from app.routers.intent_router import router as intent_router
from app.routers.search_router import router as search_router
from app.routers.tools_router import router as tools_router
from app.routers.conversation_router import router as conversation_router

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # 1. PostgreSQL tables
    await init_postgres()
    log.info("PostgreSQL initialized.")

    # 2. Qdrant collection
    ensure_collection()
    log.info("Qdrant collection ensured.")

    # 3. Embedding model warm-up
    warmup_embeddings()
    log.info("Embedding model ready.")

    # 4. Reranker model warm-up
    warmup_reranker()
    log.info("Reranker model ready.")

    # 5. Intent prototype semantic index
    warmup_intent_index()
    log.info("Intent semantic index ready.")

    # 6. RAG intent classifier (scenario / multi-article / expansion)
    warmup_rag_intent_index()
    log.info("RAG intent classifier index ready.")

    # 7. Legal domain classification index
    warmup_domain_index()
    log.info("Legal domain index ready.")

    log.info("Startup complete.")
    yield
    log.info("Shutting down…")
    await close_postgres()
    log.info("Shutdown complete.")


app = FastAPI(
    title="Government AI Copilot API",
    version="2.0.0",
    description="Production-ready legal chatbot for Vietnamese law at provincial scale.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── v2 Routers ────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(document_router)
app.include_router(health_router)

# ── Legacy Routers (now on v2 infrastructure) ─────────────
app.include_router(copilot_router)
app.include_router(procedure_router)
app.include_router(legacy_document_router)
app.include_router(intent_router)
app.include_router(search_router)
app.include_router(tools_router)
app.include_router(conversation_router)


# ── GPU status (lightweight, no v1 dependency) ────────────
@app.get("/api/gpu")
async def gpu_status():
    from app.services.gpu_info import get_gpu_status
    return get_gpu_status()
