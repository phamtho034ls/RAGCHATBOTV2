-- PostgreSQL schema for Legal Chatbot v2
-- Run: psql -U legal_bot -d legal_chatbot -f schema.sql

-- ── Documents ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    doc_number      VARCHAR(100),
    title           TEXT,
    document_type   VARCHAR(50),
    issuer          VARCHAR(255),
    issued_date     DATE,
    effective_date  DATE,
    file_path       TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_number ON documents(doc_number);
CREATE INDEX IF NOT EXISTS idx_doc_number_type ON documents(doc_number, document_type);
CREATE INDEX IF NOT EXISTS idx_doc_issuer ON documents(issuer);
CREATE INDEX IF NOT EXISTS idx_doc_effective_date ON documents(effective_date);
CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(document_type);

-- ── Articles ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS articles (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    article_number  VARCHAR(20),
    title           TEXT,
    content         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_article_document_id ON articles(document_id);
CREATE INDEX IF NOT EXISTS idx_article_doc_num ON articles(document_id, article_number);

-- ── Clauses ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS clauses (
    id              SERIAL PRIMARY KEY,
    article_id      INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    clause_number   VARCHAR(20),
    content         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_clause_article_id ON clauses(article_id);

-- ── Vector Chunks ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS vector_chunks (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    article_id      INTEGER REFERENCES articles(id) ON DELETE SET NULL,
    clause_id       INTEGER REFERENCES clauses(id) ON DELETE SET NULL,
    vector_id       VARCHAR(64) NOT NULL UNIQUE,
    chunk_text      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vchunk_vector_id ON vector_chunks(vector_id);
CREATE INDEX IF NOT EXISTS idx_vchunk_document_id ON vector_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_vchunk_doc_article ON vector_chunks(document_id, article_id);

-- ── Chat Logs ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS chat_logs (
    id              SERIAL PRIMARY KEY,
    user_query      TEXT NOT NULL,
    chatbot_answer  TEXT,
    documents_used  TEXT,
    confidence_score FLOAT,
    latency_ms      FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chatlog_created ON chat_logs(created_at);
