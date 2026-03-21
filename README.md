# 🏛️ Government AI Copilot – Trợ lý AI hành chính thực chiến cấp xã

**Government AI Copilot** là hệ thống **trợ lý hành chính AI thực chiến** dành cho cán bộ Văn hóa – Xã hội (VHXH) cấp xã, xây dựng trên kiến trúc **Production-Grade RAG + Agent Tools**. Kết hợp **LLM qua OpenAI API**, **Qdrant vector search + PostgreSQL hybrid keyword** (full-text `articles` với `unaccent` + tsvector, fallback ILIKE trên `vector_chunks`), **Redis caching**, **reranking** (**FlagEmbedding** `FlagReranker` với `BAAI/bge-reranker-v2-m3`; fallback **CrossEncoder** `BAAI/bge-reranker-base`), **GPU NVIDIA CUDA** tăng tốc embedding (`keepitreal/vietnamese-sbert`, 768 chiều), **Alembic** cho migration schema, và **Direct DB Lookup** cho tra cứu điều luật chính xác. Giao diện tiếng Việt, dark theme hiện đại, hỗ trợ streaming realtime.

> **API version:** `2.0.0` – FastAPI, app title: `Government AI Copilot API`
>
> **Architecture:** Đã thống nhất hoàn toàn lên v2 (PostgreSQL + Qdrant + Redis + async). Toàn bộ module v1 (SQLite, FAISS, langchain, sync) đã được xóa. Retrieval adapter pattern đảm bảo tương thích ngược.
>
> **Persona:** Cán bộ VHXH cấp xã – tự động phân biệt **Dạng A** (tra cứu pháp lý) và **Dạng B** (tình huống hành chính) để trả lời phù hợp.

### Cập nhật V3 (schema, retrieval, vận hành)

| Hạng mục | Mô tả |
|----------|--------|
| **Alembic** | Migration PostgreSQL: extension `unaccent` + `pg_trgm`; cột generated `articles.search_vector` (GIN); `documents.issuer` → TEXT; `documents.title` / `file_path` → TEXT. Chạy từ thư mục `backend`: `alembic upgrade head`. |
| **Keyword / FTS** | `keyword_retriever.keyword_search` ưu tiên FTS trên bảng `articles` (`ts_rank_cd`, `plainto_tsquery` + `unaccent`); nếu không có cột/migration hoặc lỗi → fallback ILIKE trên `vector_chunks`. |
| **Article selection** | `retrieval/article_selection.py`: `diversify_by_article()` — khi top-5 có ≥3 `document_id` khác nhau, sắp lại round-robin theo văn bản để tránh một nguồn chiếm hết thứ hạng; `dynamic_max_articles()` — từ khoảng cách điểm rerank + số doc khác nhau, chọn cap 1 / 3 / 5 nguồn. Được gọi trong `hybrid_retriever` sau bước rerank. |
| **Qdrant & embedding** | `QDRANT_RECREATE_ON_DIM_MISMATCH` (mặc định `true`): khi đổi model/chiều vector, có thể xóa và tạo lại collection — **cần** chạy `python scripts/reembed_all.py` để nạp lại vector. |
| **Makefile (repo root)** | `make test` — pytest `backend/tests/`; `make eval` — `tests/evaluation/`; `make all` — cả hai. |
| **Schema ORM** | `documents.issuer` lưu **trích yếu / chủ đề** do pipeline ingest trích (không chỉ “tên cơ quan”); có thêm quan hệ **Chương / Mục** (`chapters`, `sections`) trong `models.py`. |
| **Lưu file upload & Git** | Bản gốc `.doc`/`.docx` lưu dưới `backend/app/storage/<id>/`. Thư mục `backend/app/storage/` nằm trong **`.gitignore`** — không commit tài liệu upload; vector + metadata trong PostgreSQL và Qdrant là nguồn phục vụ RAG. |
| **Python dependencies** | File **`backend/requirements.txt`**: gồm **Alembic**, **asyncpg** / **psycopg2-binary**, **qdrant-client**, **redis[hiredis]**, **sentence-transformers**, **FlagEmbedding** (reranker `FlagReranker`), **torch**, **numpy** (&lt;2), **wikipedia**, **python-multipart** (upload API), **pytest** + **requests** (chạy test / e2e qua `make test`). |

---

## 📋 Tính năng

### 🏛️ AI Copilot – Trợ lý hành chính thực chiến cấp xã
- ✅ **Persona cán bộ VHXH cấp xã** — vai trò thực chiến, luật là căn cứ nhưng hướng dẫn hành động là mục tiêu
- ✅ **Tự động phân biệt Dạng A/B** — Dạng A (tra cứu pháp lý: trích dẫn điều luật) vs Dạng B (tình huống hành chính: 5 phần hành động)
- ✅ **Cấu trúc 5 phần bắt buộc** cho tình huống hành chính: Nhận định → Căn cứ pháp lý → Quy trình xử lý → Phối hợp liên ngành → Giải pháp lâu dài
- ✅ **Scenario Query Detection** — nhận diện tự động câu hỏi tình huống ("Ông/bà hãy...", "tham mưu", "trên địa bàn"...) để route đúng pipeline
- ✅ **7 intent cấp xã** — `xu_ly_vi_pham_hanh_chinh`, `kiem_tra_thanh_tra`, `thu_tuc_hanh_chinh`, `hoa_giai_van_dong`, `bao_ve_xa_hoi`, `to_chuc_su_kien_cong`, `bao_ton_phat_trien`
- ✅ **Nhận diện ý định tự động** (22+ intent) — rule-based regex + LLM fallback + scenario detection
- ✅ **RAG Intent Classifier** — tự nhận diện loại câu hỏi bằng **embedding + ngân hàng câu ví dụ** (scenario / legal_lookup / multi_article / need_expansion); luồng chính `intent_detector.get_rag_intents`; biến `USE_INTENT_CLASSIFIER` giữ tương thích `.env` cũ
- ✅ **Tra cứu văn bản pháp luật** — RAG pipeline nâng cao với citation bắt buộc
- ✅ **Direct Article Lookup** (3 mode) — (1) Điều/Khoản cụ thể, (2) Số hiệu văn bản, (3) Chủ đề trong luật — truy vấn trực tiếp PostgreSQL
- ✅ **Tra cứu metadata văn bản** — intent `document_metadata`, truy vấn trực tiếp PostgreSQL, không qua vector search
- ✅ **Mục tiêu chương trình/kế hoạch** — intent `program_goal`, prompt chuyên biệt
- ✅ **Quan hệ sửa đổi giữa văn bản** — intent `document_relation`, phân tích liên kết văn bản
- ✅ **Căn cứ pháp lý** — intent `can_cu_phap_ly`, phân tích cấp bậc văn bản
- ✅ **Giải thích quy định** — intent `giai_thich_quy_dinh`, giải thích ngôn ngữ đơn giản + ví dụ
- ✅ **Hướng dẫn thủ tục hành chính** — knowledge base 6 thủ tục + RAG bổ sung
- ✅ **Kiểm tra hồ sơ** — parse ngôn ngữ tự nhiên, phát hiện giấy tờ thiếu
- ✅ **Tóm tắt văn bản pháp luật** — 2 chế độ: (1) danh sách điều luật từ DB (precision), (2) LLM tóm tắt (fallback)
- ✅ **So sánh văn bản** — dual RAG + phân tích giống/khác/thay đổi
- ✅ **Soạn thảo văn bản** — soạn công văn, tờ trình theo thể thức hành chính chuẩn
- ✅ **Tạo báo cáo hành chính** — báo cáo có cấu trúc chuẩn hành chính
- ✅ **Agent Tools** — 7 tools cho LLM function calling (OpenAI format)
- ✅ **Ninh Bình Web Search** — Tra cứu thông tin tỉnh Ninh Bình qua Wikipedia + OpenAI web_search, với **admin scenario guard** (không route nhầm câu hỏi hành chính sang web search)
- ✅ **Conversation Memory** — lưu lịch sử hội thoại, hỗ trợ multi-turn, context follow-up

### 💬 Chat & RAG
- ✅ **Copilot Agent Pipeline** — Intent Detection → Scenario Detection → Context Resolution → Smart Routing → Tool/RAG → LLM
- ✅ **Enhanced Legal RAG Pipeline** — Query Rewriting → Query Expansion → Hybrid Search (Qdrant + PostgreSQL) → RRF Merge → Rerank → Top 5 → Citation
- ✅ **Dual RAG Versions** — `rag_chain.py` (enhanced: query rewrite + expand + validate) + `rag_chain_v2.py` (commune officer pipeline + fallback full article retrieval + Redis cache + PostgreSQL logging)
- ✅ **Hybrid Retrieval** — Qdrant semantic search + PostgreSQL (FTS `articles` + ILIKE fallback) → RRF merge → **FlagReranker** (`bge-reranker-v2-m3`) / fallback **CrossEncoder** (`bge-reranker-base`) → diversify theo document + dynamic max articles khi phù hợp
- ✅ **Multi-article context** — với câu hỏi "điều kiện", "quy định về", "phát triển thể thao thành tích cao"... dùng `single_article_only=False` + `max_articles` (MULTI_ARTICLE_MAX_ARTICLES), nhóm theo article + format có **Số hiệu**, **Hiệu lực thi hành**
- ✅ **Hiệu lực thi hành trong ngữ cảnh** — enrichment `effective_date` / `issued_date` từ bảng `documents`; header ngữ cảnh và prompt yêu cầu LLM ghi rõ hiệu lực từng văn bản
- ✅ **Domain post-filter** — khi lọc theo `legal_domains` chỉ giữ passage có `legal_domain` khớp (bỏ passage không gắn domain), tránh trích dẫn sai văn bản (vd. NĐ 103/2017 khi hỏi thể thao)
- ✅ **Auto doc_number extraction** — tự trích số hiệu văn bản từ query (VD: "nghị định 38/2021/NĐ-CP" → filter chính xác)
- ✅ **Doc reference resolution** — xử lý mismatch diacritics (NĐ-CP↔ND-CP), slash↔underscore, trailing ID; tra DB lấy doc_number chính xác
- ✅ **Direct Article Lookup (3 mode):**
  - **Mode 1 — Specific Article/Clause:** "Điều 9 Luật Di sản văn hóa" → truy vấn bảng `articles` + `clauses`
  - **Mode 2 — Doc Number Reference:** "Tóm tắt 49/2025/QĐ-UBND", "Chỉ thị 06/CT-UBND" → tìm document → lấy toàn bộ articles
  - **Mode 3 — Topic in Named Law:** "Hành vi bị nghiêm cấm trong Luật Di sản" → tìm document → search article title/content
- ✅ **Dual response format:** Dạng A (trích dẫn điều luật + giải thích thực tiễn) vs Dạng B (5 phần hành động)
- ✅ **Anti-hallucination (multi-layer)** — (1) SYSTEM_PROMPT cấm bịa điều luật, (2) post-generation strip doc numbers không có trong context, (3) diacritics-aware comparison (NĐ↔ND)
- ✅ **Fallback reasoning** — khi không tìm thấy tài liệu, LLM suy luận và đề xuất văn bản có thể liên quan
- ✅ **Fallback full article retrieval** — khi chunk ban đầu không đủ, tự động lấy ALL chunks của article từ PostgreSQL
- ✅ **Answer validation** — kiểm tra grounding trước khi trả lời (`ANSWER_VALIDATION_THRESHOLD`, mặc định ~0.40 trong `config.py`), bao gồm legal keyword detection, article reference extraction
- ✅ **Three-tier legal chunking** — chunk theo Điều (`chunk_type=article`), theo Khoản (`chunk_type=clause`), và sub-chunk 512 token (`chunk_type=token_sub`)
- ✅ **Legal Metadata Enrichment** — mỗi chunk có: `law_name`, `article_number`, `article_title`, `document_type`, `chunk_type`
- ✅ **Streaming SSE** — streaming smart routing theo intent, phát token realtime
- ✅ **Context Reference Detection** — nhận diện câu hỏi tham chiếu ngữ cảnh ("văn bản trên", "kế hoạch vừa soạn", "cái đó") qua regex patterns
- ✅ **Follow-up Question Resolution** — tự động bổ sung ngữ cảnh tài liệu (loại văn bản, lĩnh vực, cơ quan, chủ đề) cho câu hỏi nối tiếp
- ✅ **Document Context Memory** — lưu metadata văn bản đã soạn/tham chiếu, giữ lịch sử 5 tài liệu gần nhất trong conversation
- ✅ **Domain guard** — lọc câu hỏi ngoài phạm vi tài liệu + nhận diện follow-up patterns ("văn bản đó", "điều X", "khoản Y")
- ✅ **Ninh Bình routing guard** — admin scenario detection ngăn route nhầm câu hỏi hành chính (di tích, trùng tu, tôn giáo...) sang web search
- ✅ **Redis Caching** — SHA-256 query key, TTL 1h, graceful degradation
- ✅ **PostgreSQL Interaction Logging** — tự động ghi log query, answer, sources, latency vào `chat_logs`

### 📄 Quản lý tài liệu
- ✅ Upload file Word (`.docx`, `.doc`) làm nguồn tri thức
- ✅ **Upload folder** — hỗ trợ `webkitdirectory` cho upload thư mục, tự strip path prefix
- ✅ Hỗ trợ `.doc` (binary) qua Microsoft Word COM (Windows)
- ✅ **Auto-extract doc_number** từ tên file và nội dung — regex cho cả filename tiếng Việt và text body
- ✅ **Safe filename handling** — strip subdirectory paths, truncate filename >150 chars, tránh lỗi `MAX_PATH` trên Windows
- ✅ **Pipeline logging** — log chi tiết 9 stage của ingestion pipeline (parse → clean → detect → chunk → embed → store)
- ✅ **Migration script** — `scripts/fix_doc_numbers.py` sửa doc_number cũ trong cả PostgreSQL + Qdrant
- ✅ Kéo thả hoặc click để chọn file (Drag & Drop)
- ✅ Quản lý nhiều tài liệu: xem danh sách, xóa có xác nhận
- ✅ **Re-index** toàn bộ dataset sau khi thay đổi chiến lược chunking/retrieval

### 🖥️ Giao diện
- ✅ **Dark theme** hiện đại, custom color palette
- ✅ UI hoàn toàn **tiếng Việt**
- ✅ **SourcesPanel** hiển thị: số hiệu văn bản, tên luật, số Điều, tên Điều, cơ quan ban hành, độ liên quan
- ✅ **Typing indicator** (3 chấm nhấp nháy), animation slide-up
- ✅ Render **Markdown** (code block, blockquote, danh sách, inline code)
- ✅ Tùy chỉnh **Temperature** (0.0–1.0) với nhãn Chính xác / Cân bằng / Sáng tạo

### ⚡ Hiệu năng & GPU
- ✅ **GPU CUDA** cho embedding (tự động fallback CPU)
- ✅ **Badge GPU** — tên device, VRAM sử dụng
- ✅ **Batch embedding** — batch_size=32 (cấu hình được)
- ✅ **Qdrant vector search** — tìm kiếm semantic cực nhanh, hỗ trợ scale ngang
- ✅ **Redis cache** — cache result TTL 1h, giảm tải LLM + retrieval
- ✅ **PostgreSQL connection pool** — pool_size=20, max_overflow=10

---

## 🏗️ Kiến trúc

### Tổng quan (6-layer)

```
┌──────────────────────────────────────────────────────────────────────┐
│  API Layer         POST /api/chat  /chat/stream  /upload             │
│                    GET  /api/documents  /documents/{id}  /health     │
├──────────────────────────────────────────────────────────────────────┤
│  Agent Layer       Copilot Agent – Intent Detection + Routing        │
├──────────────────────────────────────────────────────────────────────┤
│  Cache Layer       Redis (SHA-256 query key → JSON, TTL 1h)          │
├──────────────────────────────────────────────────────────────────────┤
│  RAG Layer         Query Rewrite/Expand → Qdrant + PostgreSQL        │
│                    → RRF Merge → FlagReranker (bge-reranker-v2-m3) → diversify / article cap → Top-K    │
├──────────────────────────────────────────────────────────────────────┤
│  Storage Layer     PostgreSQL (documents, articles, clauses,         │
│                    vector_chunks, chat_logs) + Qdrant (law_documents)│
├──────────────────────────────────────────────────────────────────────┤
│  LLM Layer         OpenAI API (streaming + non-streaming)            │
└──────────────────────────────────────────────────────────────────────┘
```

### Luồng xử lý chính

```
User Message
  → /api/chat hoặc /api/chat/stream (entrypoint)
  → Redis Cache check (hit → return ngay)
  → Copilot Agent (process / process_stream)
      → 1. Conversation Memory: lưu user message (nếu có conversation_id)
      → 2. Intent Detection (rule-based regex → LLM fallback nếu confidence < 0.5)
      → 3. Domain Guard: câu hỏi ngoài phạm vi → trả lời từ chối
      → 4. Smart Routing (rag_chain_v2.rag_query):
          ├── Step 0: Ninh Bình pre-check
          │     ├── LEGAL_KEYWORDS? → skip (→ RAG)
          │     ├── ADMIN_SCENARIO? → skip (→ commune pipeline)  ★ NEW
          │     └── NINH_BINH/GEO keyword → search_ninh_binh_info
          │
          ├── Step 1: Commune Officer Pipeline  ★ NEW
          │     ├── intent ∈ COMMUNE_LEVEL_INTENTS (7 intent)?
          │     ├── commune_situation có vi phạm?
          │     ├── _is_scenario_query() match? ("Ông/bà hãy...", "tham mưu"...)
          │     └── → _answer_commune_officer_query()
          │           → COMMUNE_OFFICER_SYSTEM_PROMPT (5 phần hành động)
          │
          ├── Step 2: Specialized Intent Routing
          │     ├── checklist_documents → _answer_checklist_query()
          │     ├── document_drafting  → _answer_drafting_query()
          │     ├── document_summary   → list_document_articles()
          │     └── SPECIALIZED_INTENTS → route_query() (12 handlers)
          │
          └── Step 3: Default RAG Pipeline
                ├── Direct Article Lookup (3 mode)  ★ NEW
                │     ├── Mode 1: "Điều X Luật Y" → articles + clauses DB
                │     ├── Mode 2: "49/2025/QĐ-UBND" → doc_number → articles
                │     └── Mode 3: "hành vi cấm trong Luật Di sản" → topic search
                ├── Hybrid Search (Qdrant + PostgreSQL) → RRF → Rerank
                └── SYSTEM_PROMPT_V2 (auto Dạng A/B)  ★ NEW
      → 5. Conversation Memory: lưu assistant response
  → Cache result in Redis (TTL 1h)
  → Log to PostgreSQL chat_logs
  → Response {answer, sources, confidence, intent, query_analysis}
```

### Copilot Agent – Chi tiết routing

#### Luồng Non-streaming (`process`)

```
copilot_agent.process(question, temperature, filters, conversation_id)
  │
  ├─ conversation_store.add_message() (lưu user message)
  │
  ├─ detect_intent() → {intent, confidence}
  │     └── rule-based regex → LLM fallback nếu confidence < 0.5
  │
  ├─ _should_use_ninh_binh_tool()? Có (từ khóa Ninh Bình, Tràng An, ... VÀ không pháp luật)
  │     └── ninh_binh_run() → trả answer từ backend/data/ninh_binh_knowledge.json
  │
  ├─ is_in_document_domain()? Không → OUT_OF_DOMAIN_MESSAGE
  │
  ├─ intent in SPECIALIZED_INTENTS && confidence ≥ 0.5?
  │     └── route_query(intent_override) → ROUTE_MAP[intent] (11 handlers):
  │           ├── tra_cuu_van_ban      → rag_query_enhanced()
  │           ├── huong_dan_thu_tuc    → procedure KB + rag_query_enhanced()
  │           ├── kiem_tra_ho_so       → check_documents_from_query()
  │           ├── tom_tat_van_ban      → document_summarizer + RAG
  │           ├── so_sanh_van_ban      → document_comparator (dual RAG)
  │           ├── soan_thao_van_ban    → draft_tool.run() → save document metadata
  │           ├── tao_bao_cao          → report_generator + RAG
  │           ├── trich_xuat_van_ban   → extract_tool
  │           ├── can_cu_phap_ly       → rag_query_enhanced(top_k=10)
  │           ├── giai_thich_quy_dinh  → rag_query_enhanced(top_k=8)
  │           ├── ninh_binh_info       → ninh_binh_search_tool.run()
  │           └── hoi_dap_chung        → rag_query_enhanced() (fallback)
  │
  └─ else → _resolve_follow_up_question() (context resolution)
              ├── _is_context_reference()? → bổ sung ngữ cảnh tài liệu từ conversation_store
              ├── looks_like_follow_up()? → bổ sung last_topic
              └── rag_query_enhanced() → _generate_grounded_answer(intent=...)
                    ├── document_metadata → _generate_metadata_answer() (PostgreSQL)
                    ├── document_relation → _generate_relation_answer()
                    ├── program_goal      → _generate_program_goal_answer()
                    ├── article_query     → RAG + citation (default)
                    └── tra_cuu_van_ban   → RAG + citation (default)
```

#### Luồng Streaming SSE (`process_stream`)

```
yield {type:"intent", data:{intent, confidence}}   ← Token #1
  ↓
yield {type:"sources", data:[...]}                 ← Token #2
  ↓
yield answer_text (tokens)                         ← Token #3..N
  ↓
(nếu intent=soan_thao_van_ban → save document metadata vào conversation_store)
```

Frontend parse:
- `{type:"intent"}` → bỏ qua, không hiển thị
- `{type:"sources"}` → gọi `onSources()` → hiển thị SourcesPanel
- Text tokens → ghép vào chat bubble realtime

### RAG Pipeline nâng cao

```
Original Query
  → Query Analysis     (intent / keywords / metadata filters)
  → Query Rewriting    (QUERY_REWRITE_PROMPT)
  → Query Expansion    (sinh thêm 2-3 query variants)
  → Hybrid Search:
      ├── Qdrant vector search (top_k ≈ RETRIEVAL_TOP_K)
      └── PostgreSQL keyword: FTS `articles` hoặc ILIKE `vector_chunks` (top_k ≈ RETRIEVAL_TOP_K)
  → RRF Merge          (Reciprocal Rank Fusion, k=60)
  → Rerank (FlagReranker RERANKER_MODEL, fallback CrossEncoder FALLBACK)
  → diversify_by_article + dynamic_max_articles (trong hybrid path tương ứng)
  → Answer Extraction  (filter chunks theo keyword overlap)
  → Top-K passages / articles (theo cấu hình)
  → Fallback Full Article Retrieval (nếu chunk không đủ → lấy ALL chunks của article từ PostgreSQL)
  → _generate_grounded_answer(intent)
  → Answer Validation  (cosine sim vs ANSWER_VALIDATION_THRESHOLD, fallback nếu thấp hơn)
  → Redis Cache        (lưu kết quả, TTL 1h)
  → PostgreSQL Logging (ghi query, answer, sources, latency vào chat_logs)
  → _append_citations  ("Nguồn:\n- [1] Tên luật, Điều X")
  → LLM Response
```

### Luồng RAG v2 (rag_chain_v2) – Top-K, Rerank, đa article & hiệu lực

Luồng thực tế khi trả lời câu hỏi pháp lý (entrypoint `rag_chain_v2.rag_query`). **Đã cập nhật:** intent classifier, ngữ cảnh đa article cho câu hỏi điều kiện/quy định/phát triển thể thao, hiệu lực thi hành trong context, domain post-filter.

#### 1. Luồng hoạt động (chat flow) từ query đến answer

```
User query (VD: "Điều kiện kinh doanh thể thao" hoặc "Phát triển thể thao thành tích cao")
  │
  ├─ Step 0: Ninh Bình tool check → nếu không dùng → tiếp
  ├─ analyze_query() → intent, commune_situation
  ├─ Redis cache check (hit → return ngay)
  │
  ├─ get_domain_filter_values(query) → legal_domains (lọc Qdrant + post-filter)
  ├─ _get_rag_intent_flags(query) → rag_intents (tự nhận diện hoặc regex):
  │     • USE_INTENT_CLASSIFIER=true → get_rag_intents() (embedding + INTENT_PROTOTYPES)
  │     • Fallback: _is_scenario_query(), _should_use_multi_article_context(), needs_expansion()
  │     • Output: is_scenario, is_legal_lookup, use_multi_article, needs_expansion
  │
  ├─ Branch theo intent:
  │     ├─ is_commune_query? (intent ∈ COMMUNE_LEVEL_INTENTS hoặc rag_intents["is_scenario"])
  │     │     → _answer_commune_officer_query()
  │     ├─ checklist_documents / document_drafting / document_summary → handler tương ứng
  │     └─ Default RAG path (phần dưới)
  │
  └─ Default RAG path:
        ├─ use_multi = rag_intents["needs_expansion"]
        ├─ multi_article_conditions = rag_intents["use_multi_article"]
        ├─ use_multi? → _multi_query_retrieve(..., force_expansion=use_multi)
        ├─ else → hybrid_search(..., single_article_only=not multi_article_conditions,
        │                       max_articles=MULTI_ARTICLE_MAX_ARTICLES nếu multi_article_conditions)
        │
        ├─ hybrid_search:
        │     ├─ lookup_article_from_db() / _resolve_doc_number() / _direct_article_lookup()
        │     ├─ vector_search + keyword_search (FTS `articles` hoặc ILIKE; legal_domains filter) → RRF → rerank
        │     ├─ diversify_by_article() + dynamic_max_articles() (khi đủ điều kiện — `hybrid_retriever`)
        │     ├─ _enrich_missing_metadata() (doc_number, document_title, effective_date, issued_date)
        │     ├─ Post-filter: chỉ giữ passage có legal_domain ∈ legal_domains (tránh trích sai văn bản)
        │     └─ single_article_only?
        │           • True, max_articles=1: _select_best_article() → 1 article, _fetch_full_article_chunks
        │           • True, max_articles>1: _select_top_n_articles() → N article, expand từng article
        │           • False: reranked[:final_k] (nhiều passage, nhiều article)
        │
        ├─ Build context:
        │     ├─ use_multi hoặc multi_article_conditions hoặc _db_lookup
        │     │     → group_chunks_by_article + format_grouped_context (header có Số hiệu, Hiệu lực thi hành)
        │     └─ else → _select_single_article_passages() + _build_context()
        │
        ├─ RAG_PROMPT_TEMPLATE_V2 (yêu cầu: mỗi văn bản ghi Số hiệu, Hiệu lực thi hành, trích đủ điều)
        ├─ generate() → answer
        └─ Post-process: sanitize, anti-hallucination, citations, cache, log
```

#### 2. Số lượng top-k đưa vào ngữ cảnh

| Tham số | Giá trị mặc định | Ý nghĩa |
|--------|-------------------|--------|
| **RETRIEVAL_TOP_K** | 40 | Số passage lấy từ **mỗi** nguồn (vector search + keyword search) trước khi merge. |
| **RERANK_TOP_K** | 10 | `final_k`: số passage mục tiêu sau rerank (dùng trong hybrid_search khi gọi `top_k`). |
| **rerank(..., top_k)** | max(final_k*4, 12) = **40** | Số passage **giữ lại sau rerank** (để nhóm theo article rồi chọn 1 article). |
| **Ngữ cảnh thực tế gửi LLM** | **1 hoặc N article** | Khi `single_article_only=True` và `max_articles=1`: 1 điều luật. Khi `multi_article_conditions` hoặc `max_articles>1`: nhiều điều luật (tối đa MULTI_ARTICLE_MAX_ARTICLES), mỗi nguồn có Số hiệu + Hiệu lực thi hành trong header. |

- **Vector search:** `top_k=RETRIEVAL_TOP_K` (40).  
- **Keyword search:** `top_k=RETRIEVAL_TOP_K` (40).  
- **Sau RRF:** danh sách merged (đã dedup).  
- **Sau rerank:** giữ `top_k=40` passage.  
- **Post-filter domain:** chỉ giữ passage có `legal_domain` khớp (khi có `legal_domains`).  
- **Article selection:** 1 article (mặc định) hoặc top N article (khi dùng multi-article); ngữ cảnh có thể chứa **nhiều văn bản** khi query thuộc nhóm điều kiện/quy định/phát triển thể thao.

#### 3. Cách hoạt động của Rerank (rerank top_k)

- **Module:** `app/retrieval/reranker.py` — ưu tiên **FlagReranker** (`BAAI/bge-reranker-v2-m3`); fallback **CrossEncoder** (`BAAI/bge-reranker-base`).  
- **Input:** `query` (câu hỏi) và `candidates` (danh sách passage, mỗi phần tử có `text_chunk`).  
- **Cách hoạt động:**  
  - Với mỗi cặp (query, passage), reranker (FlagReranker hoặc CrossEncoder) cho một **điểm relevance** (rerank_score).  
  - Sắp xếp toàn bộ candidate theo `rerank_score` giảm dần.  
  - Cắt lấy **top_k** passage đầu tiên (`candidates[:top_k]`).  
- **Trong hybrid_retriever:**  
  - Gọi `rerank(query, candidates=merged, top_k=max(final_k*4, 12))` với `final_k = RERANK_TOP_K` (10) → **top_k = 40**.  
  - Sau rerank, mỗi passage có thêm trường `rerank_score`.  
  - Tiếp theo: nhóm theo article, tính điểm nhóm (semantic + title_sim + overlap + bonus), chọn **một** article có điểm cao nhất rồi `_fetch_full_article_chunks` cho article đó.  

Rerank không tự so sánh “nhiều điều luật”: nó chỉ xếp hạng **từng passage** theo relevance với query. Việc **chỉ giữ một article** do logic **single_article_only** và **\_select_single_article_passages** quyết định, không phải do rerank.

#### 4. Khi nào ngữ cảnh có một vs nhiều kết quả?

- **Một article (hành vi mặc định):** Khi query không trigger multi-article (không có "điều kiện", "quy định về", "phát triển thể thao thành tích cao"...), `hybrid_search(single_article_only=True)` chọn một article tốt nhất và `_select_single_article_passages()` giữ một nhóm → ngữ cảnh 1 điều luật.  
- **Nhiều article (đã triển khai):** Khi `_should_use_multi_article_context(query)` hoặc intent classifier trả `use_multi_article=True`, gọi `hybrid_search(single_article_only=False, max_articles=MULTI_ARTICLE_MAX_ARTICLES)` và **không** gọi `_select_single_article_passages`; dùng `group_chunks_by_article` + `format_grouped_context` (header có Số hiệu, Hiệu lực thi hành). LLM nhận nhiều văn bản/điều để so sánh và trích dẫn đầy đủ.

#### 5. Giảm trả lời sai điều luật (đã áp dụng)

- **Đa article cho điều kiện/quy định:** Query "điều kiện kinh doanh thể thao", "phát triển thể thao thành tích cao", "quy định về..." dùng multi-article path → context nhiều văn bản (Luật 2006/2018, NĐ 36/112...), LLM so sánh và trích dẫn từng nguồn. “khớp” query (VD Điều 50 NĐ 36/2019) có thể đủ để **cả article** đó thắng, dù các điều trong Luật 2006/2018 cũng liên quan nhưng điểm từng chunk thấp hơn hoặc bị “chia” across nhiều article nên tổng điểm nhóm thua.  
- **Domain post-filter:** Chỉ giữ passage có `legal_domain` khớp với domain của query (vd. thể thao → the_thao), tránh lẫn văn bản khác lĩnh vực. (Trước: Rerank ưu tiên passage, không ưu tiên “phủ nhiều văn bản”:** Rerank chỉ xếp hạng passage theo relevance. Một passage (VD một khoản của Điều 50) điểm cao có thể kéo theo cả article Điều 50 được chọn, dù câu trả lời “đầy đủ” hơn có thể cần cả Luật 2006, 2018 và NĐ 112/2007.  
- **Intent classifier:** Nhận diện chính xác hơn nhờ embedding + ngân hàng câu ví dụ. (Trước: Không có bước so sánh đa nguồn: Do ngữ cảnh chỉ 1 article, LLM không có thông tin từ các điều luật khác để so sánh hoặc bổ sung; có thể dẫn đến trả lời thiên lệch hoặc “đúng một phần” (đúng Điều 50 nhưng thiếu điều kiện từ Luật khác).  
- **Prompt:** Yêu cầu ghi rõ Tên văn bản, Số hiệu, Hiệu lực thi hành và trích đủ từng điều khi có nhiều nguồn.

---

#### 6. Nguyên nhân (tổng hợp)

| Nguyên nhân | Mô tả ngắn |
|-------------|------------|
| **Thiết kế single-article** | Pipeline mặc định chọn **một** article (một điều luật) duy nhất sau rerank và sau `_select_single_article_passages`, nên ngữ cảnh gửi LLM chỉ chứa một nguồn. |
| **Chọn article theo điểm nhóm** | Article được chọn theo tổng điểm (semantic + title_sim + overlap + bonus). Một passage rất khớp query có thể kéo cả article đó thắng, bỏ qua các điều luật/văn bản khác cũng liên quan. |
| **Rerank không đa nguồn** | Rerank chỉ xếp hạng từng passage, không có bước “ưu tiên phủ nhiều văn bản” hay “giữ top-k article khác nhau”, nên dễ dẫn đến một article chiếm toàn bộ context. |
| **Không so sánh trong prompt** | Do context chỉ 1 article, LLM không có dữ liệu để so sánh/tổng hợp giữa Luật 2006, 2018, NĐ 36/2019, NĐ 112/2007... dù retrieval đã lấy được nhiều passage từ các văn bản đó. |
| **Trigger expansion hẹp** | Chỉ khi `needs_expansion(query)=True` (từ khóa “các”, “liệt kê”, “điều kiện... kinh doanh”...) mới dùng multi-query và có thể giữ nhiều article; câu hỏi dạng “điều kiện kinh doanh thể thao” có thể không trigger → vẫn single-article. |
| **Trích dẫn sai điều luật** | Hệ quả của việc chọn một article: nếu article được chọn không phải điều “đúng nhất” theo người dùng (vd. thiên về NĐ 36 Điều 50, thiếu Luật 2006/2018), bot vẫn trích dẫn duy nhất điều đó và không đề cập nguồn khác. |

---

#### 7. Đã triển khai (cập nhật)

Các hướng sau đã được áp dụng trong codebase:

| Hướng | Mô tả | File / vùng liên quan |
|-------|--------|-------------------------|
| **1. Mở rộng ngữ cảnh đa article cho câu hỏi “điều kiện / quy định”** | Với query có từ khóa “điều kiện”, “quy định về”, “các văn bản”… gọi `hybrid_search(..., single_article_only=False)` và **không** gọi `_select_single_article_passages`, để giữ nhiều passage thuộc nhiều điều luật/văn bản → context có nhiều nguồn, LLM có thể so sánh/tổng hợp. | `rag_chain_v2.py`: nhánh default RAG, trước khi build context. |
| **2. Tăng số article tối đa trong context (top-articles)** | Thay vì chọn 1 article, giữ **top N article** (vd. N=3–5) sau rerank: nhóm theo article, xếp điểm nhóm, lấy N nhóm đầu, lấy đủ chunk của N article đó (có thể cap tổng số chunk để không vượt token). | `hybrid_retriever.py`: bước article selection; có thể thêm tham số `max_articles=1|3|5`. |
| **3. Mở rộng trigger query expansion** | Bổ sung trigger cho “điều kiện kinh doanh”, “quy định về”, “điều kiện hoạt động”… vào `needs_expansion()` để nhiều câu hỏi dạng này được multi-query + nhiều passage → tăng cơ hội có đa văn bản trong context. | `query_expansion.py`: `_EXPANSION_TRIGGERS`. |
| **4. Rerank / post-rerank ưu tiên đa nguồn** | Sau rerank: (a) giữ top-k passage như hiện tại, hoặc (b) thêm bước “diversify by document/article”: ưu tiên lấy passage từ nhiều (document_id, article_id) khác nhau để context không chỉ từ một điều luật. | `hybrid_retriever.py` sau bước rerank; hoặc thêm hàm diversify-by-article. |
| **5. Prompt yêu cầu so sánh khi có nhiều nguồn** | Khi context chứa nhiều điều luật/văn bản, prompt yêu cầu LLM: nêu rõ từng nguồn (Luật X, NĐ Y, Điều Z), so sánh/đối chiếu nếu có, và trích dẫn đúng từng văn bản. | `config.py`: RAG_PROMPT_TEMPLATE_V2 / SYSTEM_PROMPT_V2. |
| **6. Cấu hình hóa single vs multi-article** | Thêm cấu hình (vd. env hoặc config) để bật/tắt chế độ “chỉ 1 article” vs “nhiều article” (hoặc “tối đa N article”), dễ A/B test và triển khai từng bước. | `config.py`; `rag_chain_v2.py` và `hybrid_retriever.py` đọc config. |

**Trạng thái:** Các mục 1–6 đã triển khai; thêm **RAG Intent Classifier** (mục 7) trong `intent_classifier.py` / `get_rag_intents` — tự nhận diện loại câu hỏi bằng embedding + ngân hàng câu ví dụ.

---

## 🗂️ Cấu trúc dự án

```
rag_chatbot/
├── README.md
├── Makefile                             # make test | make eval | make all (pytest từ thư mục backend)
├── docker-compose.yml                   # PostgreSQL 16 + Qdrant + Redis
├── backend/
│   ├── main.py                          # FastAPI entry point (v2.0.0), lifespan: PostgreSQL → Qdrant → Embedding → Reranker
│   ├── requirements.txt                 # Dependencies (PostgreSQL, Qdrant, Redis, sentence-transformers, ...)
│   ├── alembic.ini                      # Alembic config
│   ├── alembic/                         # Migration PostgreSQL (V3: FTS, TEXT columns, …)
│   ├── .env                             # Cấu hình môi trường
│   ├── scripts/
│   │   ├── fix_doc_numbers.py           # Sửa doc_number cũ (PostgreSQL + Qdrant)
│   │   └── reembed_all.py               # Re-embed toàn bộ vector_chunks → Qdrant (sau đổi model/chiều vector)
│   └── app/
│       ├── config.py                    # Tất cả cấu hình (.env), prompt templates
│       ├── database/                    # PostgreSQL layer (async SQLAlchemy)
│       │   ├── models.py               # ORM: documents, articles, clauses, vector_chunks, chat_logs
│       │   ├── session.py              # Async engine, session factory, init_db()
│       │   └── schema.sql              # Raw SQL reference
│       ├── parser/                      # DOCX parser
│       │   └── docx_parser.py          # Trích xuất text từ .docx
│       ├── pipeline/                    # Document ingestion pipeline
│       │   ├── cleaner.py              # Unicode normalize, loại artifact
│       │   ├── structure_detector.py   # Regex detect Điều/Khoản/Điểm/Chương/Mục
│       │   ├── chunker.py             # 300-500 token chunks với overlap
│       │   ├── legal_chunker.py        # ★ Three-tier: article + clause + 512-token sub-chunks
│       │   ├── db_writer.py            # Insert chunks vào PostgreSQL (+ chunk_type)
│       │   ├── embedding.py            # Sentence-transformer batch encode (GPU/CPU)
│       │   ├── vector_store.py         # Qdrant client (upsert, search, delete)
│       │   └── ingestor.py             # Pipeline orchestrator (9-stage logging)
│       ├── retrieval/                   # Hybrid retrieval system
│       │   ├── vector_retriever.py     # Qdrant semantic search
│       │   ├── keyword_retriever.py    # FTS articles (unaccent+tsvector) + ILIKE fallback trên chunks
│       │   ├── hybrid_retriever.py     # RRF + rerank + diversify_by_article + dynamic_max_articles + direct lookup
│       │   ├── article_lookup.py       # ★ Direct DB lookup (3 modes: article, doc_number, topic)
│       │   ├── article_selection.py    # ★ Đa dạng hóa passage theo document + cap số nguồn động
│       │   └── reranker.py             # FlagReranker (v2-m3) + fallback CrossEncoder (base)
│       ├── cache/                       # Redis caching
│       │   └── redis_cache.py          # SHA-256 key, TTL 1h, graceful degradation
│       ├── monitoring/                  # Interaction logging
│       │   └── chat_logger.py          # PostgreSQL chat_logs audit trail
│       ├── models/
│       │   ├── schemas.py              # Pydantic schemas (legacy routers)
│       │   └── schemas_v2.py           # Pydantic v2 schemas (chat, upload, documents)
│       ├── agents/
│       │   └── copilot_agent.py        # ★ Brain: Intent → Context Resolution → Follow-up → Routing → Response
│       ├── tools/
│       │   ├── summarize_tool.py       # Tool tóm tắt văn bản (delegates to document_summarizer)
│       │   ├── extract_tool.py         # Tool trích xuất thông tin cấu trúc (9 fields)
│       │   ├── draft_tool.py           # Tool soạn thảo văn bản hành chính (8 loại VB, anti-hallucination refs)
│       │   ├── classify_tool.py        # Tool phân loại văn bản (loại, lĩnh vực, cấp, trạng thái)
│       │   ├── ninh_binh_search_tool.py # Tool tra cứu thông tin tỉnh Ninh Bình (web search)
│       │   └── openai_web_search_tool.py # OpenAI web_search API wrapper
│       ├── memory/
│       │   └── conversation_store.py   # In-memory conversation store (thread-safe, LRU) + document context tracking
│       ├── routers/
│       │   ├── chat_router.py          # POST /api/chat, /api/chat/stream (v2 hybrid retrieval)
│       │   ├── document_router_v2.py   # POST /api/upload, GET /api/documents
│       │   ├── health_router.py        # GET /api/health (PostgreSQL + Qdrant + Redis)
│       │   ├── copilot_router.py       # POST /api/copilot/chat, /chat/stream
│       │   ├── conversation_router.py  # GET/POST/DELETE /api/conversations
│       │   ├── document_router.py      # POST /api/document/summarize, /compare, /api/report/generate
│       │   ├── intent_router.py        # POST /api/intent
│       │   ├── procedure_router.py     # POST /api/procedure/steps, /check; GET /list
│       │   ├── search_router.py        # POST /api/search (hybrid Qdrant + PostgreSQL)
│       │   └── tools_router.py         # POST /api/tools/summarize|extract|draft|classify
│       └── services/
│           ├── retrieval.py            # ★ Retrieval adapter: wraps v2 hybrid_retriever cho backward-compat
│           ├── rag_chain.py            # ★ RAG pipeline: rewrite → expand → hybrid search → validate → answer
│           ├── rag_chain_v2.py         # ★ RAG v2: commune officer pipeline + auto Dạng A/B + cache + logging
│           ├── intent_detector.py      # 22+ intent, rule-based + LLM fallback, priority system
│           ├── intent_classifier.py    # RAG intent: embedding + INTENT_PROTOTYPES → get_rag_intents()
│           ├── query_router.py         # Dispatch intent → specialized handlers (ROUTE_MAP)
│           ├── query_understanding.py  # Extract intent/filters/keywords + commune situation analysis
│           ├── answer_validator.py     # Kiểm tra grounding (cosine sim + legal keyword + article ref)
│           ├── domain_guard.py         # Lọc câu hỏi ngoài phạm vi + follow-up pattern detection
│           ├── llm_client.py           # AsyncOpenAI wrapper (stream + non-stream)
│           ├── procedure_service.py    # Knowledge base 6 thủ tục hành chính
│           ├── document_checker.py     # Kiểm tra hồ sơ thiếu/đủ
│           ├── document_summarizer.py  # Tóm tắt VB: list_document_articles (DB) + summarize (LLM fallback)
│           ├── ninh_binh_web_search.py # Wikipedia + OpenAI web_search cho Ninh Bình
│           ├── ninh_binh_router.py     # Router: Ninh Bình geo/info vs legal/admin RAG (+ admin scenario guard)
│           ├── ninh_binh_info_extractor.py  # Trích xuất thông tin Ninh Bình từ kết quả search
│           ├── ninh_binh_entity_resolver.py # Phân giải entity Ninh Bình (xã, huyện, địa danh)
│           ├── document_comparator.py  # So sánh 2 văn bản (dual RAG + LLM)
│           ├── agent_tools.py          # 7 tools định nghĩa theo OpenAI function calling format
│           ├── report_generator.py     # Tạo báo cáo hành chính
│           └── gpu_info.py             # CUDA/GPU status monitoring
├── frontend/
│   ├── package.json
│   ├── vite.config.js                   # Dev proxy /api → localhost:8000
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── App.jsx                      # Layout, shared state (dataset, temperature)
│       ├── main.jsx                     # createRoot + StrictMode
│       ├── index.css                    # Tailwind + animation (blink, slideUp, typing dots)
│       ├── api/
│       │   └── client.js                # fetch + SSE client (upload, chat, datasets, GPU)
│       ├── components/
│       │   ├── ChatMessage.jsx          # Markdown-rendered messages + citation footer
│       │   ├── ChatInput.jsx            # Auto-resize textarea, Enter/Shift+Enter
│       │   ├── Sidebar.jsx              # Dataset list/select/delete, temperature, collapse
│       │   ├── UploadModal.jsx          # Drag & Drop upload modal
│       │   ├── SourcesPanel.jsx         # Nguồn tham khảo: doc_number, law_name, Điều, issuing_body
│       │   ├── TemperatureSlider.jsx    # Slider 0.0–1.0 với nhãn ngữ nghĩa
│       │   ├── GpuBadge.jsx             # GPU/CPU badge + VRAM
│       │   └── TypingIndicator.jsx      # Animation 3 chấm
│       └── pages/
│           └── ChatPage.jsx             # Màn chat: message list, streaming, sources, auto-scroll
└── tests/
    ├── test_e2e.py                      # End-to-end test qua API
    ├── test_search.py                   # Test search endpoint
    ├── test_search2.py                  # Test search với filter
    └── test_search3.py                  # Test keyword search
```

---

## 🗄️ Database Schema (PostgreSQL)

Schema đầy đủ trong ORM: `documents` → `chapters` / `sections` (tuỳ cấu trúc văn bản) → `articles` → `clauses` → `vector_chunks`. Dưới đây là các bảng chính và cột quan trọng cho RAG.

### Bảng `documents` — metadata văn bản pháp luật

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `id` | SERIAL PK | Auto-increment |
| `doc_number` | VARCHAR(255) | Số hiệu văn bản (VD: `13/2025/TT-BVHTTDL`) — widened từ 100 |
| `title` | TEXT | Tiêu đề văn bản (lấy từ filename khi upload); cột TEXT (migration V3) |
| `document_type` | VARCHAR(50) | Loại: Luật, Nghị định, Thông tư, ... |
| `issuer` | TEXT | **Trích yếu / chủ đề** do pipeline ingest trích (vd. dòng “VỀ VIỆC …”), không chỉ tên cơ quan; kiểu TEXT (migration V3) |
| `issued_date` | DATE | Ngày ban hành |
| `effective_date` | DATE | Ngày hiệu lực |
| `file_path` | TEXT | Đường dẫn file gốc (TEXT đầy đủ, migration V3) |
| `created_at` | TIMESTAMP | Thời điểm tạo (NOW()) |

### Bảng `chapters` / `sections` — Chương / Mục (tuỳ chọn)

Phân cấp theo cấu trúc văn bản đã detect khi ingest; liên kết tới `articles` qua `chapter_id` / `section_id`. Chi tiết: [backend/app/database/models.py](backend/app/database/models.py).

### Bảng `articles` — tách theo Điều

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `id` | SERIAL PK | Auto-increment |
| `document_id` | INTEGER FK | → documents(id) CASCADE |
| `chapter_id` / `section_id` | INTEGER FK | Tuỳ cấu trúc văn bản |
| `article_number` | VARCHAR(20) | VD: `Điều 5` |
| `title` | TEXT | Tiêu đề Điều |
| `content` | TEXT NOT NULL | Nội dung Điều |
| `search_vector` | tsvector (generated) | **V3:** FTS trên `title` + `content` (unaccent), index GIN — phục vụ `keyword_retriever.full_text_search` |

### Bảng `clauses` — tách theo Khoản/Điểm

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `id` | SERIAL PK | Auto-increment |
| `article_id` | INTEGER FK | → articles(id) CASCADE |
| `clause_number` | VARCHAR(20) | VD: `Khoản 3`, `Điểm a` |
| `content` | TEXT NOT NULL | Nội dung Khoản/Điểm |

### Bảng `vector_chunks` — mapping chunk text ↔ Qdrant vector

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `id` | SERIAL PK | Auto-increment |
| `document_id` | INTEGER FK | → documents(id) CASCADE |
| `article_id` | INTEGER FK | → articles(id) SET NULL |
| `clause_id` | INTEGER FK | → clauses(id) SET NULL |
| `vector_id` | VARCHAR(64) UNIQUE | Qdrant point UUID |
| `chunk_text` | TEXT NOT NULL | Nội dung chunk |
| `chunk_type` | VARCHAR(20) | `article` / `clause` / `token_sub` — loại chunk (★ NEW) |

### Bảng `chat_logs` — audit trail tương tác

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `id` | SERIAL PK | Auto-increment |
| `user_query` | TEXT NOT NULL | Câu hỏi người dùng |
| `chatbot_answer` | TEXT | Câu trả lời chatbot |
| `documents_used` | TEXT | JSON array doc_numbers đã sử dụng |
| `confidence_score` | FLOAT | Điểm tin cậy (0.0–1.0) |
| `latency_ms` | FLOAT | Thời gian phản hồi (ms) |
| `created_at` | TIMESTAMP | Thời điểm |

### Indexes

```
documents:     idx_doc_number_type, idx_doc_issuer, idx_doc_effective_date
articles:      idx_article_doc_num (document_id, article_number)
               idx_articles_search_vector GIN(search_vector)  ★ V3 (Alembic)
vector_chunks: idx_vchunk_vector_id, idx_vchunk_doc_article (document_id, article_id),
               idx_vchunk_chunk_type (chunk_type)
chat_logs:     idx_chatlog_created
```

**File**: [backend/app/database/models.py](backend/app/database/models.py) | [backend/app/database/session.py](backend/app/database/session.py)

---

## 🧠 Intent Detection (22+ intent)

Luồng: `Query → Rule-based regex (confidence ≥ 0.5) → LLM fallback nếu thấp → Scenario Detection fallback`

### Intent chung (15 intent)

| Intent | Mô tả | Routing | Priority |
|--------|--------|---------|---------|
| `document_summary` | Tóm tắt VB / danh sách điều luật | **DB direct** → `list_document_articles()` | 8 |
| `document_metadata` | Số hiệu, cơ quan ban hành, năm, hiệu lực | PostgreSQL `documents` table (không dùng vector) | 7 |
| `document_relation` | Văn bản sửa đổi/thay thế nhau | RAG + Relation Prompt | 7 |
| `article_query` | Nội dung Điều/Khoản cụ thể | Direct DB Lookup + RAG | 6 |
| `program_goal` | Mục tiêu chương trình/kế hoạch/đề án | RAG + Program Goal Prompt | 6 |
| `admin_planning` | Lập kế hoạch, phân bổ nguồn lực, giám sát | Commune Officer Pipeline | 6 |
| `kiem_tra_ho_so` | Kiểm tra hồ sơ thiếu/đủ | Document Checker | 6 |
| `trich_xuat_van_ban` | Trích xuất thông tin cấu trúc | Extract Tool | 5 |
| `so_sanh_van_ban` | So sánh hai văn bản | Dual RAG + Compare LLM | 5 |
| `tom_tat_van_ban` | Tóm tắt văn bản (legacy copilot) | `list_document_articles` + LLM fallback | 5 |
| `tao_bao_cao` | Tạo báo cáo hành chính | RAG + Report Generator | 5 |
| `soan_thao_van_ban` | Soạn thảo công văn, tờ trình | Draft Tool + RAG | 5 |
| `can_cu_phap_ly` | Căn cứ pháp lý của văn bản | RAG + CAN_CU_PHAP_LY_PROMPT | 5 |
| `giai_thich_quy_dinh` | Giải thích quy định pháp luật | RAG + GIAI_THICH_QUY_DINH_PROMPT | 5 |
| `huong_dan_thu_tuc` | Hướng dẫn thủ tục hành chính | Procedure KB + RAG | 4 |
| `tra_cuu_van_ban` | Tra cứu văn bản pháp luật tổng quát | RAG Enhanced + Citation | 3 |
| `hoi_dap_chung` | Hỏi đáp chung (fallback) | RAG Enhanced | 0 |

### 7 intent cấp xã (Commune-level) ★ NEW

| Intent | Mô tả | Ví dụ | Priority |
|--------|--------|-------|---------|
| `xu_ly_vi_pham_hanh_chinh` | Xử lý vi phạm HC: karaoke ồn, quảng cáo trái phép, tôn giáo trái phép, biển hiệu sai | "Nhóm người lạ tổ chức sinh hoạt tôn giáo trái phép" | 8 |
| `bao_ve_xa_hoi` | Bạo lực gia đình, bảo vệ trẻ em, người yếu thế, tệ nạn | "Trẻ em bị bạo hành" | 8 |
| `kiem_tra_thanh_tra` | Kiểm tra cơ sở kinh doanh, dịch vụ văn hóa, internet | "Kiểm tra quán karaoke" | 7 |
| `thu_tuc_hanh_chinh` | Đăng ký lễ hội, cấp phép, tu bổ di tích, tôn giáo | "Thủ tục xin phép trùng tu di tích" | 7 |
| `hoa_giai_van_dong` | Hòa giải tranh chấp, vận động nếp sống văn minh | "Mâu thuẫn hàng xóm" | 7 |
| `to_chuc_su_kien_cong` | Tổ chức lễ hội, đại hội TDTT, sự kiện văn hóa, nhà văn hóa | "Tổ chức Đại hội Thể dục thể thao cấp cơ sở" | 7 |
| `bao_ton_phat_trien` | Bảo tồn di sản, trùng tu di tích, di sản phi vật thể | "Ngôi chùa cấp quốc gia bị đổ nát sau bão" | 7 |

**Routing:** Tất cả 7 intent → `_answer_commune_officer_query()` → `COMMUNE_OFFICER_SYSTEM_PROMPT` → 5 phần hành động.

### RAG Intent Classifier (tự nhận diện)

Ngoài regex, RAG có thể dùng **embedding + ngân hàng câu ví dụ** để nhận diện loại câu hỏi (`intent_classifier.py` + `intent_detector.get_rag_intents`). Biến `USE_INTENT_CLASSIFIER` trong `.env` được giữ để không gãy cấu hình cũ; nhánh chính gọi `get_rag_intents()`.

- **INTENT_PROTOTYPES:** `legal_lookup`, `scenario`, `multi_article`, `need_expansion` — mỗi intent có danh sách câu ví dụ tiếng Việt.
- **classify_intent(query)** → top intent + confidence; **get_rag_intents(query)** → `{ is_scenario, is_legal_lookup, use_multi_article, needs_expansion }`.
- **Luồng:** `_get_rag_intent_flags(query)` trong `rag_chain_v2`: gọi `get_rag_intents()` (embedding + prototypes), nếu lỗi thì fallback regex. Các flag dùng cho is_commune_query, use_multi, multi_article_conditions.
- **Ngưỡng:** `INTENT_CONFIDENCE_THRESHOLD` (mặc định 0.50). Có thể bổ sung câu ví dụ vào `INTENT_PROTOTYPES` để cải thiện độ chính xác.

### Scenario Detection (fallback)

Khi intent detector không nhận diện được commune-level intent, `_is_scenario_query()` kiểm tra dấu hiệu tình huống (hoặc dùng `rag_intents["is_scenario"]` từ RAG intent classifier):

```
Dấu hiệu scenario: "Ông/bà hãy...", "tham mưu", "xử lý tình huống",
  "trên địa bàn", "của xã", "phối hợp", "kế hoạch ra quân",
  "đề xuất giải pháp", "biện pháp lâu dài"...

Loại trừ (legal lookup): "Điều X quy định gì", "Tóm tắt Nghị định X",
  số hiệu VB (49/2025/QĐ-UBND)
```

### Ninh Bình Routing Guard ★ NEW

```
should_use_ninh_binh_tool():
  1. LEGAL_KEYWORDS? → ❌ skip (→ RAG)
  2. ADMIN_SCENARIO? → ❌ skip (→ commune pipeline)  ★ ngăn route nhầm "di tích", "trùng tu"
  3. NINH_BINH/GEO keyword → ✅ route to web search
```

**Guard:** Nếu query chứa "Điều X" (số cụ thể) → luôn chuyển sang `article_query`, không dùng `document_summary`.

**SPECIALIZED_INTENTS** — đi qua `route_query()`, không dùng RAG thông thường:

```python
SPECIALIZED_INTENTS = {
    "tao_bao_cao", "soan_thao_van_ban", "tom_tat_van_ban",
    "so_sanh_van_ban", "kiem_tra_ho_so", "huong_dan_thu_tuc",
    "trich_xuat_van_ban", "can_cu_phap_ly", "giai_thich_quy_dinh",
    # + 7 commune-level intents
    "xu_ly_vi_pham_hanh_chinh", "kiem_tra_thanh_tra", "thu_tuc_hanh_chinh",
    "hoa_giai_van_dong", "bao_ve_xa_hoi", "to_chuc_su_kien_cong", "bao_ton_phat_trien",
}
```

---

## ⚙️ Cấu hình (`config.py` / `.env`)

### LLM & OpenAI

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `OPENAI_API_KEY` | _(bắt buộc)_ | API key OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model LLM |
| `OPENAI_BASE_URL` | `None` | URL proxy/self-hosted (tùy chọn) |
| `DEFAULT_TEMPERATURE` | `0.5` | Nhiệt độ mặc định LLM |
| `MAX_TOKENS` | `4096` | Giới hạn token response |

### Embedding

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `EMBEDDING_MODEL` | `keepitreal/vietnamese-sbert` | Model embedding tiếng Việt (768 chiều; xem `EMBEDDING_DIM`) |
| `EMBEDDING_FALLBACK_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Fallback khi model chính lỗi |
| `EMBEDDING_DIM` | `768` | Phải khớp output model; đổi model → cân nhắc `QDRANT_RECREATE_ON_DIM_MISMATCH` + `reembed_all.py` |
| `EMBEDDING_MAX_LENGTH` | `512` | Độ dài tối đa token encode |
| `EMBEDDING_DEVICE` | `cuda` | `cuda` hoặc `cpu` |
| `EMBEDDING_BATCH_SIZE` | `32` | Số text xử lý mỗi batch |
| `HF_TOKEN` | `None` | HuggingFace token (tránh rate-limit) |
| `QDRANT_RECREATE_ON_DIM_MISMATCH` | `true` | `true` = xóa tạo lại collection Qdrant khi size vector ≠ `EMBEDDING_DIM` (production nên `false` + migrate thủ công) |

### RAG & Retrieval

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `CHUNK_SIZE` | `1024` | Kích thước chunk (ký tự) — legacy chunker |
| `CHUNK_OVERLAP` | `128` | Overlap giữa chunks (ký tự) — legacy chunker |
| `TOP_K` | `5` | Số chunks đưa vào LLM context |
| `RETRIEVAL_TOP_K` | `40` | Số candidates từ mỗi nguồn (Qdrant, PostgreSQL) |
| `RERANK_TOP_K` | `10` | Số chunks sau rerank (final_k) |
| `MULTI_ARTICLE_MAX_ARTICLES` | `5` | Số điều luật tối đa đưa vào ngữ cảnh khi dùng chế độ đa article |
| `USE_MULTI_ARTICLE_FOR_CONDITIONS` | `true` | Bật đa article cho query điều kiện/quy định/phát triển thể thao |
| `USE_INTENT_CLASSIFIER` | `true` | Dùng embedding + ngân hàng câu ví dụ để nhận diện RAG intent (scenario / multi_article / need_expansion) |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.50` | Ngưỡng tin cậy để chấp nhận intent từ classifier |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker chính qua **FlagEmbedding** `FlagReranker` (đa ngôn ngữ) |
| `RERANKER_FALLBACK_MODEL` | `BAAI/bge-reranker-base` | Fallback **CrossEncoder** (sentence-transformers) khi không tải được FlagReranker |
| `RERANKER_BATCH_SIZE` | `16` | Batch rerank |
| `RERANKER_DEVICE` | `cpu` | Device reranker (cpu để không tranh GPU) |
| `ANSWER_VALIDATION_THRESHOLD` | `0.40` | Ngưỡng cosine sim kiểm tra grounding (`answer_validator`) |
| `CONTEXT_RELEVANCE_THRESHOLD` | `0.15` | Ngưỡng relevance context |

### Infrastructure

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `POSTGRES_USER` | `legal_bot` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `legal_bot_pass` | PostgreSQL password |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `legal_chatbot` | PostgreSQL database |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `law_documents` | Tên collection Qdrant |
| `QDRANT_API_KEY` | _(trống)_ | API key (cho Qdrant Cloud) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `REDIS_CACHE_TTL` | `3600` | Cache TTL (giây) |
| `EMBEDDING_DIM` | `768` | Dimension vector embedding (khớp model) |

> **Ghi chú `.env.example`:** Một số giá trị demo (vd. `RETRIEVAL_TOP_K=20`, `RERANK_TOP_K=5`) có thể khác default trong `config.py` (`40` / `10`). Ưu tiên đọc `config.py` hoặc đồng bộ `.env` với môi trường triển khai.

---

## 🔧 Tech Stack

| Thành phần | Công nghệ | Phiên bản |
|------------|-----------|----------|
| **Frontend** | React + Vite + Tailwind CSS | 18.3 / 5.3 / 3.4 |
| **UI Components** | lucide-react + react-markdown | 0.400 / 9.0 |
| **Backend** | FastAPI + Uvicorn + **python-multipart** (upload) | ≥0.110 / ≥0.29 / ≥0.0.9 |
| **LLM** | OpenAI API (`gpt-4o-mini`) | ≥1.30 |
| **Embedding** | sentence-transformers (`keepitreal/vietnamese-sbert`, 768-dim; fallback MiniLM) | ≥3.0 |
| **Reranker (ưu tiên)** | **FlagEmbedding** `FlagReranker` (`BAAI/bge-reranker-v2-m3`) | ≥1.2 |
| **Relational DB** | PostgreSQL (async SQLAlchemy + asyncpg) | 16 / ≥2.0 / ≥0.29 |
| **Vector DB** | Qdrant | latest |
| **Cache** | Redis | 7 |
| **Keyword Search** | PostgreSQL FTS (`articles.search_vector` + `unaccent`) + ILIKE fallback | pg_trgm / simple dict |
| **Reranker (fallback)** | sentence-transformers **CrossEncoder** (`BAAI/bge-reranker-base`) khi không tải được FlagReranker | ≥3.0 |
| **GPU Acceleration** | PyTorch + NVIDIA CUDA | ≥2.2 |
| **HTTP Client** | httpx | ≥0.27 |
| **Word Parsing** | python-docx + pywin32 (COM) | ≥1.1 / ≥306 |

---

## 🚀 Cài đặt & Chạy

### Yêu cầu

- **Python** ≥ 3.10
- **Node.js** ≥ 18
- **Docker & Docker Compose** (cho PostgreSQL, Qdrant, Redis)
- **OpenAI API Key** (hoặc API tương thích qua `OPENAI_BASE_URL`)
- **NVIDIA GPU** + CUDA (khuyến nghị, không bắt buộc)

### 1. Khởi động Infrastructure

```bash
# Khởi động PostgreSQL + Qdrant + Redis
docker compose up -d
```

### 2. Backend

```bash
cd backend

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Cài dependencies (production + test: FastAPI stack, DB drivers, Qdrant, Redis,
# sentence-transformers, FlagEmbedding, torch, wikipedia, python-multipart, pytest, requests, …)
pip install -r requirements.txt

# (Tùy chọn) PyTorch với CUDA – xem https://pytorch.org
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# Cấu hình API key
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac
# Sửa .env: OPENAI_API_KEY=sk-...

# Migration PostgreSQL (Alembic) — bật FTS + cột TEXT, v.v.
alembic upgrade head

# Chạy server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Sau khi **đổi model embedding** hoặc **EMBEDDING_DIM** (Qdrant recreate), chạy từ `backend`:

```bash
python scripts/reembed_all.py
```

### Makefile (thư mục gốc repo)

```bash
make test   # pytest backend/tests/
make eval   # pytest backend/tests/evaluation/
make all    # cả hai
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Mở trình duyệt

Truy cập **http://localhost:5173**

---

## 🔌 API Endpoints

### Legacy (tương thích ngược hoàn toàn)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/upload` | Upload file `.doc`/`.docx` → chunk, index, insert `documents` table |
| `POST` | `/api/chat` | Chat — Copilot pipeline, response kèm `confidence` + `intent` |
| `POST` | `/api/chat/stream` | Chat streaming (SSE) — intent → sources → tokens |
| `GET` | `/api/datasets` | Danh sách datasets kèm metadata |
| `DELETE` | `/api/datasets/{id}` | Xóa dataset |
| `POST` | `/api/datasets/reindex` | Rebuild toàn bộ chunk/index |
| `GET` | `/api/gpu` | Trạng thái GPU (device, VRAM, CUDA) |
| `GET` | `/api/health` | Health check: PostgreSQL, Qdrant, Redis + tên `embedding_model` / `reranker_model` trong body |

### Copilot

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/copilot/chat` | Chat AI Copilot — intent routing + context memory |
| `POST` | `/api/copilot/chat/stream` | Streaming SSE: intent → sources → answer tokens |

### Knowledge Search

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/search` | Hybrid search (Qdrant vector + PostgreSQL keyword), trả top documents |

### Intent Detection

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/intent` | Trả `intent` + `confidence` |

### Tool APIs

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/tools/summarize` | Tóm tắt văn bản pháp luật |
| `POST` | `/api/tools/extract` | Trích xuất thông tin quan trọng |
| `POST` | `/api/tools/draft` | Soạn thảo văn bản hành chính |
| `POST` | `/api/tools/classify` | Phân loại văn bản |

### Conversation Memory

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/api/conversations` | Liệt kê tất cả conversations |
| `POST` | `/api/conversations` | Tạo conversation mới |
| `GET` | `/api/conversations/{id}` | Chi tiết conversation + messages |
| `DELETE` | `/api/conversations/{id}` | Xóa conversation |

### Thủ tục hành chính

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/procedure/steps` | Tra cứu các bước thủ tục |
| `POST` | `/api/procedure/check` | Kiểm tra hồ sơ còn thiếu |
| `GET` | `/api/procedure/list` | Liệt kê tất cả thủ tục |

### Văn bản & Báo cáo

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/document/summarize` | Tóm tắt văn bản pháp luật |
| `POST` | `/api/document/compare` | So sánh hai văn bản |
| `POST` | `/api/report/generate` | Tạo báo cáo hành chính |

---

## 🛠️ Agent Tools (7 tools)

| Tool | Chức năng | Trigger |
|------|-----------|---------|
| `search_document` | Tìm kiếm văn bản qua Hybrid Retrieval (Qdrant + PostgreSQL) | `tra_cuu_van_ban`, `article_query` |
| `get_procedure_steps` | Lấy các bước thủ tục hành chính | `huong_dan_thu_tuc` |
| `check_documents` | Kiểm tra hồ sơ thiếu/đủ giấy tờ | `kiem_tra_ho_so` |
| `summarize_document` | Tóm tắt VB: danh sách điều luật (DB) + LLM fallback | `tom_tat_van_ban`, `document_summary` |
| `compare_documents` | So sánh 2 văn bản (giống/khác/thay đổi) | `so_sanh_van_ban` |
| `generate_report` | Tạo báo cáo hành chính có cấu trúc | `tao_bao_cao` |
| `search_ninh_binh_info` | Tra cứu thông tin Ninh Bình qua **Wikipedia + OpenAI web_search**. Không dùng cho câu hỏi pháp luật. | Ninh Bình router + legal guard |
| `search_web` | OpenAI web_search API wrapper (fallback khi Wikipedia không đủ) | Tự động fallback |

---

## 🧠 Embedding & Retrieval Pipeline

### Embedding Model

| Thuộc tính | Giá trị (mặc định `config.py`) |
|-----------|---------|
| Model | `keepitreal/vietnamese-sbert` |
| Số chiều vector | `EMBEDDING_DIM` = **768** (phải khớp Qdrant collection) |
| Max sequence | `EMBEDDING_MAX_LENGTH` = **512** tokens |
| Fallback | `paraphrase-multilingual-MiniLM-L12-v2` (khác chiều → cần migrate Qdrant + reembed) |
| Cache | HuggingFace cache (thường `~/.cache/huggingface/`) |

### Startup Sequence

```
lifespan(app):
  1. init_postgres()           → Tạo PostgreSQL tables (documents, chapters, sections, articles, clauses, vector_chunks, chat_logs)
  2. ensure_collection()       → Tạo/kiểm tra Qdrant collection (law_documents, cosine, EMBEDDING_DIM); có thể recreate nếu mismatch
  3. warmup_embeddings()       → Load sentence-transformers model vào GPU/CPU
  4. warmup_reranker()         → Load FlagReranker hoặc CrossEncoder (RERANKER_MODEL, fallback RERANKER_FALLBACK_MODEL)
  5. warmup_intent_index()     → Intent detector semantic index (22+ intent)
  6. warmup_rag_intent_index() → RAG intent classifier (scenario / multi_article / need_expansion)
  7. warmup_domain_index()     → Legal domain classification index (the_thao, van_hoa, ...)
```

### Ingestion Pipeline (khi upload tài liệu)

```
File .doc/.docx
  → _safe_filename()                    : strip path prefix, truncate >150 chars
  → parser/docx_parser.py              : trích text (paragraph + table, .doc qua Word COM)
  → pipeline/cleaner.py                : NFC normalize, loại bỏ artifact, collapse whitespace
  → pipeline/structure_detector.py     : phát hiện Chương/Mục/Điều/Khoản/Điểm
  → _extract_doc_number()              : regex từ text body + filename tiếng Việt
  → _title_from_filename()             : lấy tên file làm title
  → pipeline/legal_chunker.py          : ★ Three-tier chunking
  │    ├── chunk_type="article"        : 1 chunk = toàn bộ nội dung 1 Điều
  │    ├── chunk_type="clause"         : 1 chunk = nội dung 1 Khoản/Điểm
  │    └── chunk_type="token_sub"      : sub-chunk 512 tokens (overlap) cho Điều dài
  → pipeline/embedding.py              : batch encode (batch_size=32) → float32 vectors → L2 normalize
  → PostgreSQL                         : INSERT documents, articles, clauses, vector_chunks (+ chunk_type)
  → pipeline/vector_store.py           : upsert vectors vào Qdrant (batch 100)
  → Log: 9 stages chi tiết (parse → clean → detect → chunk → embed → store)
```

### Retrieval Pipeline (khi query)

```
Query
  → article_lookup.lookup_article_from_db()  ★ Direct DB Lookup (ưu tiên cao nhất)
  │
  ├─ Mode 1: parse_article_clause_query()
  │    Có "Điều X" → _find_documents() → _find_articles() → _get_clause_content()
  │    → Return passages (nếu tìm thấy)
  │
  ├─ Mode 2: _extract_doc_number_ref()  ★ NEW
  │    Có số hiệu VB (49/2025/QĐ-UBND, 06/CT-UBND)
  │    → _find_document_by_number() (exact → contains → diacritics-stripped → numeric prefix)
  │    ├─ _is_general_query()? → _get_all_articles_passages() (tất cả articles, max 30)
  │    └─ Có topic? → _find_articles_by_topic() → passages theo chủ đề
  │
  ├─ Mode 3: _extract_document_name() + _extract_topic_keywords()  ★ NEW
  │    Có tên luật nhưng không có số Điều
  │    → _find_documents() → _find_articles_by_topic() (title match → content match)
  │    → Return passages (nếu tìm thấy)
  │
  └─ [Không match / fallback] →
      → _extract_doc_reference()             : trích số hiệu VB (hỗ trợ 06/CT-UBND, 06CT–UBND)
      → _resolve_doc_number()                : normalize diacritics (NĐ→ND), slash↔underscore
      → embed_query()                        : shape (1, EMBEDDING_DIM), L2 normalize
      → retrieval/vector_retriever.py        : Qdrant search (top_k, filtered)
      → retrieval/keyword_retriever.py       : FTS `articles` (unaccent) hoặc ILIKE `vector_chunks`
      → RRF merge (k=60)
      → retrieval/reranker.py               : FlagReranker / CrossEncoder rerank → diversify_by_article + dynamic_max_articles (hybrid_retriever)
      → _score_and_group_articles()         : doc_bonus=5.0, article_number_bonus=5.0
      → _fetch_full_article_chunks()        : expand best article
      → Build context + SYSTEM_PROMPT_V2 (auto Dạng A/B)
      → OpenAI LLM → Answer + citations
      → _strip_hallucinated_doc_numbers()   : loại bỏ số hiệu VB không có trong context
```

### Tối ưu hiệu năng

| Kỹ thuật | Mô tả |
|---------|-------|
| **Singleton model** | Model embedding chỉ load 1 lần, tái sử dụng cho mọi request |
| **GPU auto-detect** | Tự phát hiện CUDA, fallback CPU nếu không có GPU |
| **Batch processing** | Encode nhiều text cùng lúc (batch_size=32 mặc định) |
| **Redis caching** | Cache kết quả query (SHA-256 key, TTL 1h), graceful degradation |
| **Qdrant vector search** | Near-realtime search, hỗ trợ filtering + payload |
| **PostgreSQL connection pool** | Async pool (pool_size=20) cho concurrent requests |
| **L2 normalize + cosine** | Cosine similarity chính xác qua Qdrant cosine distance |
| **Reranking** | FlagReranker `bge-reranker-v2-m3` (fallback CrossEncoder `base`) + diversify theo document trong `hybrid_retriever` |
| **Three-tier chunking** | Article + Clause + 512-token sub-chunks — tối ưu retrieval granularity |
| **Direct DB Lookup (3 mode)** | Bypass vector search khi có Điều/số hiệu VB/tên luật — precision 100% |
| **Fallback full article** | Nếu chunk không đủ, lấy ALL chunks của article từ PostgreSQL |
| **Doc reference resolution** | Tự normalize diacritics, slash/underscore, trailing ID khi resolve doc_number |
| **Scenario detection** | Nhận diện câu hỏi tình huống → route commune pipeline thay vì legal lookup |
| **Admin scenario guard** | Ngăn route nhầm câu hỏi hành chính (di tích, trùng tu...) sang Ninh Bình web search |
| **Anti-hallucination post-process** | Sau khi LLM trả lời, strip doc numbers không có trong context (diacritics-aware) |
| **PostgreSQL interaction log** | Ghi log mọi query/answer/latency vào chat_logs |

---

## 🔄 Context Resolution & Follow-up

### Context Reference Detection

Copilot Agent nhận diện câu hỏi tham chiếu ngữ cảnh qua regex patterns:

```python
CONTEXT_REFERENCE_PATTERNS = [
    r"văn bản (trên|này|đó|vừa rồi|vừa soạn)",
    r"kế hoạch (trên|này|đó|vừa rồi)",
    r"nội dung (trên|này|đó)",
    r"quyết định (trên|này|đó)",
    r"thông báo (trên|này|đó)",
    r"báo cáo (trên|này|đó)",
    r"công văn (trên|này|đó)",
    r"(nó|cái đó|cái này|cái trên)",
]
```

### Follow-up Question Resolution

Khi phát hiện follow-up, hệ thống tự động bổ sung ngữ cảnh:

```
User: "Soạn kế hoạch phát triển du lịch"
→ Agent soạn văn bản, lưu metadata (loại=kế hoạch, lĩnh vực=du lịch)

User: "Bổ sung thêm phần ngân sách cho văn bản trên"
→ _is_context_reference() = True
→ Lấy document context từ conversation_store
→ Enriched: "Bổ sung thêm phần ngân sách (ngữ cảnh: loại văn bản: kế hoạch, lĩnh vực: du lịch)"
```

### Document Context Memory (conversation_store)

```python
# Lưu metadata tài liệu vào conversation
conversation_store.update_document_context(conv_id, {
    "loai_van_ban": "kế hoạch",
    "linh_vuc": "du lịch",
    "co_quan": "Sở VHTTDL",
    "chu_de": "phát triển du lịch",
    "noi_dung_tom_tat": "..."
})

# Lấy tài liệu gần nhất (giữ history 5 tài liệu)
last_doc = conversation_store.get_last_document_context(conv_id)
```

---

## 📋 Prompt Templates (`config.py`)

Hệ thống sử dụng nhiều prompt template chuyên biệt cho từng tác vụ:

| Template | Mô tả | Sử dụng bởi |
|----------|--------|-------------|
| `SYSTEM_PROMPT_V2` | ★ Trợ lý hành chính thực chiến — auto Dạng A (tra cứu) / Dạng B (5 phần hành động) | `rag_chain_v2.py` (default) |
| `RAG_PROMPT_TEMPLATE_V2` | ★ Template RAG — tự xác định dạng câu hỏi → format phù hợp | `rag_chain_v2.py` (default) |
| `COMMUNE_OFFICER_SYSTEM_PROMPT` | ★ Persona cán bộ VHXH cấp xã — thẩm quyền, 5 phần bắt buộc | `rag_chain_v2.py` (commune pipeline) |
| `COMMUNE_OFFICER_RAG_TEMPLATE` | ★ Template RAG cho tình huống hành chính — field/subject/violation/severity | `rag_chain_v2.py` (commune pipeline) |
| `SYSTEM_PROMPT` | Base legal assistant (anti-hallucination, citation rules) | RAG pipeline (legacy) |
| `RAG_PROMPT_TEMPLATE` | 3-case decision tree (direct/related/no match) | `rag_chain.py` |
| `COPILOT_SYSTEM_PROMPT` | 7 administrative tasks description | Copilot agent |
| `QUERY_REWRITE_PROMPT` | Rewrite query cho retrieval tốt hơn | `rag_chain.py` |
| `SUMMARIZE_PROMPT` | Tóm tắt văn bản có cấu trúc (4 mục) | `document_summarizer.py` |
| `COMPARE_PROMPT` | So sánh hai văn bản (giống/khác/thay đổi/nhận xét) | `document_comparator.py` |
| `REPORT_PROMPT` | Tạo báo cáo hành chính chuẩn | `report_generator.py` |
| `CHECKLIST_SYSTEM_PROMPT` | Kiểm tra hồ sơ đầy đủ | `document_checker.py` |
| `CHECKLIST_PROMPT_TEMPLATE` | Template kiểm tra giấy tờ theo thủ tục | `document_checker.py` |
| `FALLBACK_REASONING_PROMPT` | Suy luận khi không tìm thấy tài liệu | `rag_chain.py` |
| `CAN_CU_PHAP_LY_PROMPT` | Phân tích căn cứ pháp lý cấp bậc | `copilot_agent.py` |
| `GIAI_THICH_QUY_DINH_PROMPT` | Giải thích quy định đơn giản + ví dụ | `copilot_agent.py` |
| `LEGAL_DOCUMENT_FORMAT_PROMPT` | Format soạn thảo theo Nghị định 30/2020 | `draft_tool.py` |

---

## 🏗️ Architecture Unification (v1 → v2)

### Đã xóa hoàn toàn module v1

| File v1 đã xóa | Thay thế bởi v2 |
|----------------|-----------------|
| `main_legacy.py` (SQLite entry point) | `main.py` (PostgreSQL + Qdrant + Redis) |
| `main_v2.py` (duplicate) | Merged vào `main.py` |
| `services/vector_store.py` (FAISS + BM25) | `retrieval/hybrid_retriever.py` |
| `services/db.py` (SQLite) | `database/session.py` + `models.py` |
| `services/embeddings.py` | `pipeline/embedding.py` |
| `services/docx_loader.py` | `parser/docx_parser.py` |
| `services/chunker.py` | `pipeline/chunker.py` |
| `services/document_registry.py` | `database/models.py` |

### Dependencies đã xóa

```
langchain, langchain-text-splitters, faiss-cpu, rank-bm25
```

### Retrieval Adapter Pattern

`services/retrieval.py` wraps v2 `hybrid_search()` với auto session management, converts v2 output format sang v1 interface:

```python
# v2 output: {text_chunk, document_id, ...}
# → v1 interface: {text, dataset_id, metadata, ...}
search_all(query)           # → hybrid_search()
search_with_fallback(query) # → hybrid_search() + reasoning
search_by_metadata(query)   # → PostgreSQL documents table
has_any_dataset()           # → Qdrant collection check
```

---

## 📝 Ví dụ sử dụng

### ★ Tình huống hành chính (Dạng B — 5 phần hành động)

```
"Có thông tin một nhóm người lạ mặt đến địa phương tổ chức sinh hoạt tôn giáo trái phép.
 Ông/bà cần phối hợp xác minh, vận động và xử lý tình huống này."
→ Scenario detection → Commune Officer Pipeline
→ Output: 5 phần (Nhận định → Căn cứ Luật Tín ngưỡng → Quy trình xử lý → Phối hợp CA xã → Giải pháp)
```

```
"Ngôi chùa là di tích lịch sử cấp quốc gia bị đổ nát sau bão. Trụ trì muốn thay mái ngói
 theo kiến trúc hiện đại. Ông/bà sẽ hướng dẫn quy trình xin phép trùng tu?"
→ intent: bao_ton_phat_trien → Commune Officer Pipeline
→ Output: Nhận định vấn đề → Căn cứ Luật Di sản → 5 bước (Hồ sơ → Thẩm định → Cấp phép → Thi công → Giám sát) → Phối hợp Sở VHTT → Giải pháp
```

```
"Nhiều biển hiệu cửa hàng trên trục đường chính xã có kích thước quá lớn, che khuất tầm nhìn
 giao thông. Ông/bà xây dựng kế hoạch ra quân chấn chỉnh và xử lý vi phạm?"
→ intent: xu_ly_vi_pham_hanh_chinh → Commune Officer Pipeline
→ Output: Kế hoạch ra quân → Căn cứ Luật Quảng cáo → Bước xử lý (Khảo sát → Thông báo → Kiểm tra → Xử phạt → Báo cáo) → Phối hợp
```

```
"Nhà văn hóa xã khang trang nhưng tỷ lệ người dân sinh hoạt rất thấp.
 Ông/bà hãy đề xuất 3 giải pháp thu hút các câu lạc bộ, đội nhóm."
→ intent: to_chuc_su_kien_cong → Commune Officer Pipeline
→ Output: 3 giải pháp cụ thể + ai thực hiện + cơ chế giám sát
```

```
"Xã chuẩn bị tổ chức Đại hội TDTT cấp cơ sở. Có ý kiến lược bỏ các môn thể thao dân tộc.
 Ông/bà sẽ tham mưu bảo tồn giá trị văn hóa truyền thống?"
→ Scenario detection ("tham mưu") → Commune Officer Pipeline
→ Output: Tham mưu cân bằng hiện đại + truyền thống + phương án tổ chức cụ thể
```

### Tra cứu điều khoản cụ thể (Dạng A — trích dẫn luật)
```
"Điều 47 Luật Di sản văn hóa 2024 quy định gì?"
→ Direct DB Lookup (Mode 1) → trích dẫn nguyên văn + giải thích thực tiễn
```

### Tra cứu theo số hiệu văn bản (Direct DB Lookup Mode 2)
```
"Tóm tắt 49/2025/QĐ-UBND"
→ _extract_doc_number_ref() → _find_document_by_number() → _get_all_articles_passages()
→ Tất cả điều luật của văn bản

"Trong chỉ thị 06/CT-UBND trách nhiệm của UBND cấp xã là gì?"
→ _extract_doc_number_ref() → _find_document_by_number() → _find_articles_by_topic("trách nhiệm UBND")
→ Các điều liên quan trong chỉ thị
```

### Tra cứu theo chủ đề trong luật (Direct DB Lookup Mode 3)
```
"Các hành vi bị nghiêm cấm trong Luật Di sản văn hóa Việt Nam"
→ _extract_document_name() → _extract_topic_keywords("hành vi nghiêm cấm")
→ Tìm article title/content match → trả kết quả chính xác
```

### Tra cứu metadata văn bản
```
"Thông tư 13/2024/TT-BVHTTDL do cơ quan nào ban hành?"
→ intent: document_metadata → PostgreSQL documents table (không dùng vector search)
```

### Kiểm tra hồ sơ
```
"Tôi đã nộp: giấy đề nghị, điều lệ công ty. Còn thiếu gì?"
→ intent: kiem_tra_ho_so → Document Checker → danh sách thiếu
```

### Soạn thảo văn bản
```
"Soạn công văn xin gia hạn giấy phép kinh doanh"
→ intent: soan_thao_van_ban → Draft Tool → văn bản theo thể thức hành chính (Nghị định 30/2020)
```

### Tóm tắt văn bản (danh sách điều luật)
```
"Luật báo chí bao gồm những quy định gì"
→ intent: document_summary → _find_document("Luật báo chí") → DB Article table
→ Output:
  Luật Báo chí (Số hiệu: 103/2016/QH13)
  Tổng số điều: 61
  - Điều 1. Phạm vi điều chỉnh
  - Điều 2. Đối tượng áp dụng
  - ...
```

### Tra cứu thông tin Ninh Bình (phi pháp lý)
```
"Ninh Bình có bao nhiêu huyện?"
→ Ninh Bình router → Wikipedia search + OpenAI web_search fallback

"Gia Hưng ở đâu?"
→ GEO_INFO_PATTERN match → web search với context "tỉnh Ninh Bình"

"Nghị định về du lịch Ninh Bình quy định gì?"
→ Legal keyword detected → RAG pipeline (không dùng web search)

"Ngôi chùa di tích quốc gia bị đổ nát, quy trình trùng tu?"
→ ADMIN_SCENARIO detected → ★ KHÔNG route sang Ninh Bình → commune pipeline
```
*Lưu ý: Nếu câu hỏi chứa từ pháp lý hoặc dấu hiệu tình huống hành chính → hệ thống KHÔNG route sang web search, mà chuyển sang RAG/commune pipeline.*

### Follow-up với context memory
```
User: "Soạn kế hoạch phát triển du lịch tỉnh năm 2025"
→ intent: soan_thao_van_ban → draft_tool → save document metadata

User: "Bổ sung phần ngân sách cho văn bản trên"
→ _is_context_reference() → enrich query với context (loại=kế hoạch, lĩnh vực=du lịch)
→ RAG Enhanced với ngữ cảnh tài liệu
```

### API trực tiếp
```bash
# Nhận diện intent
curl -X POST http://localhost:8000/api/intent \
  -H "Content-Type: application/json" \
  -d '{"question": "Điều 47 Luật Di sản văn hóa quy định gì?"}'

# Chat với conversation memory
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quy định về cấp phép xây dựng?", "conversation_id": "abc123"}'

# Tóm tắt văn bản
curl -X POST http://localhost:8000/api/tools/summarize \
  -H "Content-Type: application/json" \
  -d '{"content": "Tóm tắt Nghị định 01/2021"}'

# CRUD Conversations
curl -X POST http://localhost:8000/api/conversations \
  -H "Content-Type: application/json" \
  -d '{"title": "Tra cứu luật di sản"}'

curl -X GET http://localhost:8000/api/conversations
```

---

## 💡 Lưu ý vận hành

- **Docker Infrastructure:** PostgreSQL, Qdrant, Redis cần chạy trước khi khởi động backend. Dùng `docker compose up -d`.
- **Alembic:** Chạy `alembic upgrade head` trong `backend` sau khi pull code mới có migration; cần quyền `CREATE EXTENSION` trên DB (unaccent, pg_trgm).
- **Đổi chiều vector / model embedding:** Bật `QDRANT_RECREATE_ON_DIM_MISMATCH` hoặc xóa collection thủ công, rồi `python scripts/reembed_all.py`.
- **GPU:** Lần đầu chạy tải model embedding (kích thước tùy model; vietnamese-sbert ~vài trăm MB–GB). Đổi `EMBEDDING_DEVICE=cpu` trong `.env` nếu không có GPU NVIDIA.
- **Proxy:** Đặt `OPENAI_BASE_URL` để dùng LM Studio, OpenRouter, hay bất kỳ OpenAI-compatible API.
- **Tăng tốc:** Tăng `EMBEDDING_BATCH_SIZE` nếu GPU nhiều VRAM. Tăng `TOP_K` để tăng recall (tốn thêm token LLM).
- **File `.doc`:** Chỉ hỗ trợ trên Windows qua Microsoft Word COM automation (pywin32).
- **Conversation ID:** Gửi kèm `conversation_id` trong `/api/chat` để kích hoạt context memory multi-turn.
- **Reindex:** Gọi `POST /api/datasets/reindex` sau khi thay đổi `CHUNK_SIZE`, `CHUNK_OVERLAP`, hay chiến lược chunking.
- **Redis cache:** Hệ thống tự graceful degradation nếu Redis không khả dụng. Cache flush qua API health endpoint.
- **`backend/requirements.txt`:** Cài một lần đủ cho chạy API và chạy `make test` / `pytest` (đã gồm `pytest`, `requests`). Môi trường cũ nên `pip install -r requirements.txt` lại sau khi pull để đồng bộ **FlagEmbedding**, **Alembic**, **python-multipart**, v.v.

---

## 10. Hành vi hiện tại cần lưu ý

- **Dual response mode:** Chatbot tự động phân biệt câu hỏi tra cứu (Dạng A → trích dẫn luật) vs tình huống hành chính (Dạng B → 5 phần hành động). Không cần user chỉ định.
- **Commune officer pipeline:** Câu hỏi có "Ông/bà hãy...", "tham mưu", "trên địa bàn", "kế hoạch ra quân"... tự động route sang pipeline cán bộ VHXH với 5 phần bắt buộc.
- **Ninh Bình admin guard:** Câu hỏi chứa "di tích", "trùng tu", "tôn giáo" + dấu hiệu hành chính sẽ **KHÔNG** bị route nhầm sang Ninh Bình web search.
- **Direct DB Lookup ưu tiên cao nhất:** Khi user hỏi "Điều X", "49/2025/QĐ-UBND", hoặc "hành vi cấm trong Luật Di sản" → hệ thống truy vấn DB trước, không qua vector search.
- **Three-tier chunking:** Mỗi văn bản được chunk theo 3 cấp: article (toàn bộ Điều), clause (Khoản/Điểm), token_sub (512 tokens). Column `chunk_type` trong `vector_chunks` phân biệt loại.
- Frontend truyền `question` + `temperature` cho `/api/chat/stream`; **chưa truyền** `dataset_id`, `filter`, `conversation_id`.
- API conversation (CRUD) đã đầy đủ ở backend (`GET/POST/DELETE /api/conversations`), nhưng **frontend chưa tích hợp** UI hội thoại nhiều phiên.
- Frontend **không sử dụng** `/api/copilot/chat/stream` — chỉ dùng `/api/chat/stream`.
- Sidebar có chọn dataset, nhưng luồng chat hiện truy vấn toàn bộ dữ liệu đang index trong backend.
- Backend hỗ trợ đầy đủ intent routing, tools, conversation memory — nhưng ~70% tính năng backend **chưa** được frontend khai thác.
- Draft tool hỗ trợ 8 loại văn bản (kế hoạch, quyết định, thông báo, báo cáo, công văn, tờ trình, đơn, biên bản) theo Nghị định 30/2020/NĐ-CP. Căn cứ pháp lý chỉ lấy từ DB, có anti-hallucination refs.
- Toàn bộ module v1 (SQLite, FAISS, langchain, sync) đã bị xóa. Hệ thống hoàn toàn trên v2 (PostgreSQL + Qdrant + Redis + async).
- Ninh Bình search đã chuyển từ JSON tĩnh sang **Wikipedia + OpenAI web_search** với LLM-based context augmentation + admin scenario guard.
- Document number extraction hỗ trợ regex từ text body + filename tiếng Việt, tự strip path prefix khi upload folder. Column `doc_number` widened từ VARCHAR(100) → VARCHAR(255).
- Anti-hallucination 3 lớp: (1) strict prompt, (2) post-generation doc number stripping, (3) diacritics-aware comparison.

## 11. Kiểm thử

Thư mục tests hiện có:
- `test_e2e.py` — End-to-end integration test
- `test_search.py` / `test_search2.py` / `test_search3.py` — Search pipeline tests
- `test_ninh_binh_tool.py` — Unit test Ninh Bình search tool (huyện, du lịch, dân số, legal guard)

Chạy pytest (hoặc `make test` từ thư mục gốc repo):

```powershell
cd backend
pytest
```

## 12. Gợi ý nâng cấp gần

1. Tích hợp `conversation_id` trong frontend chat stream — kích hoạt multi-turn context memory.
2. Xây dựng UI quản lý hội thoại (ConversationList, ConversationSidebar) — tận dụng API conversation CRUD đã có.
3. Truyền `dataset_id`/`filter` từ UI xuống `/api/chat` và `/api/chat/stream`.
4. Hiển thị intent detection kết quả trên UI (badge hoặc tooltip) — đặc biệt phân biệt Dạng A/B.
5. Tạo UI cho tools (summarize, extract, draft, classify) — hiện chỉ accessible qua API.
6. Bổ sung test coverage cho routers, retrieval pipeline, và commune officer pipeline.
7. ~~Thêm PostgreSQL full-text search (tsvector)~~ — đã có trên `articles.search_vector` + fallback ILIKE; có thể mở rộng FTS sang `vector_chunks` nếu cần.
8. Chuẩn hóa schema SSE để tránh parse JSON lồng nhiều lớp.
9. Frontend hiển thị document summary dạng tree/accordion cho danh sách điều luật.
10. Mở rộng scenario detection với ML classifier (thay vì regex) để bắt thêm nhiều dạng câu hỏi hành chính.
11. Thêm mẫu biểu (form templates) cho quy trình xử lý: biên bản vi phạm, đơn xin phép, biên bản kiểm tra.
12. Evaluation framework: đánh giá chất lượng trả lời (relevance, completeness, actionability) cho cả Dạng A và Dạng B.
