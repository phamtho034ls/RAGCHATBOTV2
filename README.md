# Government AI Copilot — Trợ lý hành chính & RAG văn bản pháp luật

Hệ thống **RAG production** cho cán bộ VHXH: **PostgreSQL** (metadata + full-text `articles`), **Qdrant** (vector), **Redis** (cache), **hybrid retrieval** (semantic + FTS/ILIKE → RRF → rerank), **OpenAI API** cho LLM. Giao diện web tiếng Việt (React/Vite); API `**/api/chat`** / `**/api/chat/stream**` là đường chính — `**rag_chain_v2**`. Stream hiện tại là **pseudo-stream** (chia nhỏ câu trả lời sau khi LLM xong), không stream token trực tiếp từ OpenAI.

- **API:** FastAPI `Government AI Copilot API`, version `2.0.0`
- **Module SQLite/FAISS/langchain v1:** đã gỡ; chỉ còn stack async v2

---

## Kiến trúc đang dùng (tóm tắt)

```mermaid
flowchart TB
  subgraph clients["Client"]
    FE[Frontend React]
  end
  subgraph api["FastAPI"]
    CHAT["/api/chat → rag_chain_v2"]
    COP["/api/copilot → copilot_agent + rag_unified → v2"]
    UP["/api/upload → ingestor"]
    SR["/api/search → hybrid_search"]
  end
  subgraph data["Dữ liệu"]
    PG[(PostgreSQL)]
    QD[(Qdrant)]
    RD[(Redis)]
  end
  FE --> CHAT
  CHAT --> PG
  CHAT --> QD
  CHAT --> RD
  COP --> PG
  COP --> QD
  UP --> PG
  UP --> QD
  SR --> PG
  SR --> QD
```




| Thành phần                      | Vai trò                                                                                                                                                                                                                                                  |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Frontend**                    | React 18 + Vite + Tailwind; gọi `**/api/chat/stream`** (body: `question`, `temperature`, tuỳ chọn `conversation_id`)                                                                                                                                     |
| `**rag_chain_v2**`              | Pipeline chat production: `analyze_query` → commune arbiter → checklist/draft/summary hoặc hybrid RAG → LLM (`SYSTEM_PROMPT_V2` / `RAG_PROMPT_TEMPLATE_V2`)                                                                                              |
| `**rag_unified.py**`            | Bọc `**rag_chain_v2.rag_query**` cho Copilot: chuẩn hóa `sources` (content/metadata), stream tương thích SSE cũ                                                                                                                                          |
| `**rag_chain.py**`              | Hàm tiện ích dùng bởi v2/Copilot: `**_fallback_reasoning**`, `**strip_hallucinated_references**`, checklist helpers; `**rag_query**` / `**rag_query_stream**` là pipeline retrieval cũ trong cùng file — **không** mount qua router (chat chính dùng v2) |
| `**query_route_classifier.py`** | LLM JSON (`response_format: json_object`): pháp lý vs web, checklist vs nội dung, tham chiếu hội thoại — bổ sung/override sau `analyze_query`                                                                                                            |
| `**copilot_agent**`             | `detect_intent` → (tuỳ intent) `route_query` hoặc `rag_query_unified` + `conversation_id`; có thể có `classify_user_utterance` khi bật classifier                                                                                                        |
| **Ingest**                      | `.doc`/`.docx` → parse → chunk theo Điều/Khoản → embed → PostgreSQL + Qdrant                                                                                                                                                                             |
| **Retrieval**                   | `hybrid_retriever`: vector + keyword (FTS `articles` + fallback ILIKE) → RRF → **FlagReranker** (fallback CrossEncoder) → diversify / cap article                                                                                                        |


### Tech stack (tham chiếu)


| Lớp               | Công nghệ                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| Frontend          | React 18, Vite, Tailwind, react-markdown                                                                |
| Backend           | FastAPI, Uvicorn, SQLAlchemy async + asyncpg, Alembic                                                   |
| LLM               | OpenAI API (hoặc tương thích qua `OPENAI_BASE_URL`)                                                     |
| Embedding         | sentence-transformers (`keepitreal/vietnamese-sbert`, 768-d)                                            |
| Intent (optional) | `transformers` + PhoBERT fine-tuned cục bộ (`backend/app/intent_model`, bật/tắt `INTENT_MODEL_ENABLED`) |
| Rerank            | FlagEmbedding `FlagReranker` → fallback CrossEncoder                                                    |
| Vector            | Qdrant; Cache: Redis                                                                                    |


---

## Phiên bản schema & hành vi gần đây


| Chủ đề                                | Nội dung                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Alembic**                           | Chuỗi migration: `v3_001` (extension `unaccent`, `pg_trgm`, cột generated `articles.search_vector` + GIN), `v3_002`/`v3_003` (TEXT documents), `004_chat_conv` (hội thoại), `005_law_intents` (JSON `documents.law_intents`). Chạy `alembic upgrade head` trong `backend/`.                                                                                    |
| **FTS / keyword**                     | Ưu tiên FTS trên `articles.search_vector` (trong `begin_nested` savepoint); lỗi/thiếu cột → cảnh báo và ILIKE trên `vector_chunks`.                                                                                                                                                                                                                            |
| **Domain (truy vấn)**                 | `domain_classifier.classify_query_domain` → `get_domain_filter_values` trong `rag_chain_v2` → tiền lọc Qdrant theo `legal_domain`.                                                                                                                                                                                                                             |
| **Nhãn theo văn bản (`law_intents`)** | Ingest: `classify_document_law_intents(title, trích yếu/đoạn đầu)` — đa nhãn cùng từ vựng `LEGAL_DOMAINS`; lưu DB + payload Qdrant; `legal_domain` = phần tử đầu.                                                                                                                                                                                              |
| **Hybrid / rerank**                   | Sau rerank: boost nếu `legal_domain` ∈ domain filter; post-filter domain mềm (không trả rỗng khi metadata lệch); **phạt chủ đề** (`TOPIC_MISMATCH_PENALTY`) khi domain câu hỏi không giao với `law_intents`/`legal_domain` — bỏ qua khi user chỉ rõ số hiệu văn bản. Enrich DB: nếu thiếu cột `law_intents`, retry không chọn cột + `rollback()` (PostgreSQL). |
| **Article selection**                 | `article_selection.py`: `diversify_by_article`, `dynamic_max_articles` (sau rerank trong `hybrid_retriever`).                                                                                                                                                                                                                                                  |
| **Commune route arbiter**             | `commune_route_arbiter.py`: cosine prototype “thủ tục cấp xã” vs “tra cứu pháp lý”; nếu |s_commune − s_legal| < `COMMUNE_ROUTE_MARGIN` → LLM JSON. `warmup_commune_route_index()` sau embedding trong `main.py`.                                                                                                                                               |
| `**query_intent`**                    | `compute_intent_bundle`: regex đa văn bản / checklist + `detect_intent_rule_based` (**PhoBERT → semantic prototype → structural YAML**) + `map_intent_to_rag_flags`; LLM `merge_utterance_labels_into_analysis` khi bật `QUERY_UTTERANCE_CLASSIFIER_ENABLED`.                                                                                                  |
| **Intent async (`detect_intent`)**    | Guard → PhoBERT → semantic → structural YAML → LLM zero-shot (xem `intent_detector.py`). Khác `detect_intent_rule_based` (không LLM).                                                                                                                                                                                                                          |
| **Cấu hình pattern**                  | `intent_patterns/routing.yaml`: `structural_rules` (regex + `intent_id`), nhóm `routing` (multi-doc synthesis, checklist, …), `prototype_sentences` tùy chọn; nạp qua `intent_pattern_config.load_intent_pattern_config()` ở startup.                                                                                                                          |
| **Tra cứu vs stub đa nguồn**          | `query_expects_llm_synthesis_from_context` (`query_text_patterns`) → không dùng `_build_multi_source_answer` kiểu liệt kê; prompt thêm khối **trả lời trực tiếp** khi cần.                                                                                                                                                                                     |
| **UX**                                | Căn cứ pháp lý gom theo văn bản; follow-up cuối câu theo ngưỡng độ tin cậy.                                                                                                                                                                                                                                                                                    |
| **Lưu file**                          | Upload vào `backend/app/storage/<id>/` (thư mục **gitignore**).                                                                                                                                                                                                                                                                                                |


---

## Khởi động backend (`main.py` — thứ tự thực tế)

1. `init_postgres()`
2. `warmup_embeddings()`
3. `warmup_commune_route_index()`
4. `ensure_collection()` (Qdrant)
5. `warmup_reranker()`
6. `warmup_intent_index()` (prototype cosine cho intent)
7. `warmup_intent_classifier()` (PhoBERT trong `app/intent_model`; bỏ qua an toàn nếu tắt hoặc thiếu `transformers`/weights)
8. `warmup_domain_index()`
9. Shutdown: `close_postgres()`

---

## Luồng chat production: `POST /api/chat` & `/api/chat/stream`

**Router:** `app/routers/chat_router.py` → `**rag_query`** / `**rag_query_stream**`.

`**ChatRequest`:** nhận `query` hoặc `question` (frontend dùng `question`); tuỳ chọn `conversation_id`, `doc_number`, `temperature`.


| Bước | Việc xảy ra                                                                                                                                                                                      |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0    | `conversation_id` (tạo mới nếu cần)                                                                                                                                                              |
| 1    | Ninh Bình tool (`should_use_ninh_binh_tool` → `route_to_ninh_binh`) nếu khớp                                                                                                                     |
| 2    | `analyze_query` → `intent` (= `routing_intent`), `rag_flags`, `commune_situation`; sau đó (nếu bật classifier) `merge_utterance_labels_into_analysis(..., query=)` — xem mục **Intent & cờ RAG** |
| 3    | Lọc miền pháp lý: `get_domain_filter_values` + `classify_query_domain`                                                                                                                           |
| 4    | Redis cache (bỏ qua với checklist / draft / summary)                                                                                                                                             |

---

## Intent EDA (kiểm thử phân loại intent bằng `data.json`)

Mục tiêu: chạy nhanh pipeline **intent/routing** trên một mẫu ngẫu nhiên từ `data.json` để xem:
- phân phối `detector_intent` / `routing_intent`
- phân phối `rag_flags`
- độ dài câu hỏi, buckets confidence
- top keywords (EDA tokenizer đơn giản)
- ví dụ `out_of_scope`, `nan`, `low_confidence`

### Cách chạy

Chạy từ thư mục `backend/`:

```bash
python scripts/eda_intents_datajson.py --n 500 --seed 42 --out-prefix intent_eda_datajson_500_v2
```

Lưu ý: script tự set `INTENT_MODEL_ENABLED=false` để **không load PhoBERT/transformers**, giúp chạy nhanh và ổn định.

### Output

- `backend/tests/evaluation/results/intent_eda_datajson_500_v2.md`
- `backend/tests/evaluation/results/intent_eda_datajson_500_v2.json`

### Kết quả gần nhất (sample=500, seed=42)

- **explicit doc ref**: 56 (11.20%)
- **mentions Điều**: 58 (11.60%)
- **out_of_scope**: 0
- **detector_nan**: 0

Top `routing_intent`:
- `hoi_dap_chung`: 174 (34.80%)
- `legal_lookup`: 160 (32.00%)
- `giai_thich_quy_dinh`: 68 (13.60%)

Buckets độ dài câu hỏi (chars):
- `80-119`: 304 (60.80%)
- `40-79`: 162 (32.40%)

Buckets confidence (detector):
- `>=0.90`: 266 (53.20%)
- `<0.35`: 183 (36.60%)
| 5    | `**resolve_use_commune_officer_pipeline`** → `_answer_commune_officer_query` nếu true                                                                                                            |
| 6    | `checklist_documents` / `document_drafting` / `document_summary`                                                                                                                                 |
| 7    | `_multi_query_retrieve` hoặc `hybrid_search` (theo `needs_expansion`, `use_multi_article`, …)                                                                                                    |
| 8    | Ghép context (single-article hoặc group theo article); có thể lookup văn bản / multi-source (có điều kiện) / LLM                                                                                 |
| 9    | Hậu xử lý: validate, căn cứ, follow-up, cache, `chat_logs`, lưu hội thoại                                                                                                                        |


### Sơ đồ nhánh (production)

```mermaid
flowchart TD
  RQ[rag_query] --> NB{Ninh Bình?}
  NB -->|có| E1[Kết thúc sớm]
  NB -->|không| AQ[analyze_query + bundle]
  AQ --> CA{resolve_use_commune_officer_pipeline}
  CA -->|true| CO[Pipeline cán bộ 5 phần + RAG]
  CA -->|false| SP{routing_intent}
  SP -->|checklist| CL[Checklist]
  SP -->|draft/summary| DS[Draft / Summary]
  SP -->|khác| HY[Hybrid / multi-query retrieve]
  HY --> ANS[Lookup / multi-source có điều kiện / LLM V2]
  CO --> OUT[Trả lời]
  CL --> OUT
  DS --> OUT
  ANS --> OUT
```



### Endpoint `/api/copilot/chat`

Điều phối đầu: `detect_intent` (async) → `**route_query**` (intent chuyên biệt) hoặc `**rag_query_unified**` → `**rag_chain_v2**`. Frontend màn chat chính dùng `**/api/chat/stream**`, không bắt buộc Copilot.

---

## Intent & cờ RAG

Tầng phân loại gồm **ba lớp có thể chồng lên nhau**:

1. `**query_intent` + `intent_detector.detect_intent_rule_based`** — đồng bộ, không LLM: sinh `routing_intent`, `detector_intent`, `rag_flags` (dùng trong `analyze_query` / RAG v2).
2. `**query_route_classifier**` (LLM JSON, tùy bật) — chỉnh `intent` / substantive / checklist sau bước 1.
3. `**detect_intent` (async)** — dùng Copilot và `/api/intent`: thêm LLM khi các tầng trên không đủ (xem bảng dưới).

**File cấu hình:** `backend/app/intent_patterns/routing.yaml` — `structural_rules` (regex → `intent_id` + độ ưu tiên), các nhóm `routing` (từ khóa cho multi-doc synthesis, checklist, mở rộng nội dung, tham mưu, …), `prototype_sentences` mở rộng prototype embedding.

### Luồng tổng quan (Chat production)

```mermaid
flowchart TB
  subgraph q["Câu hỏi"]
    U[User query]
  end
  subgraph bundle["query_intent + rule-based"]
    M[Multi-doc synthesis / checklist regex]
    RB["detect_intent_rule_based: PhoBERT → semantic → YAML"]
    MAP["map_detector_to_routing_intent + map_intent_to_rag_flags"]
  end
  subgraph qu["query_understanding"]
    AQ["analyze_query: filters, keywords, commune_situation, …"]
  end
  subgraph opt["Tùy chọn"]
    UC["query_route_classifier / merge_utterance_labels"]
  end
  subgraph rag["rag_chain_v2"]
    DOM["classify_query_domain → domain filter"]
    ARB["commune_route_arbiter"]
    RET["hybrid_search: vector + keyword → RRF → rerank → domain boost / topic penalty"]
    LLM2[LLM]
  end
  U --> M
  M --> RB
  RB --> MAP
  MAP --> AQ
  AQ --> UC
  UC --> DOM
  DOM --> ARB
  ARB --> RET
  RET --> LLM2
```



### Đường Copilot (`copilot_agent`)

`detect_intent` (async) ≈ **Guard → PhoBERT → prototype semantic → structural YAML → LLM** — không thay thế `analyze_query` trong `/api/chat`; Copilot rẽ `route_query` hoặc `rag_query_unified` → vẫn vào `**rag_chain_v2`** bên trong.

### Vị trí trong pipeline (`rag_query`)

Trong `rag_chain_v2.rag_query`, **Ninh Bình** (`should_use_ninh_binh_tool`) được xử lý **trước** `analyze_query` (thoát sớm nếu khớp). Đoạn sau chỉ mô tả **tầng intent** sau khi qua Ninh Bình.

```mermaid
flowchart LR
  Q[Câu hỏi] --> QU[analyze_query]
  QU --> B[compute_intent_bundle]
  B --> A["routing_intent + rag_flags + detector_intent"]
  A --> M{Utterance classifier LLM?}
  M -->|có| MERGE[merge_utterance_labels_into_analysis]
  M -->|không| USE[Dùng intent / rag_flags]
  MERGE --> USE
  USE --> ARB[Commune arbiter]
  ARB --> BR{Rẽ nhánh checklist / draft / RAG}
```



- `**analysis["intent"]**` trong code chính là `**routing_intent**` từ bundle (chuỗi dùng rẽ nhánh: `checklist_documents`, `document_drafting`, `document_summary`, `legal_lookup`, `hoi_dap_chung`, …).
- `**analysis["detector_intent"]**`: mã từ `detect_intent_rule_based` (PhoBERT nếu bật → semantic → structural YAML; không LLM), trước khi `map_detector_to_routing_intent`.
- Sau merge classifier, log có thể hiển thị thêm `utterance_labels` trong `analysis`.

---

### Luồng `query_intent` (`query_intent.py`)

**API trung tâm:** `compute_intent_bundle(query)` → một dict gồm `detector_intent`, `detector_confidence`, `routing_intent`, `rag_flags`, `is_checklist`.

**Thứ tự xử lý bên trong `compute_intent_bundle`:**


| Bước | Logic                                                                                                                                                                                 | Ảnh hưởng                                                                                                                                     |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `query_requires_multi_document_synthesis(query)` — regex (ví dụ: *tổng hợp*, *so sánh*, *đối chiếu*, *giữa … và …*, *văn bản nào quy định*, *nằm ở những luật…*, *theo các văn bản*…) | Nếu **true**: không coi là checklist; trong bundle bật `needs_expansion`, `use_multi_article`, `is_legal_lookup`                              |
| 2    | `_is_checklist_documents` — khớp `_CHECKLIST_PATTERNS` **chỉ khi** không multi-synthesis và không khớp `_SUBSTANTIVE_EXPANSION_ROUTING_PATTERNS` (mức phạt, thủ tục, hồ sơ, UBND, …)  | `routing_intent = "checklist_documents"` nếu true                                                                                             |
| 3    | `detect_intent_rule_based` — **PhoBERT** (nếu `INTENT_MODEL_ENABLED`) → semantic (prototype embedding) → structural YAML (`routing.yaml`); **không LLM**                              | Cho `detector_intent` + confidence                                                                                                            |
| 4    | `routing_intent`                                                                                                                                                                      | Nếu không checklist: `map_detector_to_routing_intent(det)` (ví dụ `tom_tat_van_ban` → `document_summary`, `tra_cuu_van_ban` → `legal_lookup`) |
| 5    | `rag_flags`                                                                                                                                                                           | `map_intent_to_rag_flags(detector_intent)` rồi chỉnh thêm                                                                                     |
| 6    | `_narrow_multi_article_boost`                                                                                                                                                         | Nếu câu về **đầu tư kinh doanh có điều kiện** / danh mục ngành nghề → bật `needs_expansion` + `use_multi_article`                             |
| 7    | Multi-synthesis (bước 1)                                                                                                                                                              | Ghi đè/bật cờ như bảng bước 1                                                                                                                 |
| 8    | `_query_needs_substantive_expansion_not_checklist`                                                                                                                                    | Bật `needs_expansion`; với một số cụm (mức phạt, hành vi cấm, biện pháp phòng ngừa) thêm `use_multi_article`                                  |


**Hàm phụ (dùng ở module khác):**

- `compute_rag_flags_for_query(query)` — wrapper trả về chỉ `rag_flags`; được `intent_detector.get_rag_intents()` gọi để **đồng bộ** với bundle.
- `is_consultation_or_advisory_query` — regex câu tham mưu / tình huống; `rag_chain_v2` dùng để điều chỉnh kiểu trả lời (không ép template multi-source kiểu đó).
- `query_mentions_conditional_investment` — chỉ phục vụ footer / nhắc danh mục đầu tư có điều kiện khi câu liên quan.

---

### Luồng `intent_detector` (`intent_detector.py`)


| API                               | Mục đích                                                                   | Ghi chú                                                                                 |
| --------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `detect_intent_rule_based(query)` | **PhoBERT** (tuỳ cấu hình) → semantic prototype → structural YAML          | **Không LLM.** Dùng trong `compute_intent_bundle` / cờ RAG đồng bộ chat v2.             |
| `detect_intent(query)` (async)    | Guard → **PhoBERT** → semantic → structural YAML → `**detect_intent_llm`** | Endpoint `/api/intent`, Copilot; LLM chỉ khi các tầng trên không trả intent đủ tin cậy. |


**Ánh xạ cờ RAG — `map_intent_to_rag_flags(intent)`**  
Mỗi intent trong `VALID_INTENTS` thuộc **đúng một** nhóm (một trong bốn cờ là `true`; các cờ còn lại `false`). Tập hợp trong code: `_RAG_LEGAL_LOOKUP_INTENTS`, `_RAG_MULTI_ARTICLE_INTENTS`, `_RAG_NEEDS_EXPANSION_INTENTS`, `_RAG_SCENARIO_INTENTS`.


| Cờ                  | Intent được gán `true` (tóm tắt)                                                                                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `is_legal_lookup`   | `article_query`, `document_metadata`, `can_cu_phap_ly`, `trich_xuat_van_ban`                                                                                                                                     |
| `use_multi_article` | `tra_cuu_van_ban`, `document_relation`, `tom_tat_van_ban`, `so_sanh_van_ban`                                                                                                                                     |
| `needs_expansion`   | `giai_thich_quy_dinh`, `hoi_dap_chung`, `program_goal`, `bao_ve_xa_hoi`, `bao_ton_phat_trien`                                                                                                                    |
| `is_scenario`       | `huong_dan_thu_tuc`, `thu_tuc_hanh_chinh`, `kiem_tra_ho_so`, `xu_ly_vi_pham_hanh_chinh`, `kiem_tra_thanh_tra`, `admin_planning`, `to_chuc_su_kien_cong`, `hoa_giai_van_dong`, `soan_thao_van_ban`, `tao_bao_cao` |


Intent không thuộc `VALID_INTENTS` → cả bốn cờ **false**.

**Đồng bộ chat v2:** `get_rag_intents(query)` → `query_intent.compute_rag_flags_for_query` (bundle + regex multi-doc / checklist / mở rộng nội dung). `get_rag_intents_async` lấy intent từ `detect_intent` (đầy đủ tầng, có LLM) rồi `map_intent_to_rag_flags`.

**Model cục bộ:** trọng số PhoBERT trong `backend/app/intent_model/`; thứ tự lớp **0…22** khớp thứ tự `VALID_INTENTS` trong code. Cấu hình: `INTENT_MODEL_DIR`, `INTENT_MODEL_ENABLED`, `INTENT_MODEL_MIN_CONFIDENCE`, `INTENT_MODEL_DEVICE`, `INTENT_MODEL_MAX_LENGTH` (`config.py` / `.env`).

---

### `query_route_classifier` — chồng lên sau `analyze_query`

Khi `QUERY_UTTERANCE_CLASSIFIER_ENABLED` và có API key, `rag_query` gọi `classify_user_utterance` rồi `merge_utterance_labels_into_analysis(analysis, labels, query=query)`.

- **LLM JSON** trả các trường: `is_legal_or_admin_query`, `is_checklist_catalog_only`, `needs_substantive_legal_answer`, `references_prior_message_context`, `confidence`.
- **Hậu xử lý deterministic:** nếu `query_requires_multi_document_synthesis(query)` → ép không checklist, bật substantive; trong merge: không ép `intent = checklist_documents` nếu multi-doc; cuối merge luôn set `needs_expansion`, `use_multi_article`, `is_legal_lookup` cho multi-doc.
- `**needs_substantive_legal_answer` + confidence ≥ 0.35:** có thể kéo `intent` ra khỏi `checklist_documents`, bật expansion và gợi ý `use_multi_article`.

---

### `VALID_INTENTS` (tóm tắt nhóm)

Nhóm tra cứu / nội dung: `tra_cuu_van_ban`, `article_query`, `trich_xuat_van_ban`, `hoi_dap_chung`  
Metadata & quan hệ: `document_metadata`, `document_relation`, `can_cu_phap_ly`, `program_goal`  
Tác vụ: `tom_tat_van_ban`, `so_sanh_van_ban`, `soan_thao_van_ban`, `tao_bao_cao`, `giai_thich_quy_dinh`, `huong_dan_thu_tuc`, `kiem_tra_ho_so`  
Khác: `admin_planning`  

`**COMMUNE_LEVEL_INTENTS`:** gợi ý `legacy_commune_hint` cho arbiter. **Lưu ý:** câu khớp multi-document synthesis (`query_intent.query_requires_multi_document_synthesis`) **bỏ qua** pipeline cán bộ xã trong `rag_chain_v2` để không nuốt nhánh so sánh/tổng hợp đa văn bản.

### Rẽ nhánh trong `rag_query` (theo `intent`)

- `**checklist_documents`:** template checklist (danh mục văn bản I/II/III), không hybrid multi-article như câu tổng hợp pháp lý.  
- `**document_drafting` / `document_summary`:** draft tool / summarizer.  
- **Các intent còn lại:** hybrid / multi-query retrieve; `needs_expansion` và `use_multi_article` (cùng `USE_MULTI_ARTICLE_FOR_CONDITIONS`) điều khiển mở rộng truy vấn và `single_article_only` / `max_articles` trong retriever.

### `rag_flags` — ý nghĩa vận hành


| Khóa                | Ý nghĩa ngắn                                                                                       |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| `is_legal_lookup`   | Gợi tra cứu điều khoản / lookup (và được bundle bật thêm khi multi-doc synthesis).                 |
| `needs_expansion`   | Cho phép `_multi_query_retrieve` / mở rộng truy vấn thay vì một shot vector đơn.                   |
| `use_multi_article` | Kết hợp với config: cho phép nhiều văn bản (article) trong retrieval & bối cảnh nhóm theo article. |
| `is_scenario`       | Tín hiệu kịch bản hành chính (commune hint, v.v.).                                                 |


---

## Lĩnh vực pháp luật (domain) & nhãn văn bản (`law_intents`)


| Khái niệm                 | Module                                                      | Mô tả                                                                                                                                                                                                     |
| ------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Domain truy vấn**       | `domain_classifier.classify_query_domain`                   | Gán 1–3 nhãn trong `LEGAL_DOMAINS` (semantic + keyword). Dùng lọc vector và so khớp với chunk.                                                                                                            |
| **Domain / nhãn văn bản** | `classify_document_law_intents`, `classify_document_domain` | Mỗi văn bản khi ingest: danh sách `law_intents` (đa nhãn), `legal_domain` = nhãn chính (phần tử đầu). Snippet ưu tiên **tiêu đề + trích yếu** để tránh gán nhãn theo từ khóa lạc ngữ cảnh trong thân văn. |
| **Payload Qdrant**        | `ingestor`                                                  | Mỗi chunk có `legal_domain` và `law_intents` (sau khi migrate + ingest lại hoặc reembed có join `documents`).                                                                                             |


**Retrieval (`hybrid_retriever.hybrid_search`):** merge vector + keyword → rerank → enrich metadata từ PostgreSQL (join `articles`/`documents`; có `law_intents` nếu cột tồn tại). Sau đó: boost điểm nếu `legal_domain` khớp bộ lọc domain; post-filter domain **không** xoá hết kết quả khi metadata lệch; **phạt chủ đề** khi domain câu hỏi (đủ confidence) không giao với `law_intents` hoặc `legal_domain` — **không áp phạt** khi đã resolve số hiệu văn bản từ câu hỏi (`resolved_doc_number`).

---

## Retrieval & RAG v2 (chi tiết kỹ thuật ngắn)

- **Direct DB lookup** (trong `hybrid_search` / `article_lookup`): Điều/Khoản cụ thể; số hiệu văn bản; chủ đề trong luật có tên.  
- **Hybrid:** Qdrant + PostgreSQL → RRF → rerank → enrich → (boost domain / topic penalty) → diversify + chọn 1 hoặc N article.  
- **Keyword:** FTS `articles.search_vector` trong savepoint; lỗi cột → ILIKE trên `vector_chunks`.  
- **Tham số chính:** `RETRIEVAL_TOP_K`, `RERANK_TOP_K`, `MULTI_ARTICLE_MAX_ARTICLES`, `USE_MULTI_ARTICLE_FOR_CONDITIONS`, `TOPIC_MISMATCH_PENALTY`, `TOPIC_MISMATCH_QUERY_CONF_MIN` — xem `config.py`.  
- **Câu hỏi cần tổng hợp pháp lý** (mức phạt, “là gì”, căn cứ…): luôn ưu tiên **LLM** trên ngữ cảnh, không dùng template đa nguồn rỗng.

---

## Cấu trúc thư mục (chính)

```
rag_chatbot/
├── Makefile                 # make test | eval | all
├── docker-compose.yml       # PostgreSQL, Qdrant, Redis
├── backend/
│   ├── main.py
│   ├── alembic/
│   ├── scripts/             # reembed_all.py, fix_doc_numbers.py, …
│   └── app/
│       ├── config.py
│       ├── database/
│       ├── intent_model/    # PhoBERT fine-tuned (tokenizer + weights; git LFS tuỳ team)
│       ├── intent_patterns/ # routing.yaml — structural + routing regex / prototype mở rộng
│       ├── pipeline/        # ingestor, embedding, vector_store, legal_chunker, …
│       ├── retrieval/       # hybrid_retriever, article_lookup, reranker, …
│       ├── routers/         # chat_router, document_router_v2, copilot_router, …
│       ├── services/
│       │   ├── rag_chain_v2.py
│       │   ├── rag_chain.py
│       │   ├── query_intent.py
│       │   ├── commune_route_arbiter.py
│       │   ├── intent_detector.py
│       │   ├── intent_model_classifier.py
│       │   ├── query_understanding.py
│       │   └── …
│       ├── agents/copilot_agent.py
│       ├── tools/
│       └── memory/conversation_store.py
└── frontend/
```

---

## Schema PostgreSQL (rút gọn)

- `**documents**` — `doc_number`, `title`, `issuer`, `issued_date`, `effective_date`, `**law_intents**` (JSON, đa nhãn lĩnh vực; migration `005_law_intents`), …  
- `**chapters` / `sections**` — tuỳ cấu trúc ingest  
- `**articles**` — `article_number`, `title`, `content`, `**search_vector**` (cột generated `tsvector` + GIN — migration `v3_001`; bắt buộc để FTS đầy đủ)  
- `**clauses**` — Khoản/Điểm  
- `**vector_chunks**` — `chunk_text`, `vector_id`, `**chunk_type**` (`article` / `clause` / `token_sub`)  
- `**chat_conversations` / `chat_messages**` — hội thoại lưu (migration `004_chat_conv`)  
- `**chat_logs**` — audit query/answer/latency

Chi tiết: `[backend/app/database/models.py](backend/app/database/models.py)`.

**Sau khi thêm cột `law_intents`:** văn bản đã ingest trước đó có thể để `NULL` cho đến khi ingest lại hoặc script cập nhật; Qdrant cũ có thể thiếu field `law_intents` trong payload — retrieval vẫn dùng `legal_domain` trên chunk và logic phạt chủ đề tương thích.

---

## Cấu hình (`.env` / `config.py`)


| Nhóm                 | Biến tiêu biểu                                                                                                                                                 |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM                  | `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`, `DEFAULT_TEMPERATURE`, `MAX_TOKENS`                                                                       |
| Utterance classifier | `QUERY_UTTERANCE_CLASSIFIER_ENABLED`, `QUERY_UTTERANCE_CLASSIFIER_MODEL`, `QUERY_UTTERANCE_CLASSIFIER_MAX_TOKENS`                                              |
| Intent PhoBERT       | `INTENT_MODEL_ENABLED`, `INTENT_MODEL_DIR`, `INTENT_MODEL_MIN_CONFIDENCE`, `INTENT_MODEL_MAX_LENGTH`, `INTENT_MODEL_DEVICE`                                    |
| Commune arbiter      | `COMMUNE_ROUTE_MARGIN`, `COMMUNE_ROUTE_ARBITER_MODEL`, `COMMUNE_ROUTE_ARBITER_MAX_TOKENS`                                                                      |
| Embedding            | `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_DEVICE`, `EMBEDDING_BATCH_SIZE`                                                                                 |
| Reranker             | `RERANKER_MODEL`, `RERANKER_FALLBACK_MODEL`, `RERANKER_DEVICE`                                                                                                 |
| Retrieval            | `RETRIEVAL_TOP_K`, `RERANK_TOP_K`, `MULTI_ARTICLE_MAX_ARTICLES`, `USE_MULTI_ARTICLE_FOR_CONDITIONS`, `TOPIC_MISMATCH_PENALTY`, `TOPIC_MISMATCH_QUERY_CONF_MIN` |
| Infra                | `POSTGRES_`*, `QDRANT_*`, `REDIS_URL`, `REDIS_CACHE_TTL`                                                                                                       |
| Qdrant               | `QDRANT_RECREATE_ON_DIM_MISMATCH` — đổi chiều vector có thể xóa collection; cần `scripts/reembed_all.py`                                                       |


Mẫu: `[backend/.env.example](backend/.env.example)`.

---

## Cài đặt & chạy

**Yêu cầu:** Python ≥ 3.10, Node ≥ 18, Docker (infra), OpenAI API key; GPU khuyến nghị cho embedding.

```bash
docker compose up -d

cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # điền OPENAI_API_KEY
alembic upgrade head
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Sau khi đổi model embedding / `EMBEDDING_DIM`: xử lý Qdrant + chạy `python scripts/reembed_all.py`.

```bash
cd frontend && npm install && npm run dev
# UI: http://localhost:5173  (proxy /api → backend)
```

**Makefile (gốc repo):** `make test`, `make eval`, `make all`.

---

## API (tóm tắt đang mount trong `main.py`)


| Nhóm                | Endpoint                                                                                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chat production** | `POST /api/chat`, `POST /api/chat/stream`                                                                                                                               |
| **Tài liệu v2**     | `POST /api/upload`, `POST /api/upload-folder`, `GET /api/documents`, `GET /api/documents/{id}`, `GET /api/datasets`, `DELETE /api/datasets/{id}` (`document_router_v2`) |
| **Health / GPU**    | `GET /api/health`, `GET /api/gpu`                                                                                                                                       |
| **Copilot**         | `POST /api/copilot/chat`, `…/stream`                                                                                                                                    |
| **Search**          | `POST /api/search`                                                                                                                                                      |
| **Intent**          | `POST /api/intent`, `GET /api/intent/index-stats`                                                                                                                       |
| **Tools**           | `POST /api/tools/summarize`, `/extract`, `/draft`, `/classify`                                                                                                          |
| **Hội thoại**       | `GET/POST/DELETE /api/conversations`, `GET …/history`                                                                                                                   |
| **Thủ tục**         | `POST /api/procedure/steps`, `/check`, `GET /api/procedure/list`                                                                                                        |
| **Legacy document** | `POST /api/document/summarize`, `/compare`, `POST /api/report/generate` (`document_router`)                                                                             |


---

## Ingest tài liệu (upload)

File `.docx` hoặc `.doc` (Windows: COM Word) → parse → clean → detect cấu trúc → chunk (Điều / Khoản / sub-chunk) → PostgreSQL + embed → Qdrant. Sau upload: prototype intent tự bổ sung (nếu bật), invalidate cache.

---

## Prompt chính (`config.py`)


| Key                                              | Dùng cho                  |
| ------------------------------------------------ | ------------------------- |
| `SYSTEM_PROMPT_V2`, `RAG_PROMPT_TEMPLATE_V2`     | `rag_chain_v2` (mặc định) |
| `COMMUNE_OFFICER_*`                              | Pipeline cán bộ cấp xã    |
| `RAG_PROMPT_TEMPLATE`, `QUERY_REWRITE_PROMPT`, … | `rag_chain.py` / Copilot  |


---

## Vận hành & frontend thực tế

- Infra (Postgres, Qdrant, Redis) phải chạy trước backend.  
- Migration: `alembic upgrade head`; DB cần quyền tạo extension (`unaccent`, `pg_trgm`). Cần đủ chuỗi revision để có `articles.search_vector` và `documents.law_intents`; nếu thiếu cột, backend vẫn cố gắng chạy (FTS/ILIKE, enrich không `law_intents`) nhưng nên migrate production.  
- **Frontend chat** dùng `**/api/chat/stream`** — **không** dùng `/api/copilot/chat/stream`.  
- `conversation_id` đã hỗ trợ ở API; UI có thể chưa gắn đầy đủ — gửi kèm để multi-turn.  
- Redis lỗi → cache tắt im lặng (graceful degradation).  
- File `.doc` chỉ tin cậy trên Windows + Word.

---

## Kiểm thử

```bash
cd backend
pytest
# hoặc từ gốc repo: make test
```

Thư mục: `backend/tests/`, `tests/evaluation/` (tuỳ cấu hình Makefile).

---

## Đánh giá toàn diện (retrieval, E2E, A/B `query_route_classifier`)

Ngoài EDA intent + RAG flags, repo có thêm bộ đánh giá để tách bottleneck **retrieval** vs **generation** và đo tác động lớp **LLM JSON** (`QUERY_UTTERANCE_CLASSIFIER_ENABLED` / `query_route_classifier`).

| Thành phần | Mục đích |
|------------|----------|
| **Retrieval** | Precision@K / Recall@K trên đầu ra `hybrid_search`: nhãn gold theo `relevant_vector_ids` (chunk) hoặc `relevant_article_ids` (article) trong `gold_comprehensive.json`. |
| **End-to-end** | LLM judge (JSON): **relevance**, **faithfulness** (bám ngữ cảnh truy vấn được), **completeness** (so với `reference_answer`), thang 1–5. |
| **A/B** | Hai lần `rag_chain_v2.rag_query` trên cùng câu hỏi: bật/tắt classifier (patch `QUERY_UTTERANCE_CLASSIFIER_ENABLED`), tắt Redis cache trong run; so **latency** và điểm judge. |

**File chính**

- `backend/tests/evaluation/gold_comprehensive.json` — **500 câu** (không trùng) lấy từ `data_clean.json` ở gốc repo; mỗi dòng có `gold_intent` (tham chiếu), `reference_answer` rubric tự động; điền `relevant_*` nếu cần P@K/R@K có nhãn.
- `backend/tests/evaluation/gold_comprehensive_smoke.json` — 3 câu ngắn để thử pipeline.
- `backend/tests/evaluation/build_gold_comprehensive.py` — tái tạo gold từ JSONL (`--count`, `--data`, `--output`).
- `backend/tests/evaluation/retrieval_metrics.py` — định nghĩa P@K / R@K (chunk + article).
- `backend/tests/evaluation/eval_comprehensive.py` — runner + báo cáo JSON/Markdown + `retrieval_volume` (mean/median số chunk @K=10).
- `backend/tests/evaluation/test_retrieval_metrics.py` — unit test metrics (không cần DB).
- `backend/tests/evaluation/test_eval_comprehensive.py` — tích hợp đầy đủ khi `EVAL_COMPREHENSIVE=1` (mặc định chỉ 5 câu qua `EVAL_COMPREHENSIVE_MAX`).

**Lệnh**

```bash
# Tạo lại gold 500 câu (cần data_clean.json ở thư mục gốc rag_chatbot)
cd backend
python tests/evaluation/build_gold_comprehensive.py --count 500
# hoặc: make build-gold-comprehensive

# Unit + smoke evaluation (mặc định CI)
python -m pytest tests/evaluation/test_retrieval_metrics.py tests/evaluation/eval_retrieval.py -q

# Chỉ retrieval trên toàn bộ gold (500 câu — có thể chạy lâu, ~vài giây/câu sau khi model đã nạp)
python tests/evaluation/eval_comprehensive.py --skip-rag --max-samples 500

# Giới hạn N câu đầu (debug)
python tests/evaluation/eval_comprehensive.py --skip-rag --max-samples 50

# Báo cáo đầy đủ (Postgres, Qdrant, OpenAI) — ghi results/comprehensive_eval.json và .md
python tests/evaluation/eval_comprehensive.py
# hoặc từ gốc repo: make eval-comprehensive

# Tích hợp pytest (cùng stack + ghi temp dir; chỉ vài câu)
set EVAL_COMPREHENSIVE=1
set EVAL_COMPREHENSIVE_MAX=5
python -m pytest tests/evaluation/test_eval_comprehensive.py -q
```

**Kết quả (artifact):** [backend/tests/evaluation/results/comprehensive_eval.md](backend/tests/evaluation/results/comprehensive_eval.md), [backend/tests/evaluation/results/comprehensive_eval.json](backend/tests/evaluation/results/comprehensive_eval.json).

- **P@K / R@K (mean):** chỉ khác 0 khi đã gán `relevant_vector_ids` / `relevant_article_ids` trong gold.
- **`retrieval_volume`:** sau mỗi lần chạy retrieval, JSON/Markdown có thêm khối thống kê — số truy vấn, lỗi retrieval, **mean/median/min/max** `num_retrieved` @K=10, số câu trả về 0 chunk (không cần nhãn gold).
- **Smoke 3 câu:** dùng `--gold tests/evaluation/gold_comprehensive_smoke.json` nếu cần so sánh nhanh với bản cũ.

**Khi chạy không `--skip-rag`:** với 500 câu, chi phí LLM và thời gian rất lớn; nên giữ `--skip-rag` cho benchmark retrieval hoặc dùng `--max-samples` nhỏ cho E2E/A/B. Cần `OPENAI_API_KEY` cho judge và classifier LLM khi bật.

---

## EDA Intent + RAG Flags (data_clean.json)

Đã chạy evaluator trên `data_clean.json` bằng test:

`backend/tests/evaluation/test_intent_ragflags_datajson.py`

Ghi chú run này: đặt `INTENT_MODEL_ENABLED=false` để đánh giá trực tiếp hiệu quả rule/pattern sau khi cập nhật intent.

Kết quả (latest run, sau khi cập nhật lại rule cho từng intent):

- Samples: **5744**
- Detector intent accuracy: **39.62%**
- Routing intent accuracy: **38.74%**
- RAG flags exact-match accuracy: **45.37%**
- Detector macro F1: **35.49%**

RAG flag accuracy:

| Flag | Accuracy |
|---|---:|
| `is_legal_lookup` | 89.17% |
| `needs_expansion` | 53.12% |
| `use_multi_article` | 88.37% |
| `is_scenario` | 72.68% |

Chi tiết theo intent:

| Intent | Support | Predicted | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| `tra_cuu_van_ban` | 675 | 534 | 50.19% | 39.70% | 44.33% |
| `huong_dan_thu_tuc` | 634 | 101 | 62.38% | 9.94% | 17.14% |
| `article_query` | 632 | 1323 | 47.47% | 99.37% | 64.25% |
| `giai_thich_quy_dinh` | 505 | 324 | 50.31% | 32.28% | 39.32% |
| `hoi_dap_chung` | 400 | 1250 | 16.16% | 50.50% | 24.48% |
| `xu_ly_vi_pham_hanh_chinh` | 312 | 127 | 23.62% | 9.62% | 13.67% |
| `trich_xuat_van_ban` | 270 | 157 | 99.36% | 57.78% | 73.07% |
| `so_sanh_van_ban` | 228 | 139 | 66.91% | 40.79% | 50.68% |
| `tom_tat_van_ban` | 225 | 4 | 100.00% | 1.78% | 3.49% |
| `document_relation` | 218 | 156 | 57.05% | 40.83% | 47.59% |
| `bao_ton_phat_trien` | 194 | 933 | 15.54% | 74.74% | 25.73% |
| `to_chuc_su_kien_cong` | 167 | 87 | 37.93% | 19.76% | 25.98% |
| `soan_thao_van_ban` | 160 | 99 | 89.90% | 55.62% | 68.73% |
| `document_metadata` | 157 | 9 | 100.00% | 5.73% | 10.84% |
| `tao_bao_cao` | 132 | 113 | 82.30% | 70.45% | 75.92% |
| `kiem_tra_thanh_tra` | 130 | 56 | 87.50% | 37.69% | 52.69% |
| `thu_tuc_hanh_chinh` | 118 | 13 | 0.00% | 0.00% | 0.00% |
| `hoa_giai_van_dong` | 113 | 27 | 96.30% | 23.01% | 37.14% |
| `can_cu_phap_ly` | 101 | 51 | 96.08% | 48.51% | 64.47% |
| `bao_ve_xa_hoi` | 100 | 169 | 18.34% | 31.00% | 23.05% |
| `kiem_tra_ho_so` | 79 | 11 | 100.00% | 13.92% | 24.44% |
| `program_goal` | 78 | 61 | 73.77% | 57.69% | 64.75% |
| `admin_planning` | 70 | 0 | 0.00% | 0.00% | 0.00% |

Top confusion chính:

- `huong_dan_thu_tuc` -> `hoi_dap_chung`: 305
- `tra_cuu_van_ban` -> `article_query`: 154
- `trich_xuat_van_ban` -> `article_query`: 114
- `xu_ly_vi_pham_hanh_chinh` -> `bao_ton_phat_trien`: 98
- `giai_thich_quy_dinh` -> `hoi_dap_chung`: 90

Artifacts chi tiết:

- `backend/tests/evaluation/results/intent_ragflags_eda_data_clean.md`
- `backend/tests/evaluation/results/intent_ragflags_eda_data_clean.json`

Lệnh chạy lại:

```bash
cd rag_chatbot
backend\venv\Scripts\python.exe -m pytest backend/tests/evaluation/test_intent_ragflags_datajson.py -q
```

---

## Ghi chú lịch sử (v1 → v2)

Stack cũ (SQLite, FAISS, langchain đồng bộ) đã **loại bỏ**. `services/retrieval.py` bọc `hybrid_search` cho code gọi kiểu adapter; chat production không đi qua FAISS.

**Bổ sung gần đây (v2):** nhãn đa lĩnh vực theo văn bản (`law_intents`), phạt chủ đề khi domain truy vấn lệch metadata chunk, cấu hình intent tách `intent_patterns/routing.yaml`, pipeline intent async/documented đồng bộ với `intent_detector`, hội thoại lưu DB (`chat_conversations`), rollback PostgreSQL khi enrich thiếu cột.

---

## Ví dụ nhanh

**Tra cứu Điều luật:**  
`Điều 47 Luật Di sản văn hóa quy định gì?` → ưu tiên direct lookup + RAG.

**Tình huống cán bộ (sau arbiter):**  
Câu hỏi thao tác địa phương, thủ tục xã — có thể vào template 5 phần; câu chỉ hỏi **mức phạt / điều khoản** thường ở lại RAG pháp lý.

**curl chat**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Quy định về cấp phép kinh doanh karaoke?\", \"temperature\": 0.3}"
```

