"""Ninh Bình Web Search Pipeline – modular, structured, multi-source.

Pipeline:  entity_resolver → query_rewriter → web_search → info_extractor → response_generator

Features:
- Entity resolution against the Ninh Bình administrative hierarchy
- Multi-query rewriting (Vietnamese + English variants)
- Domain-filtered, multi-source retrieval (priority gov sites → Wikipedia → general web)
- Structured 6-field information extraction via LLM
- In-memory result cache with TTL (1–6 h)
- Fallback strategy with progressive broadening
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────

WEB_SEARCH_MAX_RESULTS = 8
CACHE_TTL_SECONDS = 3600  # 1 hour default

TRUSTED_DOMAINS = [
    "ninhbinh.gov.vn",
    "baoninhbinh.org.vn",
    "dulichninhbinh.com.vn",
    "xaydungchinhsach.chinhphu.vn",
]

BLOCKED_DOMAINS = [
    "blogspot.com",
    "wordpress.com",
    "weebly.com",
]

INSUFFICIENT_MARKERS = [
    "khong co thong tin",
    "khong du thong tin",
    "khong tim thay thong tin",
    "khong the tra loi",
    "khong the cung cap",
]

# ── Result cache ───────────────────────────────────────────

_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _cache_key(query: str) -> str:
    return _normalize_text(query)


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, data = entry
    if time.time() - ts > CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    return data


def _cache_set(key: str, data: Dict[str, Any]) -> None:
    if len(_cache) > 500:
        oldest_key = min(_cache, key=lambda k: _cache[k][0])
        del _cache[oldest_key]
    _cache[key] = (time.time(), data)


# ── Text utilities ─────────────────────────────────────────

def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("đ", "d").replace("Đ", "D")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def _strip_inline_citations(text: str) -> str:
    text = re.sub(r"\(\[.*?\]\(https?://[^\)]*\)\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\(https?://[^\)]*\)", r"\1", text)
    text = re.sub(r"\(https?://[^\)]*\)", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _clean_answer(text: str) -> str:
    raw = _strip_inline_citations((text or "").strip())
    return raw if raw else ""


def _is_ninh_binh_context(text: str) -> bool:
    return "ninh binh" in _normalize_text(text)


def _looks_like_non_ninh_binh_answer(text: str) -> bool:
    t = _normalize_text(text)
    other_provinces = ["khanh hoa", "van ninh", "ha noi", "da nang", "ho chi minh", "hai phong"]
    for bad in other_provinces:
        if bad in t and "ninh binh" not in t:
            return True
    for m in re.finditer(r"tinh\s+([a-z0-9\s]{2,32})", t):
        prov = m.group(1).strip()
        if "ninh binh" not in prov:
            return True
    return False


def _is_blocked_domain(url: str) -> bool:
    url_lower = (url or "").lower()
    return any(d in url_lower for d in BLOCKED_DOMAINS)


def _is_trusted_domain(url: str) -> bool:
    url_lower = (url or "").lower()
    return any(d in url_lower for d in TRUSTED_DOMAINS)


def _source_quality_score(source: Dict[str, Any]) -> float:
    """Score a source by domain trust. Trusted=2.0, neutral=1.0, blocked=0."""
    url = source.get("url", "")
    if _is_blocked_domain(url):
        return 0.0
    if _is_trusted_domain(url):
        return 2.0
    return 1.0


# ── Wikipedia search (kept for offline/fallback) ──────────

def _get_wikipedia_client():
    try:
        import wikipedia  # type: ignore
        return wikipedia
    except Exception:
        return None


def _search_wikipedia_sync(
    search_queries: List[str],
    entity_norm: str,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
) -> List[Dict[str, Any]]:
    """Search Wikipedia with multiple query variants."""
    wikipedia = _get_wikipedia_client()
    if wikipedia is None:
        return []

    try:
        all_titles: List[str] = []
        for lang in ("vi", "en"):
            wikipedia.set_lang(lang)
            for qv in search_queries[:4]:
                try:
                    for t in wikipedia.search(qv, results=max_results):
                        if t not in all_titles:
                            all_titles.append(t)
                except Exception:
                    continue

        entity_terms = [t for t in re.split(r"[^a-z0-9]+", entity_norm) if t and len(t) >= 2]
        rows: List[Dict[str, Any]] = []

        for lang in ("vi", "en"):
            wikipedia.set_lang(lang)
            for title in all_titles:
                if len(rows) >= max_results:
                    break
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    content = (page.content or "").strip()
                    if not content:
                        continue
                    haystack = _normalize_text(f"{page.title} {content}")
                    if not _is_ninh_binh_context(haystack):
                        continue
                    if entity_terms and not all(t in haystack for t in entity_terms):
                        continue
                    rows.append({
                        "title": page.title,
                        "text": content[:4000],
                        "url": page.url,
                        "source_type": "wikipedia",
                    })
                except Exception:
                    continue
            if len(rows) >= max_results:
                break
        return rows
    except Exception as e:
        log.warning("Wikipedia search failed: %s", e)
        return []


# ── OpenAI web search (priority sites + general) ──────────

async def _search_openai_web(
    query: str,
    site_filter: str = "",
    max_retries: int = 1,
) -> Dict[str, Any]:
    """Call OpenAI web_search with optional site filter."""
    from app.tools.openai_web_search_tool import get_openai_web_search_tool

    tool = get_openai_web_search_tool()
    augmented = query
    if site_filter:
        augmented = f"{query} ({site_filter})"

    result = await tool.search_web(augmented)
    answer = _clean_answer(result.get("answer", ""))
    sources = result.get("sources", [])

    # Filter out blocked domains
    sources = [s for s in sources if not _is_blocked_domain(s.get("url", ""))]

    return {"answer": answer, "sources": sources}


async def _search_priority_sites(queries: List[str]) -> Dict[str, Any]:
    """Search trusted government sites first."""
    site_filter = " OR ".join(f"site:{d}" for d in TRUSTED_DOMAINS[:3])
    primary_query = queries[0] if queries else ""

    search_query = (
        f"{primary_query} tỉnh Ninh Bình\n\n"
        f"Ưu tiên tìm kiếm trên: {', '.join(TRUSTED_DOMAINS[:3])}.\n"
        f"Trả lời bằng tiếng Việt, đầy đủ, chính xác."
    )
    result = await _search_openai_web(search_query, site_filter=site_filter)

    has_trusted = any(_is_trusted_domain(s.get("url", "")) for s in result.get("sources", []))
    answer = result.get("answer", "")

    if has_trusted and answer and len(answer) > 40:
        log.info("[NINH_BINH] Priority site search succeeded")
        return result
    return {"answer": "", "sources": []}


async def _search_general_web(queries: List[str], entity_display: str) -> Dict[str, Any]:
    """General OpenAI web search with Ninh Bình context."""
    primary_query = queries[0] if queries else ""
    context = (
        f"Ngữ cảnh: {entity_display}, tỉnh Ninh Bình (Việt Nam).\n"
        f"Nguồn ưu tiên: {', '.join(TRUSTED_DOMAINS[:3])}.\n"
        "BẮT BUỘC: chỉ dùng dữ liệu thuộc Ninh Bình. "
        "Nếu không chắc chắn, nói rõ thay vì suy đoán.\n"
        "Trả lời bằng tiếng Việt, đầy đủ, chính xác."
    )
    search_query = f"{primary_query}\n\n{context}"
    return await _search_openai_web(search_query)


# ── Multi-source retrieval orchestrator ────────────────────

async def _retrieve_multi_source(
    search_queries: List[str],
    entity_norm: str,
    entity_display: str,
    needs_freshness: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Retrieve from multiple sources and return (source_contents, source_refs).

    Strategy:
    1. Priority trusted sites (OpenAI web_search with site: filter)
    2. Wikipedia (good for stable facts)
    3. General OpenAI web_search (broadest coverage)
    """
    all_contents: List[Dict[str, Any]] = []
    all_sources: List[Dict[str, Any]] = []
    seen_urls: set = set()

    def _add(items: List[Dict[str, Any]], is_content: bool = True) -> None:
        for item in items:
            url = item.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            if is_content:
                all_contents.append(item)
            else:
                all_sources.append(item)

    # Step 1: Priority sites
    priority_result = await _search_priority_sites(search_queries)
    if priority_result.get("answer"):
        _add([{
            "title": "Nguồn chính thức",
            "text": priority_result["answer"],
            "url": (priority_result.get("sources", [{}])[0].get("url", "") if priority_result.get("sources") else ""),
        }])
        _add(priority_result.get("sources", []), is_content=False)

    # Step 2: Wikipedia (skip if freshness needed and we already have priority results)
    if not (needs_freshness and all_contents):
        wiki_results = _search_wikipedia_sync(search_queries, entity_norm)
        _add(wiki_results)
        for wr in wiki_results:
            _add([{"title": wr.get("title", ""), "url": wr.get("url", "")}], is_content=False)

    # Step 3: General web search (always run to ensure coverage)
    if len(all_contents) < 3:
        general_result = await _search_general_web(search_queries, entity_display)
        if general_result.get("answer"):
            _add([{
                "title": "Tìm kiếm web",
                "text": general_result["answer"],
                "url": "",
            }])
            _add(general_result.get("sources", []), is_content=False)

    # Sort sources by trust score
    all_sources.sort(key=_source_quality_score, reverse=True)

    return all_contents[:5], all_sources[:10]


# ── Main pipeline ──────────────────────────────────────────

async def search_ninh_binh(query: str) -> Dict[str, Any]:
    """Full pipeline: entity_resolver → query_rewriter → web_search → info_extractor → response.

    Returns: {
        "answer": str (human-readable Vietnamese),
        "sources": [...],
        "structured": {"entity": ..., "level": ..., "data": {...}} (optional),
    }
    """
    # Check cache
    ckey = _cache_key(query)
    cached = _cache_get(ckey)
    if cached:
        log.info("[NINH_BINH] Cache hit for query")
        return cached

    # Step 1: Entity resolution
    from app.services.ninh_binh_entity_resolver import (
        resolve_entity,
        get_entity_display_name,
    )
    entity = resolve_entity(query)
    entity_display = get_entity_display_name(entity)
    entity_norm = _normalize_text(entity.name)

    log.info(
        "[NINH_BINH] Entity: %s (%s) confidence=%.2f, queries=%d",
        entity.name, entity.level, entity.confidence, len(entity.search_queries),
    )

    # Step 2: Multi-source retrieval
    contents, sources = await _retrieve_multi_source(
        search_queries=entity.search_queries,
        entity_norm=entity_norm,
        entity_display=entity_display,
        needs_freshness=_needs_fresh_data(query),
    )

    # Step 3: Structured information extraction
    from app.services.ninh_binh_info_extractor import (
        extract_structured_info,
        generate_human_answer,
        build_structured_result,
    )

    structured_data = await extract_structured_info(
        sources_content=contents,
        entity_display=entity_display,
        level=entity.level,
        original_query=query,
    )

    # Step 4: Generate human-readable answer
    human_answer = await generate_human_answer(
        structured_data=structured_data,
        entity_display=entity_display,
        original_query=query,
    )

    # Post-processing: validate Ninh Bình context
    if _looks_like_non_ninh_binh_answer(human_answer):
        log.warning("[NINH_BINH] Answer contains non-Ninh-Binh data, retrying...")
        human_answer = await _retry_with_strict_context(query, entity_display, contents)

    structured_output = build_structured_result(
        entity_name=entity.name,
        entity_level=entity.level,
        data=structured_data,
        human_answer=human_answer,
        sources=sources,
        confidence=entity.confidence,
    )

    result = {
        "answer": human_answer,
        "sources": sources,
        "structured": structured_output,
    }

    _cache_set(ckey, result)
    return result


async def _retry_with_strict_context(
    query: str,
    entity_display: str,
    contents: List[Dict[str, Any]],
) -> str:
    """Retry answer generation with stricter Ninh Bình constraints."""
    from app.services.llm_client import generate

    content_text = "\n\n".join(c.get("text", "")[:2000] for c in contents[:3])
    prompt = (
        f"Dựa trên nguồn dữ liệu sau, trả lời câu hỏi về {entity_display}, tỉnh Ninh Bình.\n\n"
        f"NGUỒN:\n{content_text}\n\n"
        f"CÂU HỎI: {query}\n\n"
        "BẮT BUỘC: Chỉ sử dụng thông tin thuộc tỉnh Ninh Bình. "
        "Nếu không có dữ liệu chính xác → nói 'Chưa có dữ liệu rõ ràng'. "
        "Trả lời bằng tiếng Việt."
    )
    try:
        answer = await generate(prompt=prompt, system="", temperature=0.2)
        return _clean_answer(answer or "Chưa có dữ liệu rõ ràng cho câu hỏi này.")
    except Exception:
        return "Không tìm thấy thông tin phù hợp tại Ninh Bình cho câu hỏi này."


# ── Freshness detection ────────────────────────────────────

_FRESHNESS_HINTS = [
    "mới nhất", "moi nhat", "hiện nay", "hien nay", "hiện tại", "hien tai",
    "cập nhật", "cap nhat", "năm nay", "nam nay", "sáp nhập", "sap nhap",
]


def _needs_fresh_data(query: str) -> bool:
    q_norm = _normalize_text(query)
    if any(h in q_norm for h in _FRESHNESS_HINTS):
        return True
    current_year = datetime.now().year
    return str(current_year) in q_norm or str(current_year - 1) in q_norm


# ── Legacy compatibility ───────────────────────────────────
# Keep these for any code that still imports them directly

def _is_wikipedia_insufficient(result: Dict[str, Any]) -> bool:
    """Check if Wikipedia result is too thin to use as final answer."""
    if not isinstance(result, dict):
        return True
    answer = (result.get("answer") or "").strip()
    sources = result.get("sources") or []
    if not answer or len(answer) < 45:
        return True
    answer_norm = _normalize_text(answer)
    if any(marker in answer_norm for marker in INSUFFICIENT_MARKERS):
        return True
    return len(sources) == 0


async def search_wikipedia(query: str, query_ctx: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Legacy wrapper, still usable by other modules."""
    entity_norm = _normalize_text(query)
    results = _search_wikipedia_sync([query, f"{query} Ninh Bình"], entity_norm)
    sources = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results]
    if not results:
        return {"answer": "Không tìm thấy trên Wikipedia.", "sources": sources}

    from app.services.llm_client import generate
    context = "\n\n---\n\n".join(
        f"[{i}] {r.get('title', '')}\n{r.get('text', '')}" for i, r in enumerate(results, 1)
    )
    prompt = f"Kết quả Wikipedia:\n\n{context}\n\n---\n\nCâu hỏi: {query}\n\nTrả lời đầy đủ bằng tiếng Việt."
    answer = await generate(prompt=prompt, system="Trả lời dựa trên nguồn Wikipedia.", temperature=0.2)
    return {"answer": _clean_answer(answer), "sources": sources}
