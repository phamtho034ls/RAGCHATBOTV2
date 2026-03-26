"""Chọn pipeline cán bộ xã (COMMUNE_OFFICER) vs RAG thường — semantic prototype + margin + LLM khi mơ hồ."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Tuple

import numpy as np

from app.config import (
    COMMUNE_ROUTE_ARBITER_MAX_TOKENS,
    COMMUNE_ROUTE_ARBITER_MODEL,
    COMMUNE_ROUTE_MARGIN,
    OPENAI_API_KEY,
)
from app.pipeline.embedding import embed_texts

log = logging.getLogger(__name__)

# Câu mẫu: thủ tục / vai trò cán bộ địa phương xử lý vụ việc (KHÔNG phải tra cứu mức phạt khô).
_COMMUNE_PROCEDURAL_PROTOTYPES: Tuple[str, ...] = (
    "UBND phường xã tiếp nhận phản ánh của người dân và tổ chức xác minh hiện trường trong thời hạn quy định",
    "Cán bộ văn hóa xã phối hợp với công an lập biên bản vi phạm hành chính đối với cơ sở kinh doanh",
    "Tổ liên ngành kiểm tra đột xuất quán karaoke trên địa bàn và lập kế hoạch xử lý theo thẩm quyền",
    "Hội đồng xử phạt vi phạm hành chính cấp xã họp xét mức phạt và ra quyết định xử phạt",
    "Chủ tịch UBND ký quyết định xử phạt sau khi có biên bản và hồ sơ đầy đủ từ công chức chuyên môn",
    "Công chức địa phương hướng dẫn các bước tiếp nhận tin báo lễ hội trái phép và báo cáo cấp trên",
    "Phối hợp giữa phòng văn hóa và trật tự đô thị trong xử lý vi phạm quảng cáo ngoài trời tại xã",
    "Lập hồ sơ chuyển cơ quan có thẩm quyền khi vượt quá thẩm quyền xử phạt của UBND cấp xã",
    "Tổ công tác vận động nhắc nhở hộ kinh doanh chấp hành quy định giờ hoạt động karaoke",
    "Báo cáo định kỳ tình hình vi phạm trật tự công cộng lên UBND huyện theo quy định",
)

# Câu mẫu: tra cứu pháp luật, mức phạt, văn bản, so sánh điều khoản (không yêu cầu quy trình hành chính xã).
_LEGAL_QA_PROTOTYPES: Tuple[str, ...] = (
    "Mức phạt tiền đối với hành vi kinh doanh karaoke quá giờ quy định là bao nhiêu",
    "Các quy định về kinh doanh karaoke được quy định trong những luật nghị định thông tư nào",
    "Điều 15 Nghị định quy định mức xử phạt cụ thể như thế nào",
    "So sánh trách nhiệm của UBND cấp xã trong phòng chống bạo lực gia đình theo các văn bản pháp luật",
    "Đối chiếu quy định về quảng cáo ngoài trời giữa Luật Quảng cáo và các nghị định hướng dẫn",
    "Luật nào quy định về di sản văn hóa phi vật thể và điều kiện công nhận",
    "Giải thích nội dung khoản 2 Điều 6 theo văn bản đã trích dẫn",
    "Căn cứ pháp lý để xử phạt hành vi quảng cáo sai sự thật là gì",
    "Thủ tục cấp phép tổ chức lễ hội gồm những giấy tờ gì theo quy định hiện hành",
    "Khác biệt giữa thẩm quyền của sở và UBND tỉnh trong lĩnh vực văn hóa",
)

_commune_mat: Optional[np.ndarray] = None  # (n_c, d) L2-normalized rows
_legal_mat: Optional[np.ndarray] = None  # (n_l, d)


def warmup_commune_route_index() -> None:
    """Embed prototypes (gọi sau warmup_embeddings)."""
    global _commune_mat, _legal_mat
    try:
        c = embed_texts(list(_COMMUNE_PROCEDURAL_PROTOTYPES))
        l_ = embed_texts(list(_LEGAL_QA_PROTOTYPES))
        if c.size == 0 or l_.size == 0:
            log.warning("commune_route_arbiter: empty embeddings")
            return
        _commune_mat = np.asarray(c, dtype=np.float32)
        _legal_mat = np.asarray(l_, dtype=np.float32)
        log.info(
            "Commune route index: %d procedural + %d legal QA prototypes",
            _commune_mat.shape[0],
            _legal_mat.shape[0],
        )
    except Exception as exc:
        log.error("warmup_commune_route_index failed: %s", exc)
        _commune_mat = None
        _legal_mat = None


def semantic_commune_vs_legal_scores(query: str) -> Tuple[float, float]:
    """Cosine max (đã normalize) tới từng cụm prototype. Trả (score_commune, score_legal)."""
    q = (query or "").strip()
    if not q or _commune_mat is None or _legal_mat is None:
        return 0.0, 0.0
    qv = embed_texts([q])
    if qv.size == 0:
        return 0.0, 0.0
    v = np.asarray(qv[0], dtype=np.float32).reshape(1, -1)
    s_c = float((_commune_mat @ v.T).max())
    s_l = float((_legal_mat @ v.T).max())
    return s_c, s_l


def _parse_arbiter_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


async def _llm_arbitrate_commune_route(query: str, *, legacy_commune_hint: bool) -> bool:
    """Khi margin thấp: một lần gọi LLM, trả JSON use_commune_officer: bool."""
    if not OPENAI_API_KEY or not (query or "").strip():
        log.warning("commune_route LLM arbiter skipped (no API key or empty query) → default legal RAG")
        return False

    from app.services.llm_client import generate

    hint = ""
    if legacy_commune_hint:
        hint = (
            "\n(Gợi ý phụ: bộ phân loại cũ có tín hiệu 'tình huống cấp xã', nhưng thường nhầm với tra cứu pháp luật. "
            "Chỉ true nếu câu thật sự yêu cầu quy trình/thao tác cán bộ địa phương.)\n"
        )

    user = (
        "Phân loại MỘT câu hỏi tiếng Việt.\n"
        '- Trả về use_commune_officer = true CHỈ KHI người hỏi cần **hướng dẫn thủ tục hành chính cho cán bộ xã/phường** '
        "(các bước tiếp nhận, xác minh, lập biên bản, phối hợp liên ngành, báo cáo, xử lý vụ việc **tại địa bàn**).\n"
        "- use_commune_officer = false nếu câu là **tra cứu pháp luật thông thường**: mức phạt bao nhiêu, điều khoản nào, "
        "liệt kê luật/nghị định nào, so sánh/đối chiếu nội dung văn bản, giải thích quy định cho người dân.\n"
        "Nếu phân vân giữa hai loại, chọn use_commune_officer = false.\n"
        f"{hint}\n"
        f'Câu hỏi: """{query.strip()[:3500]}"""\n\n'
        'Trả lời CHỈ một JSON, không markdown: {"use_commune_officer": true hoặc false}'
    )

    try:
        raw = await generate(
            prompt=user,
            system="Bạn chỉ trả về một object JSON hợp lệ, không giải thích thêm.",
            temperature=0.0,
            model=COMMUNE_ROUTE_ARBITER_MODEL,
            max_tokens=COMMUNE_ROUTE_ARBITER_MAX_TOKENS,
        )
        data = _parse_arbiter_json(raw or "")
        if data is None:
            raise ValueError("no JSON object in arbiter response")
        val = data.get("use_commune_officer")
        if isinstance(val, bool):
            log.info("commune_route LLM arbiter → use_commune_officer=%s", val)
            return val
        if isinstance(val, str) and val.lower() in ("true", "false"):
            return val.lower() == "true"
    except Exception as exc:
        log.warning("commune_route LLM arbiter failed: %s → default legal RAG", exc)
    return False


async def resolve_use_commune_officer_pipeline(
    query: str,
    *,
    legacy_commune_hint: bool = False,
) -> bool:
    """Quyết định có dùng pipeline COMMUNE_OFFICER (template 5 mục) hay không."""
    q = (query or "").strip()
    if len(q) < 4:
        return False

    s_c, s_l = semantic_commune_vs_legal_scores(q)
    diff = s_c - s_l
    m = COMMUNE_ROUTE_MARGIN

    log.info(
        "commune_route semantic: s_commune=%.4f s_legal=%.4f diff=%.4f margin=%.4f legacy_hint=%s",
        s_c,
        s_l,
        diff,
        m,
        legacy_commune_hint,
    )

    if diff >= m:
        return True
    if -diff >= m:
        return False

    return await _llm_arbitrate_commune_route(q, legacy_commune_hint=legacy_commune_hint)
