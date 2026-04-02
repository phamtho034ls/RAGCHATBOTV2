"""
Tool: Soạn thảo công văn / văn bản hành chính.

Tìm tài liệu tham khảo qua RAG → LLM soạn văn bản theo thể thức chuẩn.
Hỗ trợ: kế hoạch, quyết định, thông báo, báo cáo, công văn, tờ trình.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.services.llm_client import generate
from app.services.retrieval import search_all, format_sources

log = logging.getLogger(__name__)

_DOC_NUMBER_RE = re.compile(r"\b(\d+[/_]\d{4}[/_][A-ZĐa-zđ\-]+)\b")

DRAFT_SYSTEM = (
    "Bạn là chuyên gia soạn thảo văn bản hành chính nhà nước Việt Nam. "
    "Hãy soạn thảo văn bản theo đúng thể thức văn bản hành chính. "
    "CHỈ ĐƯỢC ghi và trích dẫn căn cứ pháp lý, số hiệu văn bản từ TÀI LIỆU THAM KHẢO được cung cấp trong câu trả lời. "
    "TUYỆT ĐỐI KHÔNG bịa đặt số hiệu văn bản pháp luật từ kiến thức bên ngoài. "
    "Phần căn cứ pháp lý phải nêu rõ nội dung Điều/Khoản liên quan, không chỉ liệt kê tên văn bản."
)

DRAFT_PROMPT = """Hãy soạn thảo văn bản hành chính theo yêu cầu sau:

YÊU CẦU: {request}

TÀI LIỆU THAM KHẢO (từ cơ sở dữ liệu):
{context}

BẮT BUỘC tuân thủ thể thức văn bản hành chính Việt Nam:

1. QUỐC HIỆU, TIÊU NGỮ:
   CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
   Độc lập – Tự do – Hạnh phúc
   ────────────────────

2. TÊN CƠ QUAN:
   [TÊN CƠ QUAN CHỦ QUẢN]
   [TÊN CƠ QUAN BAN HÀNH]

3. SỐ, KÝ HIỆU VĂN BẢN:
   Số: .../[Năm]/[Loại VB]-[Tên CQ]

4. ĐỊA DANH, NGÀY THÁNG NĂM:
   [Địa danh], ngày ... tháng ... năm ...

5. TÊN LOẠI VĂN BẢN VÀ TRÍCH YẾU:
   [TÊN LOẠI VĂN BẢN]
   [Trích yếu nội dung]

6. NỘI DUNG CHÍNH:
   Phải đầy đủ, rõ ràng, phù hợp với loại văn bản.
   - Kế hoạch: Mục đích, yêu cầu, nội dung, tổ chức thực hiện
   - Quyết định: Căn cứ pháp lý, các điều khoản
   - Thông báo: Nội dung thông báo, yêu cầu
   - Báo cáo: Tình hình, kết quả, kiến nghị
   - Công văn: Nội dung yêu cầu/đề nghị
   - Tờ trình: Căn cứ, nội dung đề xuất

7. NƠI NHẬN:
   Nơi nhận:
   - Như trên;
   - Lưu: VT, [Đơn vị soạn thảo].

8. CHỮ KÝ:
   [CHỨC VỤ NGƯỜI KÝ]
   (Ký, đóng dấu)
   [HỌ VÀ TÊN]

9. CĂN CỨ PHÁP LÝ (NGUYÊN TẮC QUAN TRỌNG):
   - CHỈ ĐƯỢC trích dẫn văn bản pháp luật có SỐ HIỆU xuất hiện trong TÀI LIỆU THAM KHẢO ở trên.
   - BẮT BUỘC ghi rõ SỐ HIỆU văn bản (ví dụ: 23/2025/TT-BVHTTDL, 86/2023/NĐ-CP), kèm tên đầy đủ.
   - Với mỗi căn cứ phải nêu Điều/Khoản áp dụng và tóm lược nội dung quy định.
   - TUYỆT ĐỐI KHÔNG ĐƯỢC bịa đặt hoặc thêm văn bản từ kiến thức bên ngoài.
   - Nếu TÀI LIỆU THAM KHẢO không có căn cứ pháp lý phù hợp thì để trống phần căn cứ.
   - KHÔNG ĐƯỢC tự nghĩ ra số hiệu nghị định, thông tư, luật.
"""

# Mapping loại văn bản từ request
DOCUMENT_TYPE_PATTERNS = {
    "ke_hoach": [r"kế hoạch", r"lập kế hoạch", r"xây dựng kế hoạch"],
    "quyet_dinh": [r"quyết định", r"ra quyết định", r"ban hành quyết định"],
    "thong_bao": [r"thông báo", r"ra thông báo"],
    "bao_cao": [r"báo cáo", r"viết báo cáo", r"lập báo cáo"],
    "cong_van": [r"công văn", r"viết công văn"],
    "to_trinh": [r"tờ trình", r"viết tờ trình"],
    "don": [r"đơn", r"viết đơn", r"lập đơn"],
    "bien_ban": [r"biên bản", r"lập biên bản"],
}


def _detect_document_type(request: str) -> str:
    """Detect document type from request text."""
    request_lower = request.lower()
    for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, request_lower):
                return doc_type
    return "cong_van"


def _detect_linh_vuc(request: str) -> str:
    """Detect domain/field from request text."""
    request_lower = request.lower()
    field_map = {
        "le_hoi": [r"lễ hội", r"quản lý lễ hội", r"tổ chức lễ hội"],
        "van_hoa": [r"văn hóa", r"di sản", r"bảo tàng"],
        "giao_duc": [r"giáo dục", r"đào tạo", r"trường học"],
        "y_te": [r"y tế", r"sức khỏe", r"bệnh viện"],
        "dat_dai": [r"đất đai", r"nhà ở", r"quy hoạch"],
        "hanh_chinh": [r"hành chính", r"thủ tục", r"cải cách"],
        "tai_chinh": [r"tài chính", r"ngân sách", r"thuế"],
    }
    for field, patterns in field_map.items():
        for pattern in patterns:
            if re.search(pattern, request_lower):
                return field
    return "chung"


def _detect_co_quan(request: str) -> str:
    """Detect issuing authority from request text."""
    request_lower = request.lower()
    authority_map = {
        "UBND_xa": [r"ubnd xã", r"ubnd phường", r"ủy ban nhân dân xã"],
        "UBND_huyen": [r"ubnd huyện", r"ubnd quận", r"ủy ban nhân dân huyện"],
        "UBND_tinh": [r"ubnd tỉnh", r"ủy ban nhân dân tỉnh"],
        "So": [r"sở", r"sở văn hóa", r"sở giáo dục"],
        "Phong": [r"phòng", r"phòng văn hóa", r"phòng giáo dục"],
    }
    for authority, patterns in authority_map.items():
        for pattern in patterns:
            if re.search(pattern, request_lower):
                return authority
    return "UBND_xa"


def extract_document_metadata(request: str) -> Dict[str, str]:
    """Extract document metadata from the drafting request.

    Returns a dict with keys: loai_van_ban, linh_vuc, co_quan, chu_de.
    This metadata is stored in session memory for follow-up questions.
    """
    return {
        "loai_van_ban": _detect_document_type(request),
        "linh_vuc": _detect_linh_vuc(request),
        "co_quan": _detect_co_quan(request),
        "chu_de": request[:100],
    }


async def run(
    content: str,
    temperature: float = 0.3,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Thực thi tool soạn thảo văn bản.

    Args:
        content: Yêu cầu soạn thảo (VD: "Soạn kế hoạch quản lý lễ hội").
        temperature: Temperature cho LLM.
        conversation_id: ID conversation để lưu document metadata.

    Returns:
        {"tool": "draft", "result": str, "sources": List[dict], "document_meta": dict}
    """
    linh_vuc = _detect_linh_vuc(content)
    linh_vuc_query = {
        "le_hoi": "quản lý lễ hội văn hóa",
        "van_hoa": "quản lý văn hóa di sản",
        "giao_duc": "quản lý giáo dục đào tạo",
        "y_te": "quản lý y tế sức khỏe",
        "dat_dai": "quản lý đất đai quy hoạch",
        "hanh_chinh": "quản lý hành chính công",
        "tai_chinh": "quản lý tài chính ngân sách",
    }.get(linh_vuc, "")
    search_queries = [content]
    if linh_vuc_query:
        search_queries.append(linh_vuc_query)
    results = await search_all(content, top_k=8, query_variants=search_queries)
    context = (
        "\n\n".join(doc["text"] for doc in results)
        if results
        else "Không có tài liệu tham khảo."
    )

    prompt = DRAFT_PROMPT.format(request=content, context=context)
    draft = await generate(prompt, system=DRAFT_SYSTEM, temperature=temperature)
    draft = _strip_hallucinated_refs(draft, results)
    draft = _ensure_legal_basis_section(draft, results)

    # Extract metadata for session memory
    doc_meta = extract_document_metadata(content)

    # Save to conversation memory if conversation_id provided
    if conversation_id:
        from app.memory.conversation_store import conversation_store
        conversation_store.update_document_context(conversation_id, doc_meta)

    return {
        "tool": "draft",
        "result": draft,
        "sources": format_sources(results),
        "document_meta": doc_meta,
    }


def _norm_doc(doc_num: str) -> str:
    """Normalize doc number for comparison: NĐ→ND, QĐ→QD, lowercase."""
    return doc_num.replace("_", "/").replace("Đ", "D").replace("đ", "d").lower()


def _collect_context_doc_numbers(results: List[dict]) -> set:
    """Collect all doc numbers from retrieved context (normalized)."""
    nums: set = set()
    for item in results:
        text = item.get("text", "")
        meta = item.get("metadata", {}) or {}
        for field in [text, meta.get("law_name", ""), meta.get("doc_number", "")]:
            for m in _DOC_NUMBER_RE.finditer(field or ""):
                nums.add(_norm_doc(m.group(1)))
    return nums


def _strip_hallucinated_refs(draft: str, results: List[dict]) -> str:
    """Remove lines citing doc numbers not found in retrieved context."""
    if not draft:
        return draft

    context_nums = _collect_context_doc_numbers(results) if results else set()
    answer_nums = {m.group(1).replace("_", "/") for m in _DOC_NUMBER_RE.finditer(draft)}
    if not answer_nums:
        return draft

    if not context_nums:
        hallucinated = answer_nums
    else:
        hallucinated = {n for n in answer_nums if _norm_doc(n) not in context_nums}

    if not hallucinated:
        return draft

    log.warning("[DRAFT ANTI-HALLUCINATION] Stripping: %s (context has: %s)", hallucinated, context_nums or "none")
    cleaned = draft
    for bad in hallucinated:
        cleaned = re.sub(rf"[^\n]*{re.escape(bad)}[^\n]*\n?", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _ensure_legal_basis_section(draft: str, results: list[dict]) -> str:
    """Ensure legal basis section uses ONLY references from the database."""
    text = (draft or "").strip()
    if not text:
        return text

    db_refs: list[str] = []
    seen = set()
    for item in results:
        meta = item.get("metadata", {}) or {}
        doc_number = (meta.get("doc_number") or "").strip().replace("_", "/")
        law_name = (meta.get("law_name") or "").strip()
        article = (meta.get("article_number") or "").strip()
        if not doc_number and not law_name:
            continue
        if doc_number and law_name:
            label = f"{doc_number} ({law_name})"
        else:
            label = doc_number or law_name
        ref = f"{label} – Điều {article}" if article else label
        k = (doc_number or law_name).lower()
        if k in seen:
            continue
        seen.add(k)
        db_refs.append(ref)
        if len(db_refs) >= 8:
            break

    if "căn cứ pháp lý" in text.lower() or "căn cứ:" in text.lower():
        if db_refs:
            return text
        return _replace_legal_basis_with_db_refs(text, db_refs)

    if not db_refs:
        return text

    section = "Căn cứ pháp lý:\n" + "\n".join(f"    {r};" for r in db_refs)
    return f"{text}\n\n{section}"


def _replace_legal_basis_with_db_refs(draft: str, db_refs: list[str]) -> str:
    """Replace the existing legal basis section with only database-verified references."""
    basis_pattern = re.compile(
        r"(Căn cứ[^:]*:)\s*([\s\S]*?)(?=\n\s*(?:I\.|II\.|III\.|1\.|MỤC ĐÍCH|NỘI DUNG|YÊU CẦU|Điều|\Z))",
        re.IGNORECASE,
    )
    match = basis_pattern.search(draft)
    if not match:
        return draft

    if db_refs:
        new_basis = match.group(1) + "\n" + "\n".join(f"    {r};" for r in db_refs)
    else:
        new_basis = match.group(1) + "\n    (Cần bổ sung căn cứ pháp lý từ cơ sở dữ liệu)"

    return draft[:match.start()] + new_basis + draft[match.end():]
