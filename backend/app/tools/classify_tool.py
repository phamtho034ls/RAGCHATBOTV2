"""
Tool: Phân loại văn bản pháp luật.

Tìm văn bản liên quan qua RAG → LLM phân loại theo các tiêu chí chuẩn.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.llm_client import generate
from app.services.retrieval import search_all, format_sources

CLASSIFY_SYSTEM = "Bạn là chuyên gia phân loại văn bản pháp luật Việt Nam."

CLASSIFY_PROMPT = """Hãy phân loại văn bản/yêu cầu sau:

NỘI DUNG: {content}

TÀI LIỆU THAM KHẢO:
{context}

Phân loại theo các tiêu chí:
1. **Loại văn bản**: (Luật / Nghị định / Thông tư / Quyết định / Công văn / Chỉ thị / Nghị quyết / Khác)
2. **Lĩnh vực**: (Hành chính / Giáo dục / Y tế / Đất đai / Tài chính / Lao động / Văn hóa thể thao / Môi trường / An ninh quốc phòng / Nông nghiệp / Khác)
3. **Cấp ban hành**: (Trung ương / Tỉnh / Huyện / Xã)
4. **Mức độ quan trọng**: (Cao / Trung bình / Thấp)
5. **Trạng thái**: (Còn hiệu lực / Hết hiệu lực / Không xác định)
6. **Tóm tắt ngắn**: Nội dung chính trong 1-2 câu
"""


async def run(content: str, temperature: float = 0.1) -> Dict[str, Any]:
    """Thực thi tool phân loại văn bản.

    Args:
        content: Nội dung hoặc câu truy vấn cần phân loại.
        temperature: Temperature cho LLM.

    Returns:
        {"tool": "classify", "result": str, "sources": List[dict]}
    """
    results = await search_all(content, top_k=5)
    context = (
        "\n\n".join(doc["text"] for doc in results)
        if results
        else "Không có tài liệu tham khảo."
    )

    prompt = CLASSIFY_PROMPT.format(content=content, context=context)
    classification = await generate(prompt, system=CLASSIFY_SYSTEM, temperature=temperature)

    return {
        "tool": "classify",
        "result": classification,
        "sources": format_sources(results),
    }
