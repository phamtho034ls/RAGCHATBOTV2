"""
Tool: Trích xuất thông tin quan trọng từ văn bản pháp luật.

Tìm văn bản liên quan qua RAG → LLM trích xuất thông tin có cấu trúc.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.llm_client import generate
from app.services.retrieval import search_all, format_sources

EXTRACT_SYSTEM = (
    "Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam. "
    "Hãy trích xuất các thông tin quan trọng một cách chính xác. "
    "Khi nêu Điều/Khoản phải nêu rõ nội dung quy định tương ứng từ ngữ cảnh."
)

EXTRACT_PROMPT = """Hãy trích xuất các thông tin quan trọng từ văn bản sau:

VĂN BẢN:
{document_text}

Thông tin cần trích xuất:
1. **Số hiệu văn bản**: (nếu có)
2. **Ngày ban hành**: (nếu có)
3. **Cơ quan ban hành**: (nếu có)
4. **Loại văn bản**: (Luật / Nghị định / Thông tư / ...)
5. **Lĩnh vực**: (nếu xác định được)
6. **Các điều khoản chính**: Liệt kê Điều/Khoản và nội dung chính của từng điều khoản
7. **Đối tượng áp dụng**: (nếu có)
8. **Hiệu lực thi hành**: (nếu có)
9. **Các từ khóa quan trọng**: Liệt kê

YÊU CẦU BẮT BUỘC:
- Không chỉ liệt kê số điều; phải nêu nội dung mỗi điều khoản chính.
- Không bịa đặt nội dung ngoài văn bản được cung cấp.
"""


async def run(content: str, temperature: float = 0.1) -> Dict[str, Any]:
    """Thực thi tool trích xuất thông tin.

    Args:
        content: Câu truy vấn hoặc nội dung cần trích xuất.
        temperature: Temperature cho LLM.

    Returns:
        {"tool": "extract", "result": str, "sources": List[dict]}
    """
    results = await search_all(content, top_k=8)
    if not results:
        return {
            "tool": "extract",
            "result": "Không tìm thấy văn bản liên quan để trích xuất.",
            "sources": [],
        }

    document_text = "\n\n---\n\n".join(
        f"[Phần {i + 1}]\n{doc['text']}" for i, doc in enumerate(results)
    )
    prompt = EXTRACT_PROMPT.format(document_text=document_text)
    extraction = await generate(prompt, system=EXTRACT_SYSTEM, temperature=temperature)

    return {
        "tool": "extract",
        "result": extraction,
        "sources": format_sources(results),
    }
