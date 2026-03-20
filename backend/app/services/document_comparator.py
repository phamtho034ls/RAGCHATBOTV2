"""
Document Comparator – so sánh hai văn bản pháp luật.

Tìm nội dung liên quan đến hai văn bản qua RAG → gửi cho LLM phân tích.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.config import COMPARE_PROMPT, COPILOT_SYSTEM_PROMPT
from app.services.llm_client import generate
from app.services.retrieval import search_all, format_sources

log = logging.getLogger(__name__)


def _extract_document_names(query: str) -> tuple[str, str]:
    """Trích xuất tên hai văn bản từ câu hỏi so sánh.

    Ví dụ:
        "So sánh nghị định 01/2021 và nghị định 02/2023"
        → ("nghị định 01/2021", "nghị định 02/2023")
    """
    # Pattern: "so sánh X và Y" hoặc "so sánh X với Y"
    patterns = [
        r"so sánh\s+(.+?)\s+(?:và|với|vs)\s+(.+?)(?:\s*$|\s*\.)",
        r"(?:khác nhau|khác biệt)\s+(?:giữa\s+)?(.+?)\s+(?:và|với)\s+(.+?)(?:\s*$|\s*\.)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()

    # Fallback: chia đôi query
    return query, ""


async def compare_documents(
    query: str,
    temperature: float = 0.3,
    top_k: int = 8,
    dataset_id_1: Optional[str] = None,
    dataset_id_2: Optional[str] = None,
) -> Dict[str, Any]:
    """So sánh hai văn bản pháp luật.

    Pipeline:
        Query → Extract document names
        → RAG search cho mỗi văn bản
        → Compare LLM

    Returns:
        {"comparison": str, "sources": List[dict]}
    """
    # 1. Extract document names
    doc_name_1, doc_name_2 = _extract_document_names(query)

    if not doc_name_2:
        return {
            "comparison": (
                "Vui lòng chỉ rõ hai văn bản cần so sánh.\n"
                "Ví dụ: 'So sánh nghị định 01/2021 và nghị định 02/2023'"
            ),
            "sources": [],
        }

    # 2. RAG search cho từng văn bản
    results_1 = await search_all(doc_name_1, top_k=top_k)
    results_2 = await search_all(doc_name_2, top_k=top_k)

    if not results_1 and not results_2:
        return {
            "comparison": "Không tìm thấy thông tin về cả hai văn bản trong cơ sở dữ liệu.",
            "sources": [],
        }

    # 3. Build document contexts
    doc_text_1 = "\n\n".join(doc["text"] for doc in results_1) if results_1 else "Không tìm thấy nội dung."
    doc_text_2 = "\n\n".join(doc["text"] for doc in results_2) if results_2 else "Không tìm thấy nội dung."

    # 4. Compare prompt
    prompt = COMPARE_PROMPT.format(
        document_1=f"**{doc_name_1}**\n\n{doc_text_1}",
        document_2=f"**{doc_name_2}**\n\n{doc_text_2}",
    )

    # 5. Generate comparison
    comparison = await generate(
        prompt,
        system=COPILOT_SYSTEM_PROMPT,
        temperature=temperature,
    )

    all_sources = format_sources(results_1) + format_sources(results_2)
    return {
        "comparison": comparison,
        "sources": all_sources,
        "sources_doc1": format_sources(results_1),
        "sources_doc2": format_sources(results_2),
    }
