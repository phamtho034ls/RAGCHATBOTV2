"""Ninh Bình Structured Information Extractor.

Takes raw web search content and extracts structured info into 6 predefined fields.
Uses LLM for intelligent extraction and summarization across multiple sources.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.llm_client import generate

log = logging.getLogger(__name__)

STRUCTURED_FIELDS = [
    "hanh_chinh",
    "dia_ly",
    "kinh_te",
    "ha_tang",
    "du_lich",
    "tin_moi",
]

FIELD_LABELS = {
    "hanh_chinh": "Hành chính",
    "dia_ly": "Địa lý",
    "kinh_te": "Kinh tế",
    "ha_tang": "Hạ tầng",
    "du_lich": "Du lịch & Văn hóa",
    "tin_moi": "Tin mới",
}

EXTRACTION_SYSTEM_PROMPT = """\
Bạn là chuyên gia phân tích thông tin hành chính Việt Nam.
Nhiệm vụ: Trích xuất và tổng hợp thông tin từ nhiều nguồn tìm kiếm web,
phân loại vào 6 lĩnh vực dưới đây.

NGUYÊN TẮC:
- Chỉ sử dụng thông tin có trong nguồn được cung cấp
- Nếu không có dữ liệu cho một lĩnh vực → trả "Chưa có dữ liệu rõ ràng"
- Trả lời bằng tiếng Việt, rõ ràng, ngắn gọn nhưng đầy đủ
- KHÔNG bịa thông tin, KHÔNG thêm dữ liệu không có trong nguồn
- Ưu tiên thông tin mới nhất nếu có mâu thuẫn giữa các nguồn
- Gộp thông tin trùng lặp từ nhiều nguồn thành một đoạn mạch lạc

6 LĨNH VỰC:
1. hanh_chinh: Phân cấp hành chính, dân số, diện tích, đơn vị trực thuộc
2. dia_ly: Vị trí địa lý, ranh giới, địa hình, khí hậu
3. kinh_te: Ngành kinh tế chủ lực, sản xuất, nông nghiệp, công nghiệp, thương mại
4. ha_tang: Giao thông, trường học, bệnh viện, điện nước, công trình
5. du_lich: Danh lam thắng cảnh, di tích, lễ hội, văn hóa, đặc sản
6. tin_moi: Tin tức, dự án mới, phát triển gần đây

Trả về JSON với đúng 6 key trên. Mỗi value là string tiếng Việt.
"""


EXTRACTION_USER_TEMPLATE = """\
## ĐỊA DANH: {entity_display}
## CẤP HÀNH CHÍNH: {level}
## CÂU HỎI GỐC: {original_query}

## NỘI DUNG TỪ CÁC NGUỒN:
{sources_content}

---
Hãy trích xuất thông tin vào 6 lĩnh vực (hanh_chinh, dia_ly, kinh_te, ha_tang, du_lich, tin_moi).
Trả về JSON hợp lệ, KHÔNG bao gồm markdown code fences.
"""


RESPONSE_SYSTEM_PROMPT = """\
Bạn là trợ lý AI chuyên cung cấp thông tin về tỉnh Ninh Bình, Việt Nam.
Nhiệm vụ: Viết câu trả lời đầy đủ, tự nhiên bằng tiếng Việt dựa trên dữ liệu có cấu trúc.

NGUYÊN TẮC:
- Viết đầy đủ, chi tiết, không cắt ngắn
- Sử dụng tiếng Việt tự nhiên, dễ hiểu
- Cấu trúc rõ ràng với tiêu đề phần
- Chỉ trình bày thông tin có trong dữ liệu
- Nếu một lĩnh vực là "Chưa có dữ liệu rõ ràng" → bỏ qua lĩnh vực đó
- KHÔNG thêm thông tin ngoài dữ liệu được cung cấp
- KHÔNG thêm citation dạng [nguồn](url) trong câu trả lời
"""

RESPONSE_USER_TEMPLATE = """\
## ĐỊA DANH: {entity_display}
## CÂU HỎI: {original_query}

## DỮ LIỆU CẤU TRÚC:
{structured_json}

---
Viết câu trả lời đầy đủ, tự nhiên bằng tiếng Việt cho câu hỏi trên.
Tổ chức thành các phần rõ ràng (sử dụng tiêu đề **In đậm**).
Chỉ đưa ra các phần có dữ liệu thực tế.
"""


@dataclass
class StructuredResult:
    entity: str
    level: str
    data: Dict[str, str] = field(default_factory=dict)
    human_answer: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


async def extract_structured_info(
    sources_content: List[Dict[str, str]],
    entity_display: str,
    level: str,
    original_query: str,
) -> Dict[str, str]:
    """Extract structured 6-field info from raw search results via LLM."""
    if not sources_content:
        return {f: "Chưa có dữ liệu rõ ràng" for f in STRUCTURED_FIELDS}

    merged_content = ""
    for i, src in enumerate(sources_content, 1):
        title = src.get("title", f"Nguồn {i}")
        url = src.get("url", "")
        text = src.get("text", "")
        merged_content += f"\n### Nguồn {i}: {title}\nURL: {url}\n{text}\n"

    prompt = EXTRACTION_USER_TEMPLATE.format(
        entity_display=entity_display,
        level=level,
        original_query=original_query,
        sources_content=merged_content.strip(),
    )

    try:
        raw = await generate(prompt, system=EXTRACTION_SYSTEM_PROMPT, temperature=0.2)
        data = _parse_json_response(raw)
        for f in STRUCTURED_FIELDS:
            if f not in data or not data[f]:
                data[f] = "Chưa có dữ liệu rõ ràng"
        return data
    except Exception as exc:
        log.error("LLM extraction failed: %s", exc)
        return {f: "Chưa có dữ liệu rõ ràng" for f in STRUCTURED_FIELDS}


async def generate_human_answer(
    structured_data: Dict[str, str],
    entity_display: str,
    original_query: str,
) -> str:
    """Generate a natural Vietnamese answer from structured data."""
    has_data = any(
        v and v != "Chưa có dữ liệu rõ ràng"
        for v in structured_data.values()
    )
    if not has_data:
        return (
            f"Hiện tại chưa tìm được thông tin chi tiết về {entity_display}. "
            "Bạn có thể thử hỏi cụ thể hơn hoặc kiểm tra lại tên địa danh."
        )

    prompt = RESPONSE_USER_TEMPLATE.format(
        entity_display=entity_display,
        original_query=original_query,
        structured_json=json.dumps(structured_data, ensure_ascii=False, indent=2),
    )

    try:
        return await generate(prompt, system=RESPONSE_SYSTEM_PROMPT, temperature=0.3)
    except Exception as exc:
        log.error("LLM response generation failed: %s", exc)
        return _fallback_format(structured_data, entity_display)


def _fallback_format(data: Dict[str, str], entity_display: str) -> str:
    """Format structured data without LLM when generation fails."""
    parts = [f"**Thông tin về {entity_display}**\n"]
    for f in STRUCTURED_FIELDS:
        val = data.get(f, "")
        if val and val != "Chưa có dữ liệu rõ ràng":
            label = FIELD_LABELS.get(f, f)
            parts.append(f"**{label}:** {val}\n")
    return "\n".join(parts) if len(parts) > 1 else parts[0] + "\nChưa có dữ liệu rõ ràng."


def _parse_json_response(raw: str) -> Dict[str, str]:
    """Parse LLM response, handling markdown code fences and partial JSON."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    log.warning("Failed to parse LLM JSON, returning empty fields")
    return {}


def build_structured_result(
    entity_name: str,
    entity_level: str,
    data: Dict[str, str],
    human_answer: str,
    sources: List[Dict[str, Any]],
    confidence: float = 0.0,
) -> Dict[str, Any]:
    """Build the final structured response dict for API consumption."""
    return {
        "entity": entity_name,
        "level": entity_level,
        "data": {f: data.get(f, "Chưa có dữ liệu rõ ràng") for f in STRUCTURED_FIELDS},
        "answer": human_answer,
        "sources": sources,
        "confidence": confidence,
    }
