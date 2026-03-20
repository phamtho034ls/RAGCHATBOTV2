"""
Statistics Estimator – ước tính quy mô quản lý và nguồn lực hành chính.

Dựa trên dân số, diện tích, cấp hành chính để ước tính:
    - Quy mô quản lý (loại I / II / III)
    - Số lượng cán bộ, công chức
    - Cơ cấu tổ chức
    - Nguồn lực cần thiết
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.services.llm_client import generate
from app.config import STATISTICS_ESTIMATOR_PROMPT

log = logging.getLogger(__name__)

_SYSTEM = "Bạn là chuyên gia quản lý hành chính công Việt Nam."

_LEVEL_LABELS = {
    "xa": "Xã/Phường/Thị trấn",
    "huyen": "Huyện/Quận/Thị xã",
    "tinh": "Tỉnh/Thành phố",
}


async def estimate_resources(
    population: Optional[int] = None,
    area: Optional[float] = None,
    admin_level: str = "xa",
    context: str = "",
) -> Dict[str, Any]:
    """Ước tính quy mô quản lý và nguồn lực.

    Args:
        population: Dân số (người).
        area: Diện tích (km²).
        admin_level: Cấp hành chính ("xa", "huyen", "tinh").
        context: Bối cảnh bổ sung.

    Returns:
        Dict with estimation (text) and input parameters.
    """
    input_parts: list[str] = []
    if population is not None:
        input_parts.append(f"- Dân số: {population:,} người")
    if area is not None:
        input_parts.append(f"- Diện tích: {area:,.1f} km²")
    input_parts.append(
        f"- Cấp hành chính: {_LEVEL_LABELS.get(admin_level, admin_level)}"
    )
    if context:
        input_parts.append(f"- Bối cảnh: {context}")

    if not input_parts:
        return {
            "estimation": (
                "Cần cung cấp ít nhất một thông tin "
                "(dân số, diện tích, cấp hành chính) để ước tính."
            ),
            "input": {},
        }

    input_data = "\n".join(input_parts)
    prompt = STATISTICS_ESTIMATOR_PROMPT.format(input_data=input_data)

    log.info(
        "[STATS_ESTIMATOR] population=%s, area=%s, level=%s",
        population, area, admin_level,
    )

    result = await generate(prompt, system=_SYSTEM, temperature=0.3)

    return {
        "estimation": result,
        "input": {
            "population": population,
            "area": area,
            "admin_level": admin_level,
        },
    }
