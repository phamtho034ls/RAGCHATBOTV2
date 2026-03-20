"""
Ninh Bình router – xác định khi nào route câu hỏi sang Ninh Bình search tool.

Dùng bởi: rag_chain_v2 (chat flow), copilot_agent (copilot flow).

Rule: Chỉ route khi câu hỏi liên quan Ninh Bình (tỉnh, huyện, địa danh) VÀ không phải pháp luật.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Từ khóa địa danh Ninh Bình – tỉnh, huyện, địa điểm nổi tiếng
NINH_BINH_KEYWORDS = [
    r"ninh\s+bình",
    r"tràng\s+an",
    r"tam\s+cốc",
    r"tam\s+coc",
    r"bái\s+đính",
    r"bai\s+dinh",
    r"bích\s+động",
    r"vân\s+long",
    r"cố\s+đô\s+hoa\s+lư",
    r"hoa\s+lư",
    # Các huyện, thành phố thuộc Ninh Bình
    r"yên\s+mô",
    r"yên\s+khánh",
    r"gia\s+viễn",
    r"kim\s+sơn",
    r"nho\s+quan",
    r"tam\s+điệp",
    r"tam\s+diep",
    r"thành\s+phố\s+ninh\s+bình",
    r"huyện\s+yên\s+mô",
    r"huyện\s+yên\s+khánh",
    r"huyện\s+gia\s+viễn",
    r"huyện\s+kim\s+sơn",
    r"huyện\s+nho\s+quan",
    r"huyện\s+hoa\s+lư",
    # Xã phổ biến (Liên Sơn, Trường Yên, ...)
    r"liên\s+sơn",
    r"trường\s+yên",
    # Mở rộng cho các địa danh thường hỏi (không cần dataset)
    r"gia\s+tường",
    r"gia\s+tuong",
    r"gia\s+viên",
    r"gia\s+vien",
    r"gia\s+lập",
    r"gia\s+lap",
    r"gia\s+phương",
    r"gia\s+phuong",
    r"gia\s+sinh",
    r"gia\s+tân",
    r"gia\s+tan",
    r"gia\s+trung",
]
NINH_BINH_PATTERN = re.compile("|".join(NINH_BINH_KEYWORDS), re.IGNORECASE)

# Từ khóa pháp lý – nếu có → KHÔNG route sang Ninh Bình
LEGAL_KEYWORDS_PATTERN = re.compile(
    r"\bluật\b|\bnghị\s*định\b|\bthông\s*tư\b"
    r"|\bđiều\s+\d+|\bkhoản\s+\d+|\bpháp\s+luật\b"
    r"|\bquy\s*định\b|\bquyết\s*định\b|\bnghị\s*quyết\b"
    r"|\bthủ\s*tục\s+hành\s+chính\b|\bcăn\s+cứ\s+pháp\s+lý\b"
    r"|\bchỉ\s*thị\b|\bchủ\s*trương\b",
    re.IGNORECASE,
)

# Dấu hiệu câu hỏi tình huống hành chính – nếu có → KHÔNG route sang Ninh Bình
ADMIN_SCENARIO_PATTERN = re.compile(
    r"(?:"
    r"ông[\s/]bà"
    r"|\b(?:xử\s+lý|hướng\s+dẫn|quy\s+trình|tham\s+mưu|đề\s+xuất)\b"
    r"|\b(?:vi\s+phạm|biên\s+bản|xử\s+phạt|lập\s+biên\s+bản)\b"
    r"|\b(?:xin\s+phép|cấp\s+phép|đăng\s+ký)\b"
    r"|\b(?:trùng\s+tu|tôn\s+tạo|tu\s+bổ|phục\s+dựng|bảo\s+tồn)\b"
    r"|\b(?:phối\s+hợp|liên\s+ngành|vận\s+động|tuyên\s+truyền)\b"
    r"|\b(?:giải\s+pháp|biện\s+pháp|phương\s+án|kế\s+hoạch)\s+(?:ra\s+quân|xử\s+lý|chấn\s+chỉnh|nào|gì)\b"
    r"|\b(?:ra\s+quân|chấn\s+chỉnh)\b"
    r"|\b(?:karaoke|quảng\s+cáo|biển\s+hiệu|băng\s+rôn)\s+.{0,20}(?:vi\s+phạm|trái|sai)\b"
    r"|\b(?:bạo\s+lực\s+gia\s+đình|tôn\s+giáo\s+trái\s+phép)\b"
    r"|\b(?:nhà\s+văn\s+hóa|thiết\s+chế\s+văn\s+hóa)\b"
    r"|\b(?:đại\s+hội\s+thể\s+dục|đại\s+hội\s+thể\s+thao)\b"
    r")",
    re.IGNORECASE,
)

# Heuristic fallback cho các câu hỏi hành chính địa phương dạng:
# "phường X sau sáp nhập có diện tích bao nhiêu", "xã Y mới có dân số bao nhiêu", ...
MERGE_ADMIN_PATTERN = re.compile(
    r"\b(xã|xa|phường|phuong|huyện|huyen|thị\s*trấn|thi\s*tran)\b.{0,80}\b("
    r"mới|sáp\s*nhập|sát\s*nhập|sap\s*nhap|sat\s*nhap|"
    r"diện\s*tích|dân\s*số|dien\s*tich|dan\s*so"
    r")\b",
    re.IGNORECASE,
)

# Câu hỏi mang tính địa lý / thông tin chung (phi pháp lý) – catch rộng hơn keyword tĩnh.
GEO_INFO_PATTERN = re.compile(
    r"(?:"
    # "X ở đâu", "X nằm ở đâu"
    r".+\b(ở\s+đâu|o\s+dau|nằm\s+ở|nam\s+o)\b"
    r"|"
    # "X thuộc huyện/tỉnh/xã nào", "X thuộc đâu"
    r".+\bthuộc\b.{0,40}\b(huyện|tỉnh|xã|phường|quận|thành\s*phố|nào|đâu|dau)\b"
    r"|"
    # "diện tích / dân số + tên"
    r"\b(diện\s*tích|dân\s*số|dien\s*tich|dan\s*so)\b"
    r"|"
    # "tin tức / thông tin / cập nhật ... mới/về/của"
    r"\b(tin\s+tức|thông\s+tin|cập\s+nhật|cap\s+nhat)\b.{0,60}\b(mới|về|của|cho|nhất)\b"
    r"|"
    # Câu hỏi bắt đầu bằng đơn vị hành chính + tên riêng
    r"^\s*(xã|xa|phường|phuong|huyện|huyen|thị\s*trấn|thi\s*tran)\s+[A-ZÀ-Ỹa-zà-ỹ]"
    r"|"
    # "du lịch / tham quan / đặc sản / lễ hội ..."
    r"\b(du\s+lịch|tham\s+quan|đặc\s+sản|lễ\s+hội|di\s+tích|danh\s+lam)\b"
    r"|"
    # "bao nhiêu" kèm đơn vị hành chính
    r"\b(xã|xa|phường|phuong|huyện|huyen)\b.{0,60}\b(bao\s+nhiêu|mấy)\b"
    r")",
    re.IGNORECASE,
)


def should_use_ninh_binh_tool(question: str) -> bool:
    """True nếu câu hỏi về Ninh Bình (địa lý, địa danh) và KHÔNG liên quan pháp luật/hành chính."""
    q = (question or "").strip()
    if not q:
        return False
    if LEGAL_KEYWORDS_PATTERN.search(q):
        return False
    if ADMIN_SCENARIO_PATTERN.search(q):
        return False
    if NINH_BINH_PATTERN.search(q):
        return True
    if MERGE_ADMIN_PATTERN.search(q):
        return True
    if GEO_INFO_PATTERN.search(q):
        return True
    return False


async def route_to_ninh_binh(query: str) -> dict:
    """Gọi Ninh Bình tool và trả về {answer, sources} theo format rag_chain_v2."""
    from app.tools.ninh_binh_search_tool import run as ninh_binh_run

    result = await ninh_binh_run(query)
    return {
        "answer": result.get("result", ""),
        "sources": result.get("sources", []),
    }
