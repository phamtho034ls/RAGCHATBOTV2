"""
Procedure Knowledge Base – cơ sở dữ liệu thủ tục hành chính.

Cung cấp thông tin về:
    - Các bước thực hiện thủ tục
    - Hồ sơ yêu cầu
    - Thời gian xử lý
    - Lệ phí
    - Cơ quan thực hiện
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ── Procedure Knowledge Base (in-memory) ────────────────────
# Có thể mở rộng sang database hoặc JSON file ngoài
PROCEDURES: Dict[str, dict] = {
    "dang_ky_doanh_nghiep": {
        "procedure_id": "TTHC-001",
        "procedure_name": "Đăng ký thành lập doanh nghiệp",
        "description": "Thủ tục đăng ký thành lập doanh nghiệp tư nhân, công ty TNHH, công ty cổ phần.",
        "steps": [
            {"step_number": 1, "description": "Tiếp nhận hồ sơ tại Phòng Đăng ký kinh doanh", "note": "Có thể nộp trực tuyến qua Cổng thông tin quốc gia"},
            {"step_number": 2, "description": "Kiểm tra tính hợp lệ của hồ sơ", "note": "Trong vòng 3 ngày làm việc"},
            {"step_number": 3, "description": "Nhập dữ liệu vào hệ thống đăng ký kinh doanh quốc gia", "note": None},
            {"step_number": 4, "description": "Cấp Giấy chứng nhận đăng ký doanh nghiệp", "note": None},
            {"step_number": 5, "description": "Trả kết quả cho doanh nghiệp", "note": "Trả trực tiếp hoặc qua bưu điện"},
        ],
        "required_documents": [
            "giay_de_nghi_dang_ky_doanh_nghiep",
            "dieu_le_cong_ty",
            "danh_sach_thanh_vien",
            "ban_sao_cmnd_cccd",
            "giay_chung_nhan_dia_diem",
        ],
        "processing_time": "3 ngày làm việc",
        "fee": "Miễn phí",
        "authority": "Phòng Đăng ký kinh doanh - Sở Kế hoạch và Đầu tư",
    },
    "cap_giay_phep_xay_dung": {
        "procedure_id": "TTHC-002",
        "procedure_name": "Cấp giấy phép xây dựng",
        "description": "Thủ tục cấp giấy phép xây dựng công trình, nhà ở riêng lẻ.",
        "steps": [
            {"step_number": 1, "description": "Nộp hồ sơ tại Bộ phận một cửa", "note": None},
            {"step_number": 2, "description": "Kiểm tra hồ sơ và thẩm định thiết kế", "note": "Thẩm định trong 15 ngày"},
            {"step_number": 3, "description": "Lấy ý kiến cơ quan liên quan", "note": "Phòng cháy, môi trường"},
            {"step_number": 4, "description": "Phê duyệt và cấp giấy phép", "note": None},
            {"step_number": 5, "description": "Trả kết quả", "note": None},
        ],
        "required_documents": [
            "don_xin_cap_phep_xay_dung",
            "ban_ve_thiet_ke",
            "giay_chung_nhan_quyen_su_dung_dat",
            "ban_sao_cmnd_cccd",
            "ho_so_thiet_ke_ky_thuat",
        ],
        "processing_time": "15 ngày làm việc (nhà ở riêng lẻ), 30 ngày (công trình khác)",
        "fee": "Theo quy định địa phương",
        "authority": "UBND cấp huyện hoặc Sở Xây dựng",
    },
    "dang_ky_ho_tich": {
        "procedure_id": "TTHC-003",
        "procedure_name": "Đăng ký hộ tịch (khai sinh, khai tử, kết hôn)",
        "description": "Thủ tục đăng ký các sự kiện hộ tịch tại UBND cấp xã.",
        "steps": [
            {"step_number": 1, "description": "Nộp hồ sơ tại UBND xã/phường", "note": None},
            {"step_number": 2, "description": "Cán bộ tư pháp kiểm tra hồ sơ", "note": None},
            {"step_number": 3, "description": "Ghi vào sổ hộ tịch", "note": None},
            {"step_number": 4, "description": "Cấp giấy chứng nhận", "note": "Trong ngày làm việc"},
        ],
        "required_documents": [
            "to_khai_dang_ky_ho_tich",
            "giay_chung_sinh_benh_vien",
            "ban_sao_cmnd_cccd_bo_me",
            "so_ho_khau",
        ],
        "processing_time": "Trong ngày làm việc",
        "fee": "Miễn phí (đăng ký đúng hạn)",
        "authority": "UBND cấp xã/phường",
    },
    "cap_so_do": {
        "procedure_id": "TTHC-004",
        "procedure_name": "Cấp giấy chứng nhận quyền sử dụng đất (sổ đỏ)",
        "description": "Thủ tục cấp giấy chứng nhận quyền sử dụng đất lần đầu.",
        "steps": [
            {"step_number": 1, "description": "Nộp hồ sơ tại Văn phòng đăng ký đất đai", "note": "Hoặc UBND cấp xã"},
            {"step_number": 2, "description": "Đo đạc, khảo sát thực địa", "note": None},
            {"step_number": 3, "description": "Thẩm định hồ sơ", "note": None},
            {"step_number": 4, "description": "Niêm yết công khai tại UBND xã", "note": "15 ngày"},
            {"step_number": 5, "description": "Trình UBND cấp có thẩm quyền ký", "note": None},
            {"step_number": 6, "description": "Trả kết quả", "note": None},
        ],
        "required_documents": [
            "don_dang_ky_cap_gcn_quyen_su_dung_dat",
            "giay_to_nguon_goc_dat",
            "ban_sao_cmnd_cccd",
            "so_ho_khau",
            "ban_do_dia_chinh",
        ],
        "processing_time": "30 ngày làm việc",
        "fee": "Theo quy định địa phương",
        "authority": "UBND cấp huyện / Sở Tài nguyên và Môi trường",
    },
    "dang_ky_tam_tru": {
        "procedure_id": "TTHC-005",
        "procedure_name": "Đăng ký tạm trú",
        "description": "Thủ tục đăng ký tạm trú cho công dân tại địa phương.",
        "steps": [
            {"step_number": 1, "description": "Nộp hồ sơ tại Công an xã/phường", "note": None},
            {"step_number": 2, "description": "Kiểm tra hồ sơ", "note": None},
            {"step_number": 3, "description": "Cập nhật thông tin vào cơ sở dữ liệu cư trú", "note": None},
            {"step_number": 4, "description": "Cấp xác nhận đăng ký tạm trú", "note": None},
        ],
        "required_documents": [
            "phieu_bao_thay_doi_nhan_khau",
            "ban_sao_cmnd_cccd",
            "giay_to_chung_minh_cho_o_hop_phap",
        ],
        "processing_time": "3 ngày làm việc",
        "fee": "Miễn phí",
        "authority": "Công an cấp xã/phường",
    },
    "cap_phep_kinh_doanh": {
        "procedure_id": "TTHC-006",
        "procedure_name": "Cấp giấy phép kinh doanh có điều kiện",
        "description": "Thủ tục cấp phép kinh doanh cho ngành nghề có điều kiện (thực phẩm, dược phẩm, ...).",
        "steps": [
            {"step_number": 1, "description": "Nộp hồ sơ tại cơ quan quản lý ngành", "note": None},
            {"step_number": 2, "description": "Thẩm định điều kiện kinh doanh", "note": "Kiểm tra cơ sở vật chất"},
            {"step_number": 3, "description": "Thẩm tra hồ sơ pháp lý", "note": None},
            {"step_number": 4, "description": "Cấp giấy phép", "note": None},
            {"step_number": 5, "description": "Trả kết quả", "note": None},
        ],
        "required_documents": [
            "don_xin_cap_phep",
            "giay_chung_nhan_dang_ky_doanh_nghiep",
            "giay_chung_nhan_du_dieu_kien",
            "ban_sao_cmnd_cccd",
            "ho_so_nhan_su",
        ],
        "processing_time": "15-30 ngày làm việc",
        "fee": "Theo quy định ngành",
        "authority": "Sở quản lý ngành / UBND cấp tỉnh",
    },
}

# Ánh xạ từ khóa để tìm kiếm thủ tục
PROCEDURE_KEYWORDS: Dict[str, List[str]] = {
    "dang_ky_doanh_nghiep": [
        "đăng ký doanh nghiệp", "thành lập công ty", "thành lập doanh nghiệp",
        "mở công ty", "đăng ký kinh doanh", "giấy phép kinh doanh mới",
    ],
    "cap_giay_phep_xay_dung": [
        "giấy phép xây dựng", "xin phép xây", "cấp phép xây dựng",
        "xây nhà", "xây dựng công trình",
    ],
    "dang_ky_ho_tich": [
        "hộ tịch", "khai sinh", "khai tử", "kết hôn",
        "đăng ký khai sinh", "giấy khai sinh", "đăng ký kết hôn",
    ],
    "cap_so_do": [
        "sổ đỏ", "giấy chứng nhận quyền sử dụng đất",
        "cấp sổ đỏ", "quyền sử dụng đất", "gcn đất",
    ],
    "dang_ky_tam_tru": [
        "tạm trú", "đăng ký tạm trú", "tạm vắng",
        "cư trú", "nhập hộ khẩu",
    ],
    "cap_phep_kinh_doanh": [
        "giấy phép kinh doanh", "kinh doanh có điều kiện",
        "cấp phép kinh doanh", "giấy phép ngành",
    ],
}


def get_procedure(procedure_id: str) -> Optional[dict]:
    """Lấy thông tin thủ tục theo ID."""
    return PROCEDURES.get(procedure_id)


def get_procedure_steps(procedure_id: str) -> Optional[List[dict]]:
    """Lấy danh sách các bước thực hiện thủ tục."""
    proc = PROCEDURES.get(procedure_id)
    if proc:
        return proc["steps"]
    return None


def get_required_documents(procedure_id: str) -> Optional[List[str]]:
    """Lấy danh sách hồ sơ yêu cầu cho thủ tục."""
    proc = PROCEDURES.get(procedure_id)
    if proc:
        return proc["required_documents"]
    return None


def list_procedures() -> List[dict]:
    """Liệt kê tất cả thủ tục có sẵn."""
    return [
        {
            "procedure_id": proc["procedure_id"],
            "procedure_name": proc["procedure_name"],
            "description": proc["description"],
        }
        for proc in PROCEDURES.values()
    ]


def search_procedure(query: str) -> Optional[dict]:
    """Tìm thủ tục phù hợp nhất dựa trên query (keyword matching).

    Returns:
        Dict thông tin thủ tục hoặc None nếu không tìm thấy.
    """
    query_lower = query.lower()

    # Tìm theo keyword matching
    best_match: Optional[str] = None
    best_score = 0

    for proc_id, keywords in PROCEDURE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > best_score:
            best_score = score
            best_match = proc_id

    if best_match and best_score > 0:
        log.info("Found procedure: %s (score=%d)", best_match, best_score)
        return PROCEDURES[best_match]

    # Fallback: tìm theo tên thủ tục (partial match)
    for proc_id, proc in PROCEDURES.items():
        proc_name = proc["procedure_name"].lower()
        if any(word in query_lower for word in proc_name.split() if len(word) > 2):
            return proc

    return None
