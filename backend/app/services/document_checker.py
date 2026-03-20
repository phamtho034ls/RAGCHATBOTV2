"""
Document Checker – kiểm tra hồ sơ còn thiếu.

Cho phép cán bộ nhập danh sách hồ sơ đã nhận và kiểm tra so với
yêu cầu của thủ tục hành chính.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.services.procedure_service import PROCEDURES, search_procedure, get_required_documents

log = logging.getLogger(__name__)

# Ánh xạ tên hồ sơ tiếng Việt → ID chuẩn hóa
DOCUMENT_NAME_MAP: Dict[str, List[str]] = {
    "giay_de_nghi_dang_ky_doanh_nghiep": [
        "giấy đề nghị", "đơn đề nghị", "giấy đề nghị đăng ký",
        "giay_de_nghi", "giay_de_nghi_dang_ky_doanh_nghiep",
    ],
    "dieu_le_cong_ty": [
        "điều lệ", "điều lệ công ty", "dieu_le_cong_ty", "dieu_le",
    ],
    "danh_sach_thanh_vien": [
        "danh sách thành viên", "danh sách cổ đông",
        "danh_sach_thanh_vien", "ds thành viên",
    ],
    "ban_sao_cmnd_cccd": [
        "cmnd", "cccd", "căn cước", "chứng minh nhân dân",
        "ban_sao_cmnd_cccd", "bản sao cmnd", "giấy tờ tùy thân",
    ],
    "giay_chung_nhan_dia_diem": [
        "giấy chứng nhận địa điểm", "địa điểm kinh doanh",
        "giay_chung_nhan_dia_diem", "hợp đồng thuê nhà",
    ],
    "don_xin_cap_phep_xay_dung": [
        "đơn xin cấp phép", "đơn xin xây dựng",
        "don_xin_cap_phep_xay_dung",
    ],
    "ban_ve_thiet_ke": [
        "bản vẽ", "thiết kế", "bản vẽ thiết kế",
        "ban_ve_thiet_ke",
    ],
    "giay_chung_nhan_quyen_su_dung_dat": [
        "sổ đỏ", "giấy chứng nhận đất", "quyền sử dụng đất",
        "giay_chung_nhan_quyen_su_dung_dat", "gcn đất",
    ],
    "ho_so_thiet_ke_ky_thuat": [
        "hồ sơ thiết kế", "thiết kế kỹ thuật",
        "ho_so_thiet_ke_ky_thuat",
    ],
    "to_khai_dang_ky_ho_tich": [
        "tờ khai", "tờ khai hộ tịch", "to_khai_dang_ky_ho_tich",
    ],
    "giay_chung_sinh_benh_vien": [
        "giấy chứng sinh", "chứng sinh bệnh viện",
        "giay_chung_sinh_benh_vien",
    ],
    "ban_sao_cmnd_cccd_bo_me": [
        "cmnd bố mẹ", "cccd bố mẹ", "giấy tờ bố mẹ",
        "ban_sao_cmnd_cccd_bo_me",
    ],
    "so_ho_khau": [
        "sổ hộ khẩu", "hộ khẩu", "so_ho_khau",
    ],
    "don_dang_ky_cap_gcn_quyen_su_dung_dat": [
        "đơn đăng ký cấp sổ đỏ", "đơn xin cấp sổ",
        "don_dang_ky_cap_gcn_quyen_su_dung_dat",
    ],
    "giay_to_nguon_goc_dat": [
        "giấy tờ nguồn gốc đất", "nguồn gốc đất",
        "giay_to_nguon_goc_dat",
    ],
    "ban_do_dia_chinh": [
        "bản đồ địa chính", "ban_do_dia_chinh",
    ],
    "phieu_bao_thay_doi_nhan_khau": [
        "phiếu báo thay đổi nhân khẩu",
        "phieu_bao_thay_doi_nhan_khau",
    ],
    "giay_to_chung_minh_cho_o_hop_phap": [
        "chỗ ở hợp pháp", "giấy tờ chỗ ở",
        "giay_to_chung_minh_cho_o_hop_phap",
    ],
    "don_xin_cap_phep": [
        "đơn xin cấp phép", "don_xin_cap_phep",
    ],
    "giay_chung_nhan_dang_ky_doanh_nghiep": [
        "giấy chứng nhận đăng ký doanh nghiệp", "giấy phép doanh nghiệp",
        "giay_chung_nhan_dang_ky_doanh_nghiep",
    ],
    "giay_chung_nhan_du_dieu_kien": [
        "giấy chứng nhận đủ điều kiện", "chứng nhận đủ điều kiện",
        "giay_chung_nhan_du_dieu_kien",
    ],
    "ho_so_nhan_su": [
        "hồ sơ nhân sự", "ho_so_nhan_su", "lý lịch nhân sự",
    ],
}


def normalize_document_name(name: str) -> str:
    """Chuẩn hóa tên hồ sơ từ input (tiếng Việt hoặc ID) về dạng ID chuẩn."""
    name_lower = name.lower().strip()

    # Kiểm tra trực tiếp nếu input đã là ID chuẩn
    for doc_id in DOCUMENT_NAME_MAP:
        if name_lower == doc_id:
            return doc_id

    # Tìm theo alias
    for doc_id, aliases in DOCUMENT_NAME_MAP.items():
        for alias in aliases:
            if alias in name_lower or name_lower in alias:
                return doc_id

    # Không tìm thấy → trả về nguyên gốc
    return name_lower.replace(" ", "_")


def check_missing_documents(
    procedure_id: str,
    submitted: List[str],
) -> Dict[str, object]:
    """Kiểm tra hồ sơ còn thiếu so với yêu cầu thủ tục.

    Args:
        procedure_id: ID thủ tục hành chính.
        submitted: Danh sách hồ sơ đã nộp (tên tiếng Việt hoặc ID).

    Returns:
        {
            "procedure_name": str,
            "required_documents": List[str],
            "submitted_documents": List[str],
            "missing_documents": List[str],
            "is_complete": bool,
            "message": str,
        }
    """
    required = get_required_documents(procedure_id)
    if required is None:
        return {
            "procedure_name": procedure_id,
            "required_documents": [],
            "submitted_documents": submitted,
            "missing_documents": [],
            "is_complete": False,
            "message": f"Không tìm thấy thủ tục '{procedure_id}' trong hệ thống.",
        }

    proc = PROCEDURES[procedure_id]
    # Chuẩn hóa submitted documents
    normalized_submitted = [normalize_document_name(doc) for doc in submitted]
    missing = [doc for doc in required if doc not in normalized_submitted]
    is_complete = len(missing) == 0

    if is_complete:
        message = f"✅ Hồ sơ cho thủ tục '{proc['procedure_name']}' đã đầy đủ."
    else:
        missing_names = ", ".join(missing)
        message = (
            f"⚠️ Hồ sơ cho thủ tục '{proc['procedure_name']}' còn thiếu "
            f"{len(missing)} giấy tờ:\n"
            + "\n".join(f"  - {doc}" for doc in missing)
        )

    return {
        "procedure_name": proc["procedure_name"],
        "required_documents": required,
        "submitted_documents": normalized_submitted,
        "missing_documents": missing,
        "is_complete": is_complete,
        "message": message,
    }


def check_documents_from_query(query: str) -> Dict[str, object]:
    """Parse query tự nhiên để kiểm tra hồ sơ.

    Ví dụ input:
        "Kiểm tra hồ sơ đăng ký doanh nghiệp, tôi đã nộp: giấy đề nghị, điều lệ công ty"
        "Tôi đã nhận: giay_de_nghi, dieu_le_cong_ty cho thủ tục đăng ký doanh nghiệp"
    """
    # 1. Tìm thủ tục phù hợp
    procedure = search_procedure(query)
    if not procedure:
        return {
            "procedure_name": "",
            "required_documents": [],
            "submitted_documents": [],
            "missing_documents": [],
            "is_complete": False,
            "message": "Không xác định được thủ tục hành chính. Vui lòng chỉ rõ tên thủ tục.",
        }

    # 2. Parse danh sách hồ sơ đã nộp từ query
    submitted = _parse_submitted_documents(query)

    if not submitted:
        # Nếu không parse được → chỉ trả về danh sách yêu cầu
        proc_id = _find_procedure_id(procedure)
        required = procedure.get("required_documents", [])
        return {
            "procedure_name": procedure["procedure_name"],
            "required_documents": required,
            "submitted_documents": [],
            "missing_documents": required,
            "is_complete": False,
            "message": (
                f"Thủ tục '{procedure['procedure_name']}' yêu cầu các hồ sơ sau:\n"
                + "\n".join(f"  - {doc}" for doc in required)
                + "\n\nVui lòng liệt kê hồ sơ đã nộp để kiểm tra."
            ),
        }

    # 3. Kiểm tra
    proc_id = _find_procedure_id(procedure)
    return check_missing_documents(proc_id, submitted)


def _parse_submitted_documents(query: str) -> List[str]:
    """Trích xuất danh sách hồ sơ đã nộp từ câu hỏi tự nhiên."""
    # Tìm phần sau "đã nộp:", "đã nhận:", "đã có:", "có:"
    patterns = [
        r"(?:đã nộp|đã nhận|đã có|tôi có|tôi nộp|nộp)\s*:\s*(.+)",
        r"(?:bao gồm|gồm có|gồm)\s*:\s*(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            doc_str = match.group(1)
            # Tách bằng dấu phẩy hoặc "và"
            docs = re.split(r"[,;]\s*|\s+và\s+", doc_str)
            return [d.strip() for d in docs if d.strip()]

    return []


def _find_procedure_id(procedure: dict) -> str:
    """Tìm procedure_id từ dict procedure."""
    for proc_id, proc in PROCEDURES.items():
        if proc["procedure_id"] == procedure["procedure_id"]:
            return proc_id
    return ""
