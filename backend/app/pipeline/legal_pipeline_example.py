"""Examples for Vietnamese legal preprocessing pipeline.

Run:
    python -m app.pipeline.legal_pipeline_example
"""

from __future__ import annotations

import sys

from app.pipeline.cleaner import clean_text
from app.pipeline.legal_chunker import chunk_by_article
from app.pipeline.structure_detector import build_legal_tree, detect_structure


EXAMPLES = {
    "45_2024_QH15": """
LUẬT DI SẢN VĂN HÓA
Ch0ơng II Bảo vệ di sản văn hóa
Điề6 6. Bảo vệ di sản văn hóa
1. Nhà nước bảo vệ và phát huy giá trị di sản văn hóa.
a) Bảo đảm nguồn lực bảo tồn.
b) Khuyến khích cộng đồng tham gia bảo vệ.
2. Tổ chức, cá nhân có trách nhiệm chấp hành quy định pháp luật.
Trang 3
""",
    "16_2012_QH13": """
LUẬT QUẢNG CÁO
Chương I. Những quy định chung
Điều 2. Giải thích từ ngữ
1. Quảng cáo là việc sử dụng phương tiện nhằm giới thiệu sản phẩm.
Điều 3. Chính sách của Nhà nước về hoạt động quảng cáo
1. Nhà nước tạo điều kiện cho tổ chức, cá nhân đầu tư phát triển quảng cáo.
""",
    "38_2021_ND-CP": """
NGHỊ ĐỊNH
Chương I. Quy định chung
Điều 5. Điều kiện kinh doanh dịch vụ
1. Tổ chức cung cấp dịch vụ phải đáp ứng điều kiện sau đây:
a) Có đăng ký kinh doanh phù hợp.
b) Có nhân sự chuyên môn theo quy định.
Phụ lục I Mẫu biểu thống kê
Bảng 1   Cột A   Cột B
""",
    "13_2025_TT-BVHTTDL": """
THÔNG TƯ
Chương I - Quy định chung
Điều 7. Hồ sơ đề nghị
1. Hồ sơ đề nghị gồm:
a) Đơn đề nghị theo mẫu.
b) Bản sao giấy tờ pháp lý.
Mẫu số 01
""",
}


def main() -> None:
    # Windows terminals may default to cp1258; force UTF-8 for Vietnamese text.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    for doc_id, raw in EXAMPLES.items():
        cleaned = clean_text(raw)
        structure = detect_structure(cleaned)
        chunks = chunk_by_article(
            structure.articles,
            document_title=doc_id,
            doc_number=doc_id.replace("_", "/"),
            document_type=structure.document_type,
        )
        legal_tree = build_legal_tree(structure, document_name=doc_id)

        print("=" * 80)
        print(f"Document: {doc_id}")
        print(f"Type: {structure.document_type}")
        print(f"Articles: {len(structure.articles)}")
        print(f"Excluded sections: {len(structure.excluded_sections)}")
        if chunks:
            first = chunks[0]
            print("First chunk metadata:")
            print(
                {
                    "document_type": first.document_type,
                    "doc_number": first.doc_number,
                    "year": first.year,
                    "chapter": first.chapter,
                    "article": first.article_number,
                    "clause": first.clause_number,
                }
            )
            print("First chunk text preview:")
            print(first.text[:240])
        print(f"Legal tree chapters: {len(legal_tree['chapters'])}")


if __name__ == "__main__":
    main()
