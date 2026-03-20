"""
Module Query Understanding – phân tích câu hỏi người dùng để xác định
intent và trích xuất metadata filters, hỗ trợ pipeline RAG retrieval.

Pipeline tích hợp:
    User Query
    → QueryUnderstanding.analyze_query()
    → Extract intent + metadata filters + keywords
    → FAISS search → Metadata filtering → Reranker → Top context → LLM
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class QueryUnderstanding:
    """
    Phân tích câu hỏi người dùng để xác định:
        - intent: loại ý định (legal_lookup, checklist_documents, ...)
        - filters: ràng buộc metadata (field, government_level, ...)
        - keywords: từ khóa quan trọng trích xuất từ câu hỏi
        - sort: hướng sắp xếp kết quả (newest, oldest, relevance)
    """

    # ================================================================
    #  Bảng ánh xạ keyword → giá trị chuẩn hóa
    # ================================================================

    # Từ khóa nhận diện intent
    INTENT_PATTERNS: Dict[str, List[str]] = {
        "document_summary": [
            r"(luật|nghị\s*định|thông\s*tư|quyết\s*định|văn bản)\s+.*\b(bao gồm|gồm có|gồm những|có những)\b",
            r"(luật|nghị\s*định|thông\s*tư|quyết\s*định)\s+.*\b(nội dung|quy định)\s+(gì|những gì|nào)",
            r"\b(tóm tắt|tổng quan|mục lục|danh sách các điều)\b\s+.*(luật|nghị\s*định|thông\s*tư|văn bản)",
            r"(luật|nghị\s*định|thông\s*tư|quyết\s*định)\s+.*\bcó\s+(bao nhiêu|mấy)\s+(điều|chương)",
            r"\b(các điều|những điều|các chương)\b\s+.*(luật|nghị\s*định|thông\s*tư)",
            r"(luật|nghị\s*định|thông\s*tư|quyết\s*định)\s+.*\b(gồm|bao gồm)\b",
            r"\d+[/_]\d{4}[/_]\S+\s+.*(bao gồm|gồm|quy định\s+(gì|những gì|nào))",
            r"(liệt kê|cho biết)\s+(các\s+)?(điều\s+)(trong\s+)?(luật|nghị\s*định|thông\s*tư)",
        ],
        "article_query": [
            r"điều\s+\d+\s+(quy định|nêu|nói|ghi)",
            r"(quy định|nội dung)\s+(của\s+)?điều\s+\d+",
            r"(cấm|cho phép|bắt buộc)\s+.*(sản phẩm|hành vi|hoạt động)",
        ],
        "document_metadata": [
            r"ban hành\s+(ngày|năm|khi)\s+nào",
            r"(ngày|năm)\s+ban hành",
            r"(ai|cơ quan nào)\s+ban hành",
            r"(số hiệu|số)\s+(văn bản|quyết định|nghị định|thông tư)",
            r"(quyết định|nghị định|thông tư|luật)\s+.*(ban hành|có hiệu lực|hết hiệu lực)\s+.*(ngày|năm|khi|bao giờ)",
            r"(hiệu lực|có hiệu lực|hết hiệu lực)\s+(từ|ngày|khi)\s+nào",
        ],
        "program_goal": [
            r"mục tiêu\s+(của\s+)?(chương trình|đề án|kế hoạch|dự án)",
            r"(chương trình|đề án|kế hoạch|dự án)\s+.*(mục tiêu|mục đích|nhằm)",
            r"(mục đích|mục tiêu|nhiệm vụ)\s+(chính|chủ yếu|cơ bản)\s+(của|là)",
        ],
        "document_relation": [
            r"(sửa đổi|bổ sung|thay thế|bãi bỏ)\s+(những|các)?\s*(luật|nghị định|thông tư|văn bản|quy định)",
            r"(luật|nghị định|thông tư)\s+.*(sửa đổi|bổ sung|thay thế)\s+(những|các)?\s*(luật|nghị định|thông tư|văn bản)",
            r"(văn bản|luật)\s+(nào|gì)\s+(bị|được)\s+(sửa đổi|thay thế|bãi bỏ)",
        ],
        "checklist_documents": [
            r"liệt kê", r"danh sách", r"các văn bản", r"những văn bản",
            r"tổng hợp", r"thống kê", r"bao nhiêu văn bản",
            r"cho biết các", r"kể tên",
            r"check\s*list", r"checklist", r"văn bản liên quan",
            r"văn bản nào", r"có những văn bản",
        ],
        "can_cu_phap_ly": [
            r"căn cứ", r"dựa (vào|trên|theo)\s+(luật|văn bản|quy định)",
            r"cơ sở pháp lý", r"căn cứ pháp lý",
            r"văn bản (trên|đó|này)\s+(dựa|căn cứ)",
            r"(dựa|căn cứ)\s+(vào|theo)\s+luật nào",
            r"luật nào (quy định|điều chỉnh|áp dụng)",
            r"theo luật nào", r"căn cứ vào đâu",
        ],
        "document_drafting": [
            r"soạn thảo", r"viết mẫu", r"tạo văn bản", r"mẫu đơn",
            r"biên bản", r"tờ trình", r"công văn mẫu", r"dự thảo",
            r"lập văn bản", r"viết công văn",
            r"soạn (kế hoạch|quyết định|thông báo|báo cáo)",
        ],
        "giai_thich_quy_dinh": [
            r"giải thích", r"nghĩa là (gì|sao)",
            r"hiểu (như thế nào|thế nào|ra sao)",
            r"có nghĩa", r"ý nghĩa",
            r"áp dụng (như thế nào|thế nào|ra sao)",
            r"tại sao", r"vì sao", r"phân tích", r"diễn giải",
        ],
        "admin_planning": [
            r"(lập|xây dựng|soạn)\s+(kế hoạch|phương án)\s+(quản lý|triển khai|thực hiện)",
            r"(tổ chức|triển khai)\s+(thực hiện|quản lý)\s+.*(tại|ở|trên địa bàn)",
            r"(bố trí|phân bổ)\s+(nhân (sự|lực)|nguồn lực|cán bộ|biên chế)",
            r"(cơ cấu|tổ chức)\s+(bộ máy|nhân sự|chính quyền)",
            r"(quy hoạch|kế hoạch)\s+(quản lý|phát triển|triển khai|sử dụng)",
            r"(giám sát|kiểm tra)\s+(việc\s+)?(thực hiện|triển khai|quản lý)",
            r"(biện pháp|giải pháp)\s+(quản lý|tổ chức|triển khai)",
            r"quản lý\s+(hành chính|nhà nước)\s+.*(tại|ở|trên)",
        ],
        "legal_lookup": [
            r"quy định", r"điều\s+\d+", r"khoản\s+\d+", r"luật\s+",
            r"nghị định", r"thông tư", r"chức năng", r"nhiệm vụ",
            r"quyền hạn", r"trách nhiệm", r"thẩm quyền", r"xử phạt",
            r"chế tài", r"hướng dẫn", r"theo quy định", r"pháp luật",
            r"căn cứ", r"cơ sở pháp lý", r"điều kiện", r"thủ tục",
            r"quy trình", r"UBND", r"ủy ban",
            r"quy định mới nhất", r"hiện hành", r"còn hiệu lực",
        ],
        # ── Commune-level intents (Cán bộ VHXH cấp xã) ──────
        "xu_ly_vi_pham_hanh_chinh": [
            r"karaoke\s+(gây|ồn|tiếng\s+ồn|quá\s+giờ)",
            r"(gây|tiếng)\s+ồn",
            r"vi phạm\s+(hành chính|quảng cáo|văn hóa|lễ hội)",
            r"xử phạt\s+(hành chính|vi phạm|karaoke|tiếng ồn)",
            r"(internet|game)\s+.*(vi phạm|không phép|học sinh)",
            r"mê tín\s+dị\s+đoan",
        ],
        "kiem_tra_thanh_tra": [
            r"kiểm tra\s+(cơ sở|quán|karaoke|internet|game|dịch vụ)",
            r"thanh tra\s+(văn hóa|cơ sở|hoạt động)",
            r"(đoàn|tổ)\s+kiểm tra",
            r"(rà soát|kiểm tra)\s+(định kỳ|đột xuất)",
        ],
        "thu_tuc_hanh_chinh": [
            r"(đăng ký|xin phép|cấp phép)\s+(lễ hội|biểu diễn|sự kiện|quảng cáo)",
            r"(tu bổ|tôn tạo|sửa chữa)\s+(di tích|đình|chùa|miếu)",
            r"đăng ký\s+(hoạt động|sinh hoạt)\s+tôn giáo",
            r"công nhận\s+(di tích|di sản|làng văn hóa|gia đình văn hóa)",
        ],
        "hoa_giai_van_dong": [
            r"hòa giải\s+(tranh chấp|mâu thuẫn|xung đột)",
            r"vận động\s+(người dân|nhân dân|cộng đồng)",
            r"(xây dựng|thực hiện)\s+(nếp sống|đời sống)\s+(văn hóa|văn minh)",
            r"(hương ước|quy ước)\s+(thôn|xóm|khu dân cư)",
        ],
        "bao_ve_xa_hoi": [
            r"bạo lực\s+gia đình",
            r"(xâm hại|bạo hành|ngược đãi)\s+(trẻ em|phụ nữ|người già)",
            r"(trẻ em|học sinh)\s+(bỏ học|lang thang|chơi game|nghiện)",
            r"(phòng chống|ngăn chặn)\s+(bạo lực gia đình|tệ nạn)",
        ],
        "to_chuc_su_kien_cong": [
            r"(tổ chức|chuẩn bị)\s+(lễ hội|sự kiện|hội thi|hội diễn)",
            r"(lễ hội|sự kiện)\s+(văn hóa|thể thao|truyền thống)",
            r"(biểu diễn|văn nghệ)\s+(công cộng|quần chúng)",
        ],
        "bao_ton_phat_trien": [
            r"(bảo tồn|gìn giữ|phát huy)\s+(di sản|di tích|văn hóa)",
            r"(đình|chùa|miếu)\s+.*(xuống cấp|tu bổ|tôn tạo)",
            r"(làng nghề|nghề truyền thống)\s+.*(bảo tồn|phát triển)",
        ],
        # general_question là default, không cần pattern
    }

    # Ánh xạ lĩnh vực (field)
    FIELD_MAP: Dict[str, List[str]] = {
        "van_hoa_the_thao": [
            r"văn hóa", r"thể thao", r"nghệ thuật", r"di sản",
            r"lễ hội", r"du lịch", r"biểu diễn", r"bảo tàng",
            r"thư viện", r"di tích",
        ],
        "giao_duc": [
            r"giáo dục", r"đào tạo", r"trường học", r"học sinh",
            r"sinh viên", r"giáo viên", r"chương trình học",
            r"tuyển sinh", r"bằng cấp", r"phổ thông", r"đại học",
        ],
        "y_te": [
            r"y tế", r"sức khỏe", r"bệnh viện", r"khám chữa bệnh",
            r"dược", r"bảo hiểm y tế", r"phòng chống dịch",
            r"vệ sinh", r"an toàn thực phẩm",
        ],
        "dat_dai": [
            r"đất đai", r"nhà ở", r"bất động sản", r"quy hoạch",
            r"xây dựng", r"cấp phép xây dựng", r"sổ đỏ",
            r"giải phóng mặt bằng", r"thu hồi đất",
        ],
        "tai_chinh": [
            r"tài chính", r"ngân sách", r"thuế", r"phí", r"lệ phí",
            r"kế toán", r"kiểm toán", r"đầu tư", r"chi ngân sách",
        ],
        "lao_dong": [
            r"lao động", r"việc làm", r"tiền lương", r"bảo hiểm xã hội",
            r"hợp đồng lao động", r"an toàn lao động", r"nghỉ phép",
            r"thai sản", r"hưu trí",
        ],
        "hanh_chinh": [
            r"hành chính", r"thủ tục hành chính", r"cải cách",
            r"một cửa", r"công chức", r"viên chức", r"cán bộ",
            r"hộ tịch", r"chứng thực",
        ],
        "moi_truong": [
            r"môi trường", r"tài nguyên", r"khoáng sản", r"nước",
            r"rừng", r"biển", r"ô nhiễm", r"xử lý rác",
            r"bảo vệ môi trường",
        ],
        "an_ninh_quoc_phong": [
            r"an ninh", r"quốc phòng", r"quân sự", r"công an",
            r"trật tự", r"phòng cháy", r"cứu hộ", r"dân quân",
        ],
        "nong_nghiep": [
            r"nông nghiệp", r"nông thôn", r"thủy sản", r"chăn nuôi",
            r"trồng trọt", r"lâm nghiệp", r"thủy lợi",
        ],
    }

    # Ánh xạ cấp chính quyền (government_level)
    GOVERNMENT_LEVEL_MAP: Dict[str, List[str]] = {
        "xa": [r"xã", r"phường", r"thị trấn", r"UBND xã", r"UBND phường"],
        "huyen": [r"huyện", r"quận", r"thị xã", r"thành phố thuộc tỉnh"],
        "tinh": [r"tỉnh", r"thành phố trực thuộc", r"UBND tỉnh", r"cấp tỉnh"],
        "trung_uong": [r"trung ương", r"chính phủ", r"quốc hội", r"bộ\s"],
    }

    # Ánh xạ loại văn bản (document_type)
    DOCUMENT_TYPE_MAP: Dict[str, List[str]] = {
        "luat": [r"\bluật\b"],
        "nghi_dinh": [r"nghị định"],
        "thong_tu": [r"thông tư"],
        "quyet_dinh": [r"quyết định"],
        "nghi_quyet": [r"nghị quyết"],
        "cong_van": [r"công văn"],
        "chi_thi": [r"chỉ thị"],
        "thong_bao": [r"thông báo"],
        "ke_hoach": [r"kế hoạch"],
    }

    # Từ khóa trạng thái hiệu lực (status)
    STATUS_MAP: Dict[str, List[str]] = {
        "effective": [
            r"hiệu lực", r"còn hiệu lực", r"đang có hiệu lực",
            r"hiện hành", r"đang áp dụng",
        ],
        "expired": [
            r"hết hiệu lực", r"không còn hiệu lực", r"đã bãi bỏ",
            r"bị thay thế", r"ngừng áp dụng",
        ],
    }

    # Từ khóa sắp xếp (sort)
    SORT_PATTERNS: Dict[str, List[str]] = {
        "newest": [r"mới nhất", r"gần đây", r"gần nhất", r"mới ban hành"],
        "oldest": [r"cũ nhất", r"đầu tiên", r"sớm nhất"],
    }

    # Ánh xạ chức danh (position/role)
    POSITION_MAP: Dict[str, List[str]] = {
        "pho_chu_tich_ubnd": [
            r"phó chủ tịch UBND", r"phó chủ tịch ủy ban",
            r"PCT UBND", r"phó chủ tịch",
        ],
        "chu_tich_ubnd": [
            r"chủ tịch UBND", r"chủ tịch ủy ban nhân dân",
            r"CT UBND",
        ],
        "bi_thu": [
            r"bí thư đảng ủy", r"bí thư chi bộ", r"bí thư",
        ],
        "truong_cong_an": [
            r"trưởng công an", r"công an xã", r"công an phường",
        ],
        "chi_huy_truong_quan_su": [
            r"chỉ huy trưởng quân sự", r"ban chỉ huy quân sự",
        ],
        "can_bo_tu_phap": [
            r"cán bộ tư pháp", r"tư pháp hộ tịch",
        ],
        "can_bo_dia_chinh": [
            r"cán bộ địa chính", r"địa chính xây dựng",
        ],
        "can_bo_van_hoa": [
            r"cán bộ văn hóa", r"văn hóa xã hội",
        ],
    }

    # ── Commune-level situation analysis maps ──────────────
    SUBJECT_MAP: Dict[str, List[str]] = {
        "nguoi_dan": [
            r"người dân", r"công dân", r"nhân dân", r"hộ dân",
            r"hàng xóm", r"láng giềng", r"chủ hộ",
        ],
        "doanh_nghiep": [
            r"doanh nghiệp", r"cơ sở kinh doanh", r"chủ cơ sở",
            r"quán", r"nhà hàng", r"khách sạn", r"cửa hàng",
            r"công ty", r"hộ kinh doanh",
        ],
        "hoc_sinh": [
            r"học sinh", r"sinh viên", r"trẻ em", r"thanh niên",
            r"thiếu niên", r"vị thành niên", r"trẻ vị thành niên",
        ],
        "to_chuc_ton_giao": [
            r"nhà chùa", r"nhà thờ", r"tổ chức tôn giáo", r"cơ sở tôn giáo",
            r"ban trị sự", r"hội đồng mục vụ", r"tín đồ",
        ],
        "can_bo": [
            r"cán bộ", r"công chức", r"viên chức",
            r"UBND", r"ủy ban", r"chính quyền",
        ],
        "phu_nu": [r"phụ nữ", r"người vợ", r"người mẹ", r"nạn nhân bạo lực"],
        "nguoi_cao_tuoi": [r"người già", r"người cao tuổi", r"cụ già", r"neo đơn"],
    }

    VIOLATION_MAP: Dict[str, List[str]] = {
        "tieng_on": [
            r"tiếng ồn", r"gây ồn", r"ồn ào", r"karaoke", r"nhạc to",
            r"hát quá giờ", r"loa (kéo|thùng|phóng thanh)",
        ],
        "quang_cao_trai_phep": [
            r"quảng cáo trái phép", r"quảng cáo không phép",
            r"biển quảng cáo", r"băng rôn trái phép", r"dán tờ rơi",
        ],
        "vi_pham_le_hoi": [
            r"vi phạm lễ hội", r"mê tín dị đoan", r"lợi dụng lễ hội",
            r"đốt vàng mã", r"chen lấn xô đẩy",
        ],
        "bao_luc_gia_dinh": [
            r"bạo lực gia đình", r"đánh đập", r"hành hạ",
            r"bạo hành", r"ngược đãi", r"xâm hại",
        ],
        "kinh_doanh_trai_phep": [
            r"không phép", r"không giấy phép", r"chưa đăng ký",
            r"kinh doanh trái phép", r"hoạt động trái phép",
        ],
        "te_nan_xa_hoi": [
            r"ma túy", r"cờ bạc", r"mại dâm", r"tệ nạn",
            r"nghiện", r"sử dụng chất cấm",
        ],
        "xam_pham_di_tich": [
            r"xâm phạm di tích", r"phá hoại di tích", r"lấn chiếm di tích",
            r"xây dựng trái phép.*di tích",
        ],
    }

    SEVERITY_MAP: Dict[str, List[str]] = {
        "nghiem_trong": [
            r"nghiêm trọng", r"nguy hiểm", r"đe dọa tính mạng",
            r"thương tích", r"tử vong", r"gây thương tích",
            r"nhiều lần", r"tái phạm", r"có tổ chức",
        ],
        "trung_binh": [
            r"nhiều lần phản ánh", r"kéo dài", r"ảnh hưởng",
            r"gây bức xúc", r"người dân phản ánh",
        ],
        "nhe": [
            r"lần đầu", r"vi phạm nhỏ", r"không đáng kể",
            r"nhắc nhở", r"cảnh cáo",
        ],
    }

    # Stopwords tiếng Việt (dùng để trích xuất keywords)
    STOPWORDS = {
        "và", "của", "là", "các", "cho", "có", "được", "trong", "với",
        "này", "đó", "một", "về", "theo", "từ", "đến", "trên", "tại",
        "những", "không", "cũng", "như", "hay", "hoặc", "nếu", "khi",
        "thì", "đã", "sẽ", "đang", "rằng", "bởi", "vì", "nên", "mà",
        "để", "do", "bằng", "hơn", "nhất", "rất", "lại", "ra", "vào",
        "lên", "xuống", "nào", "gì", "ai", "đâu", "bao", "mấy",
        "liệt", "kê", "cho", "biết", "hãy", "xin", "vui", "lòng",
        "tôi", "check", "list", "liên", "quan", "yêu", "cầu", "mới",
        "thi", "hiện", "nay", "lĩnh", "vực", "ở", "còn", "đang",
        "văn", "bản", "chức", "năng", "nhiệm", "vụ", "quản", "lý",
        "điều", "hành", "phó", "chủ", "tịch", "thể", "hóa", "thao",
        "hiệu", "lực", "cấp",
    }

    # ================================================================
    #  Phương thức chính
    # ================================================================

    def detect_intent(self, query: str) -> str:
        """
        Phân loại ý định (intent) câu hỏi dựa trên keyword matching.

        Trả về một trong:
            - "checklist_documents": liệt kê văn bản
            - "document_drafting": soạn thảo văn bản
            - "legal_lookup": tra cứu pháp luật
            - "general_question": câu hỏi chung

        Ưu tiên theo thứ tự: checklist > drafting > legal > general.
        """
        query_lower = query.lower()

        # Duyệt theo thứ tự ưu tiên
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Guard: "Điều X" in query → article_query, not summary
                    if intent == "document_summary" and re.search(r"điều\s+\d+", query_lower):
                        return "article_query"
                    log.debug("Intent detected: %s (matched: %s)", intent, pattern)
                    return intent

        return "general_question"

    def extract_metadata_filters(self, query: str) -> Dict[str, Any]:
        """
        Trích xuất ràng buộc metadata từ câu hỏi.

        Trả về dict có thể chứa:
            - field: lĩnh vực (van_hoa_the_thao, giao_duc, ...)
            - government_level: cấp chính quyền (xa, huyen, tinh, trung_uong)
            - document_type: loại văn bản (luat, nghi_dinh, ...)
            - status: trạng thái hiệu lực (effective, expired)
            - year: năm ban hành (int)
        """
        query_lower = query.lower()
        filters: Dict[str, Any] = {}

        # Trích xuất lĩnh vực
        field = self._match_first(query_lower, self.FIELD_MAP)
        if field:
            filters["field"] = field

        # Trích xuất cấp chính quyền
        gov_level = self._match_first(query_lower, self.GOVERNMENT_LEVEL_MAP)
        if gov_level:
            filters["government_level"] = gov_level

        # Trích xuất loại văn bản
        doc_type = self._match_first(query_lower, self.DOCUMENT_TYPE_MAP)
        if doc_type:
            filters["document_type"] = doc_type

        # Trích xuất chức danh
        position = self._match_first(query_lower, self.POSITION_MAP)
        if position:
            filters["position"] = position

        # Trích xuất trạng thái hiệu lực
        status = self._match_first(query_lower, self.STATUS_MAP)
        if status:
            filters["status"] = status

        # Trích xuất năm ban hành
        year = self._extract_year(query_lower)
        if year:
            filters["year"] = year

        # Trích xuất article_number nếu có (e.g. "Điều 47")
        article = self._extract_article_number(query)
        if article:
            filters["article_number"] = article

        return filters

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Phân tích toàn diện câu hỏi người dùng.

        Trả về:
            {
                "intent": "legal_lookup",
                "filters": {
                    "field": "van_hoa_the_thao",
                    "government_level": "xa",
                    ...
                },
                "keywords": ["chức năng", "phó chủ tịch", ...],
                "sort": "relevance"
            }
        """
        intent = self.detect_intent(query)
        filters = self.extract_metadata_filters(query)
        keywords = self._extract_keywords(query)
        sort_order = self._detect_sort(query)

        # Không tự động inject status filter vì chunk metadata không chứa trường status.
        # Chỉ filter khi user cung cấp filter rõ ràng hoặc metadata thực sự có trường đó.

        analysis = {
            "intent": intent,
            "filters": filters,
            "keywords": keywords,
            "sort": sort_order,
        }

        from app.services.intent_detector import COMMUNE_LEVEL_INTENTS
        commune_intents_for_check = COMMUNE_LEVEL_INTENTS | {
            "xu_ly_vi_pham_hanh_chinh", "kiem_tra_thanh_tra",
            "thu_tuc_hanh_chinh", "hoa_giai_van_dong",
            "bao_ve_xa_hoi", "to_chuc_su_kien_cong", "bao_ton_phat_trien",
        }
        situation = self.analyze_commune_situation(query)
        has_commune_signal = (
            situation["violation"] != "không có"
            or situation["subject"] != "không xác định"
        )
        if intent in commune_intents_for_check or has_commune_signal:
            analysis["commune_situation"] = situation

        log.info("Query analysis: %s → %s", query, analysis)
        return analysis

    # ================================================================
    #  Helpers nội bộ
    # ================================================================

    @staticmethod
    def _match_first(
        text: str,
        mapping: Dict[str, List[str]],
    ) -> Optional[str]:
        """Tìm giá trị đầu tiên khớp pattern trong mapping."""
        for value, patterns in mapping.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return value
        return None

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        """Trích xuất năm ban hành từ câu hỏi (ví dụ: năm 2023, 2024)."""
        # Pattern: "năm YYYY" hoặc số năm đứng độc lập (2000-2099)
        match = re.search(r"năm\s+(20\d{2})", text)
        if match:
            return int(match.group(1))
        # Fallback: tìm năm đứng riêng biệt trong câu
        match = re.search(r"\b(20\d{2})\b", text)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _extract_article_number(query: str) -> Optional[str]:
        """Trích xuất số Điều từ câu hỏi (ví dụ: 'Điều 47' → '47')."""
        match = re.search(r"điều\s+(\d+[A-Za-z]?)", query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_keywords(self, query: str) -> List[str]:
        """Trích xuất từ khóa quan trọng từ câu hỏi (loại bỏ stopwords)."""
        # Trích xuất cụm từ quan trọng (bigrams) trước
        bigrams = self._extract_bigrams(query)
        # Tách từ, loại bỏ dấu câu
        words = re.findall(r"[\w]+", query.lower())
        # Loại bỏ stopwords và từ quá ngắn (< 3 ký tự)
        keywords = [
            w for w in words
            if w not in self.STOPWORDS and len(w) >= 3 and not w.isdigit()
        ]
        # Gộp và loại trùng (ưu tiên bigrams)
        seen = set()
        result = []
        for phrase in bigrams + keywords:
            if phrase not in seen:
                seen.add(phrase)
                result.append(phrase)
        return result

    @staticmethod
    def _extract_bigrams(query: str) -> List[str]:
        """Trích xuất cụm 2 từ liền kề có ý nghĩa."""
        words = re.findall(r"[\w]+", query.lower())
        bigrams = []
        # Các cụm từ quan trọng trong bối cảnh pháp luật VN
        important_bigrams = {
            "phó chủ", "chủ tịch", "hội đồng", "ủy ban", "nhân dân",
            "quản lý", "cấp phép", "xử phạt", "vi phạm", "hành chính",
            "văn bản", "quy phạm", "pháp luật", "an ninh", "trật tự",
            "bảo hiểm", "xã hội", "lao động", "môi trường",
            "chức năng", "nhiệm vụ", "quyền hạn", "trách nhiệm",
            "văn hóa", "thể thao", "thể dục", "quản lý điều",
            "điều hành", "nhà nước", "chính quyền", "hiệu lực",
            "cấp xã", "cấp huyện", "cấp tỉnh",
        }
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            if bigram in important_bigrams:
                bigrams.append(bigram)
        return bigrams

    def _detect_sort(self, query: str) -> str:
        """Phát hiện hướng sắp xếp kết quả mong muốn."""
        query_lower = query.lower()
        for sort_type, patterns in self.SORT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return sort_type
        return "relevance"

    def analyze_commune_situation(self, query: str) -> Dict[str, Any]:
        """Phân tích tình huống hành chính cấp xã.

        Trích xuất thêm:
            - subject: đối tượng liên quan
            - violation: hành vi vi phạm (nếu có)
            - severity: mức độ ảnh hưởng
        """
        query_lower = query.lower()
        subject = self._match_first(query_lower, self.SUBJECT_MAP) or "không xác định"
        violation = self._match_first(query_lower, self.VIOLATION_MAP) or "không có"
        severity = self._match_first(query_lower, self.SEVERITY_MAP) or "chưa xác định"

        return {
            "subject": subject,
            "violation": violation,
            "severity": severity,
        }


# ── Singleton instance ─────────────────────────────────────
_query_understanding = QueryUnderstanding()


# ── Module-level functions ─────────────────────────────────
def analyze_query(query: str) -> Dict[str, Any]:
    """Phân tích câu hỏi – shortcut cho singleton."""
    return _query_understanding.analyze_query(query)


def detect_intent(query: str) -> str:
    """Phát hiện intent – shortcut cho singleton."""
    return _query_understanding.detect_intent(query)


def extract_metadata_filters(query: str) -> Dict[str, Any]:
    """Trích xuất metadata filters – shortcut cho singleton."""
    return _query_understanding.extract_metadata_filters(query)


def analyze_commune_situation(query: str) -> Dict[str, Any]:
    """Phân tích tình huống cấp xã – shortcut cho singleton."""
    return _query_understanding.analyze_commune_situation(query)
