"""
Module Query Understanding – phân tích câu hỏi người dùng để xác định
intent và trích xuất metadata filters, hỗ trợ pipeline RAG retrieval.

Pipeline tích hợp:
    User Query
    → analyze_query() → compute_intent_bundle + filters/keywords/sort
    → hybrid retrieval (PostgreSQL + Qdrant) → Reranker → LLM
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.query_intent import compute_intent_bundle
from app.services.query_text_patterns import (
    detect_sort_from_patterns,
    document_type_luat_is_false_positive,
    document_type_quyet_dinh_is_false_positive,
    extract_article_number_from_user_query,
    extract_year_from_query_text,
    match_first_mapping_value,
    tokenize_query_words_alnum,
)

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
    # Intent / routing: ``compute_intent_bundle`` + ``intent_detector`` (không regex intent tại đây).

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
        "phu_nu": [
            r"phụ nữ",
            r"người vợ",
            r"vợ\s+đánh",
            r"người mẹ",
            r"nạn nhân bạo lực",
            r"\bvợ\b.*\bchồng\b",
            r"\bchồng\b.*\bvợ\b",
        ],
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
            r"bạo lực gia đình",
            r"đánh đập",
            r"hành hạ",
            r"bạo hành",
            r"ngược đãi",
            r"xâm hại",
            r"vợ\s+đánh\s+chồng",
            r"chồng\s+đánh\s+vợ",
            r"\bđánh\s+chồng\b",
            r"\bđánh\s+vợ\b",
            r"đánh\s+nhau",
            r"thương\s+tích",
            r"bạo\s+hành\s+gia\s+đình",
            r"mâu\s+thuẫn",
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

    def _resolve_routing_intent(self, query: str) -> str:
        """Routing intent — cùng logic với ``compute_intent_bundle`` (query_intent)."""
        return compute_intent_bundle(query)["routing_intent"]

    def detect_intent(self, query: str) -> str:
        """Intent routing cho RAG (checklist + detector + map)."""
        return self._resolve_routing_intent(query)

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
        field = match_first_mapping_value(query_lower, self.FIELD_MAP)
        if field:
            filters["field"] = field

        # Trích xuất cấp chính quyền
        gov_level = match_first_mapping_value(query_lower, self.GOVERNMENT_LEVEL_MAP)
        if gov_level:
            filters["government_level"] = gov_level

        # Trích xuất loại văn bản (tránh nhầm cụm hành chính / ngữ pháp)
        doc_type = match_first_mapping_value(query_lower, self.DOCUMENT_TYPE_MAP)
        if doc_type == "quyet_dinh" and document_type_quyet_dinh_is_false_positive(query_lower):
            doc_type = None  # "quyết định" = thẩm quyền/xử lý, không phải văn bản "Quyết định"
        if doc_type == "luat" and document_type_luat_is_false_positive(query_lower):
            doc_type = None  # "điều luật" = cụm chỉ điều khoản, không phải loại "Luật"
        if doc_type:
            filters["document_type"] = doc_type

        # Trích xuất chức danh
        position = match_first_mapping_value(query_lower, self.POSITION_MAP)
        if position:
            filters["position"] = position

        # Trích xuất trạng thái hiệu lực
        status = match_first_mapping_value(query_lower, self.STATUS_MAP)
        if status:
            filters["status"] = status

        # Trích xuất năm ban hành
        year = extract_year_from_query_text(query_lower)
        if year:
            filters["year"] = year

        # Trích xuất article_number nếu có (e.g. "Điều 47")
        article = extract_article_number_from_user_query(query)
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
        bundle = compute_intent_bundle(query)
        intent = bundle["routing_intent"]
        filters = self.extract_metadata_filters(query)
        keywords = self._extract_keywords(query)
        sort_order = self._detect_sort(query)

        # Không tự động inject status filter vì chunk metadata không chứa trường status.
        # Chỉ filter khi user cung cấp filter rõ ràng hoặc metadata thực sự có trường đó.

        analysis = {
            "intent": intent,
            "detector_intent": bundle["detector_intent"],
            "rag_flags": bundle["rag_flags"],
            "filters": filters,
            "keywords": keywords,
            "sort": sort_order,
        }

        from app.services.intent_detector import COMMUNE_LEVEL_INTENTS
        commune_intents_for_check = set(COMMUNE_LEVEL_INTENTS)
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

    def _extract_keywords(self, query: str) -> List[str]:
        """Trích xuất từ khóa quan trọng từ câu hỏi (loại bỏ stopwords)."""
        # Trích xuất cụm từ quan trọng (bigrams) trước
        bigrams = self._extract_bigrams(query)
        # Tách từ, loại bỏ dấu câu
        words = tokenize_query_words_alnum(query.lower())
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
        words = tokenize_query_words_alnum(query.lower())
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
        found = detect_sort_from_patterns(query_lower, self.SORT_PATTERNS)
        return found or "relevance"

    def analyze_commune_situation(self, query: str) -> Dict[str, Any]:
        """Phân tích tình huống hành chính cấp xã.

        Trích xuất thêm:
            - subject: đối tượng liên quan
            - violation: hành vi vi phạm (nếu có)
            - severity: mức độ ảnh hưởng
        """
        query_lower = query.lower()
        subject = match_first_mapping_value(query_lower, self.SUBJECT_MAP) or "không xác định"
        violation = match_first_mapping_value(query_lower, self.VIOLATION_MAP) or "không có"
        severity = match_first_mapping_value(query_lower, self.SEVERITY_MAP) or "chưa xác định"

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
