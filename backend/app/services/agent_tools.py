"""
AI Agent Tools – hệ thống tools cho LLM gọi.

Cung cấp các tool functions mà AI Agent có thể sử dụng:
    - search_document: Tìm kiếm văn bản pháp luật
    - get_procedure_steps: Lấy các bước thủ tục hành chính
    - check_documents: Kiểm tra hồ sơ
    - summarize_document: Tóm tắt văn bản
    - compare_documents: So sánh văn bản
    - generate_report: Tạo báo cáo

Mỗi tool được định nghĩa với:
    - name: tên tool
    - description: mô tả chức năng
    - parameters: tham số đầu vào
    - function: hàm thực thi
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


# ── Tool Definition ─────────────────────────────────────────
class AgentTool:
    """Định nghĩa một tool cho AI Agent."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    def to_openai_function(self) -> dict:
        """Chuyển sang format OpenAI function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ── Tool Functions ──────────────────────────────────────────

async def tool_search_document(query: str, top_k: int = 8) -> Dict[str, Any]:
    """Tìm kiếm văn bản pháp luật trong cơ sở dữ liệu."""
    from app.services.retrieval import search_all

    results = await search_all(query, top_k=top_k)
    return {
        "tool": "search_document",
        "results": [
            {
                "text": doc["text"][:500],
                "score": round(doc.get("rerank_score", doc.get("score", 0)), 4),
                "metadata": doc.get("metadata", {}),
            }
            for doc in results
        ],
        "total_found": len(results),
    }


async def tool_get_procedure_steps(procedure_name: str) -> Dict[str, Any]:
    """Lấy các bước thực hiện thủ tục hành chính."""
    from app.services.procedure_service import search_procedure, list_procedures

    procedure = search_procedure(procedure_name)
    if procedure:
        return {
            "tool": "get_procedure_steps",
            "found": True,
            "procedure": procedure,
        }
    else:
        available = list_procedures()
        return {
            "tool": "get_procedure_steps",
            "found": False,
            "message": f"Không tìm thấy thủ tục '{procedure_name}'.",
            "available_procedures": [p["procedure_name"] for p in available],
        }


async def tool_check_documents(
    procedure_name: str,
    submitted_documents: List[str],
) -> Dict[str, Any]:
    """Kiểm tra hồ sơ còn thiếu."""
    from app.services.document_checker import check_missing_documents
    from app.services.procedure_service import search_procedure

    procedure = search_procedure(procedure_name)
    if not procedure:
        return {
            "tool": "check_documents",
            "error": f"Không tìm thấy thủ tục '{procedure_name}'.",
        }

    # Tìm procedure_id
    from app.services.procedure_service import PROCEDURES
    proc_id = ""
    for pid, proc in PROCEDURES.items():
        if proc["procedure_id"] == procedure["procedure_id"]:
            proc_id = pid
            break

    result = check_missing_documents(proc_id, submitted_documents)
    return {"tool": "check_documents", **result}


async def tool_summarize_document(query: str) -> Dict[str, Any]:
    """Tóm tắt văn bản pháp luật."""
    from app.services.document_summarizer import summarize_document

    result = await summarize_document(query)
    return {
        "tool": "summarize_document",
        "summary": result["summary"],
        "source_count": len(result["sources"]),
    }


async def tool_compare_documents(query: str) -> Dict[str, Any]:
    """So sánh hai văn bản pháp luật."""
    from app.services.document_comparator import compare_documents

    result = await compare_documents(query)
    return {
        "tool": "compare_documents",
        "comparison": result["comparison"],
        "source_count": len(result.get("sources", [])),
    }


async def tool_generate_report(request: str) -> Dict[str, Any]:
    """Tạo báo cáo hành chính."""
    from app.services.report_generator import generate_report

    result = await generate_report(request)
    return {
        "tool": "generate_report",
        "report": result["report"],
        "source_count": len(result["sources"]),
    }


async def tool_search_ninh_binh_info(query: str) -> Dict[str, Any]:
    """Tra cứu thông tin chung về tỉnh Ninh Bình (địa lý, du lịch, dân số, kinh tế, ...)."""
    from app.tools.ninh_binh_search_tool import run as ninh_binh_run

    result = await ninh_binh_run(query)
    return {
        "tool": "search_ninh_binh_info",
        "result": result.get("result", ""),
        "sources": result.get("sources", []),
    }


async def tool_search_web(query: str) -> Dict[str, Any]:
    """Search internet via OpenAI web_search for non-legal fallback."""
    from app.tools.openai_web_search_tool import run as openai_web_search_run

    result = await openai_web_search_run(query)
    return {
        "tool": "search_web",
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }


async def tool_local_data_lookup(
    query: str,
    data_type: str = "general",
) -> Dict[str, Any]:
    """Tra cứu thông tin hành chính địa phương (dân số, diện tích, đơn vị hành chính)."""
    from app.services.local_data_service import lookup_local_data

    result = await lookup_local_data(query, data_type=data_type)
    return {
        "tool": "local_data_lookup",
        "location": result.get("location", ""),
        "admin_level": result.get("admin_level", "unknown"),
        "data": result.get("data", ""),
        "sources": result.get("sources", []),
    }


async def tool_statistics_estimator(
    population: Optional[int] = None,
    area: Optional[float] = None,
    admin_level: str = "xa",
    context: str = "",
) -> Dict[str, Any]:
    """Ước tính quy mô quản lý và nguồn lực dựa trên dân số, diện tích."""
    from app.services.statistics_estimator import estimate_resources

    result = await estimate_resources(
        population=population,
        area=area,
        admin_level=admin_level,
        context=context,
    )
    return {
        "tool": "statistics_estimator",
        "estimation": result.get("estimation", ""),
        "input": result.get("input", {}),
    }


# ── Tool Registry ──────────────────────────────────────────

TOOLS: List[AgentTool] = [
    AgentTool(
        name="search_document",
        description="Tìm kiếm văn bản pháp luật trong cơ sở dữ liệu. Dùng khi cần tra cứu quy định, điều luật, nghị định.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Câu truy vấn tìm kiếm"},
                "top_k": {"type": "integer", "description": "Số kết quả trả về", "default": 8},
            },
            "required": ["query"],
        },
        function=tool_search_document,
    ),
    AgentTool(
        name="get_procedure_steps",
        description="Lấy các bước thực hiện thủ tục hành chính. Dùng khi cán bộ cần hướng dẫn quy trình.",
        parameters={
            "type": "object",
            "properties": {
                "procedure_name": {"type": "string", "description": "Tên thủ tục cần tra cứu"},
            },
            "required": ["procedure_name"],
        },
        function=tool_get_procedure_steps,
    ),
    AgentTool(
        name="check_documents",
        description="Kiểm tra hồ sơ còn thiếu so với yêu cầu thủ tục. Dùng khi cần xác minh hồ sơ đầy đủ.",
        parameters={
            "type": "object",
            "properties": {
                "procedure_name": {"type": "string", "description": "Tên thủ tục"},
                "submitted_documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Danh sách hồ sơ đã nộp",
                },
            },
            "required": ["procedure_name", "submitted_documents"],
        },
        function=tool_check_documents,
    ),
    AgentTool(
        name="summarize_document",
        description="Tóm tắt văn bản pháp luật. Dùng khi cần nắm nhanh nội dung chính của văn bản.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Tên hoặc mô tả văn bản cần tóm tắt"},
            },
            "required": ["query"],
        },
        function=tool_summarize_document,
    ),
    AgentTool(
        name="compare_documents",
        description="So sánh hai văn bản pháp luật. Dùng khi cần phân tích sự khác biệt giữa các văn bản.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Yêu cầu so sánh, VD: 'So sánh nghị định A và nghị định B'"},
            },
            "required": ["query"],
        },
        function=tool_compare_documents,
    ),
    AgentTool(
        name="generate_report",
        description="Tạo báo cáo hành chính. Dùng khi cần soạn thảo báo cáo tổng hợp.",
        parameters={
            "type": "object",
            "properties": {
                "request": {"type": "string", "description": "Yêu cầu tạo báo cáo"},
            },
            "required": ["request"],
        },
        function=tool_generate_report,
    ),
    AgentTool(
        name="search_ninh_binh_info",
        description="Search general information about Ninh Binh province such as tourism, population, geography, culture, economy, administrative divisions. Not for legal questions.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User question about Ninh Binh"},
            },
            "required": ["query"],
        },
        function=tool_search_ninh_binh_info,
    ),
    AgentTool(
        name="search_web",
        description="Search the internet using OpenAI web search for general information when Wikipedia is insufficient.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User question"},
            },
            "required": ["query"],
        },
        function=tool_search_web,
    ),
    AgentTool(
        name="local_data_lookup",
        description="Tra cứu thông tin hành chính địa phương: dân số, diện tích, đơn vị hành chính, số thôn/xóm. Dùng khi câu hỏi liên quan đến địa bàn cụ thể.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Tên địa phương hoặc câu hỏi về địa bàn",
                },
                "data_type": {
                    "type": "string",
                    "enum": ["general", "population", "area", "admin_units"],
                    "description": "Loại dữ liệu cần tra cứu",
                    "default": "general",
                },
            },
            "required": ["query"],
        },
        function=tool_local_data_lookup,
    ),
    AgentTool(
        name="statistics_estimator",
        description="Ước tính quy mô quản lý và nguồn lực cần thiết dựa trên dân số, diện tích, cấp hành chính. Dùng cho bài toán quy hoạch, bố trí nhân sự.",
        parameters={
            "type": "object",
            "properties": {
                "population": {
                    "type": "integer",
                    "description": "Dân số (người)",
                },
                "area": {
                    "type": "number",
                    "description": "Diện tích (km²)",
                },
                "admin_level": {
                    "type": "string",
                    "enum": ["xa", "huyen", "tinh"],
                    "description": "Cấp hành chính",
                    "default": "xa",
                },
                "context": {
                    "type": "string",
                    "description": "Bối cảnh bổ sung",
                },
            },
            "required": [],
        },
        function=tool_statistics_estimator,
    ),
]


def get_tools_for_openai() -> List[dict]:
    """Trả về danh sách tools dạng OpenAI function calling format."""
    return [tool.to_openai_function() for tool in TOOLS]


def get_tool_by_name(name: str) -> Optional[AgentTool]:
    """Tìm tool theo tên."""
    for tool in TOOLS:
        if tool.name == name:
            return tool
    return None


async def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Thực thi tool theo tên và tham số.

    Returns:
        Kết quả từ tool function.
    """
    tool = get_tool_by_name(name)
    if not tool:
        return {"error": f"Tool '{name}' không tồn tại."}

    try:
        result = await tool.function(**arguments)
        return result
    except Exception as e:
        log.error("Tool execution error [%s]: %s", name, e)
        return {"error": f"Lỗi khi thực thi tool '{name}': {str(e)}"}
