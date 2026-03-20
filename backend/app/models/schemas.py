"""
Pydantic schemas cho request / response.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Upload ──────────────────────────────────────────────────
class UploadResponse(BaseModel):
    dataset_id: str
    file_name: str
    total_chunks: int
    total_chars: int
    message: str


class FolderUploadItem(BaseModel):
    """Kết quả upload cho từng file trong folder."""
    file_name: str
    dataset_id: Optional[str] = None
    total_chunks: int = 0
    total_chars: int = 0
    success: bool
    error: Optional[str] = None


class FolderUploadResponse(BaseModel):
    """Kết quả upload cả folder."""
    total_files: int
    success_count: int
    fail_count: int
    results: List[FolderUploadItem]
    message: str


# ── Chat ────────────────────────────────────────────────────
class MetadataFilter(BaseModel):
    """Bộ lọc metadata cho retrieval pipeline."""
    field: Optional[str] = None
    status: Optional[str] = None
    document_type: Optional[str] = None
    government_level: Optional[str] = None
    year: Optional[int] = None

    def to_dict(self) -> dict:
        """Chuyển thành dict, bỏ qua các field None."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class QueryAnalysis(BaseModel):
    """Kết quả phân tích câu hỏi từ module Query Understanding."""
    intent: str
    filters: dict = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    sort: str = "relevance"


class ChatRequest(BaseModel):
    question: str
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    filters: Optional[MetadataFilter] = None
    conversation_id: Optional[str] = None


class SourceChunk(BaseModel):
    content: str
    score: float
    dataset_id: Optional[str] = None
    metadata: Optional[dict] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    query_analysis: Optional[QueryAnalysis] = None
    confidence: Optional[float] = None
    intent: Optional[str] = None


# ── Dataset ─────────────────────────────────────────────────
class DatasetInfo(BaseModel):
    dataset_id: str
    file_name: str
    total_chunks: int
    created_at: str


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]


# ── GPU ─────────────────────────────────────────────────────
class GpuStatus(BaseModel):
    available: bool
    device_name: Optional[str] = None
    vram_total_mb: Optional[float] = None
    vram_used_mb: Optional[float] = None
    cuda_version: Optional[str] = None


# ── Copilot ─────────────────────────────────────────────────
class IntentResult(BaseModel):
    """Kết quả nhận diện intent."""
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


class CopilotRequest(BaseModel):
    """Request cho Copilot chat endpoint."""
    question: str
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    dataset_id: Optional[str] = None


class CopilotResponse(BaseModel):
    """Response từ Copilot chat endpoint."""
    answer: str
    intent: IntentResult
    sources: List[SourceChunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Procedure ──────────────────────────────────────────────
class ProcedureStep(BaseModel):
    """Một bước trong thủ tục hành chính."""
    step_number: int
    description: str
    note: Optional[str] = None


class ProcedureInfo(BaseModel):
    """Thông tin thủ tục hành chính."""
    procedure_id: str
    procedure_name: str
    description: str
    steps: List[ProcedureStep]
    required_documents: List[str]
    processing_time: Optional[str] = None
    fee: Optional[str] = None
    authority: Optional[str] = None


class ProcedureRequest(BaseModel):
    """Request tra cứu thủ tục."""
    procedure_name: str


class ProcedureResponse(BaseModel):
    """Response thủ tục hành chính."""
    procedure: Optional[ProcedureInfo] = None
    message: str
    available_procedures: Optional[List[str]] = None


# ── Document Check ──────────────────────────────────────────
class DocumentCheckRequest(BaseModel):
    """Request kiểm tra hồ sơ."""
    procedure_name: str
    submitted_documents: List[str]


class DocumentCheckResponse(BaseModel):
    """Response kiểm tra hồ sơ."""
    procedure_name: str
    required_documents: List[str]
    submitted_documents: List[str]
    missing_documents: List[str]
    is_complete: bool
    message: str


# ── Document Summarize ──────────────────────────────────────
class SummarizeRequest(BaseModel):
    """Request tóm tắt văn bản."""
    query: str
    dataset_id: Optional[str] = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class SummarizeResponse(BaseModel):
    """Response tóm tắt văn bản."""
    summary: str
    sources: List[SourceChunk] = Field(default_factory=list)


# ── Document Compare ────────────────────────────────────────
class CompareRequest(BaseModel):
    """Request so sánh văn bản."""
    query: str
    dataset_id_1: Optional[str] = None
    dataset_id_2: Optional[str] = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class CompareResponse(BaseModel):
    """Response so sánh văn bản."""
    comparison: str
    sources_doc1: List[SourceChunk] = Field(default_factory=list)
    sources_doc2: List[SourceChunk] = Field(default_factory=list)


# ── Report ──────────────────────────────────────────────────
class ReportRequest(BaseModel):
    """Request tạo báo cáo."""
    request: str
    dataset_id: Optional[str] = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class ReportResponse(BaseModel):
    """Response báo cáo."""
    report: str
    sources: List[SourceChunk] = Field(default_factory=list)


# ── Intent API ──────────────────────────────────────────────
class IntentRequest(BaseModel):
    """Request cho intent detection."""
    question: str


class IntentResponse(BaseModel):
    """Response intent detection."""
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


# ── Search API ──────────────────────────────────────────────
class SearchRequest(BaseModel):
    """Request tìm kiếm vector database."""
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[MetadataFilter] = None


class SearchResponse(BaseModel):
    """Response tìm kiếm."""
    results: List[SourceChunk]
    total: int


# ── Tool API ────────────────────────────────────────────────
class ToolRequest(BaseModel):
    """Request cho tool execution."""
    content: str
    options: Optional[Dict[str, Any]] = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class ToolResponse(BaseModel):
    """Response từ tool."""
    tool: str
    result: str
    sources: List[SourceChunk] = Field(default_factory=list)


# ── Conversation API ────────────────────────────────────────
class ConversationCreate(BaseModel):
    """Request tạo conversation mới."""
    title: Optional[str] = None


class ConversationMessage(BaseModel):
    """Một message trong conversation."""
    role: str
    content: str
    timestamp: str


class ConversationInfo(BaseModel):
    """Thông tin tóm tắt conversation."""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationDetail(BaseModel):
    """Chi tiết conversation với messages."""
    id: str
    title: str
    messages: List[ConversationMessage]
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    """Danh sách conversations."""
    conversations: List[ConversationInfo]
