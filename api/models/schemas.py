"""
Pydantic Schemas for Academic RAG API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============ Enums ============

class PipelineType(str, Enum):
    BASELINE = "baseline"
    ADVANCED = "advanced"


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    ERROR = "error"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============ Query Schemas ============

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="Question to answer")
    pipeline_type: PipelineType = Field(
        default=PipelineType.ADVANCED,
        description="Pipeline type: 'baseline' or 'advanced'"
    )
    max_results: int = Field(default=5, ge=1, le=20)


class SourceDocument(BaseModel):
    """Source document in response."""
    id: str
    content: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    status: str = "success"
    answer: str
    confidence: float
    sources: List[SourceDocument]
    query_id: str
    pipeline_used: str
    response_time: Optional[float] = None


class BatchQueryRequest(BaseModel):
    """Batch query request."""
    questions: List[str]
    pipeline_type: PipelineType = PipelineType.ADVANCED


class BatchQueryResponse(BaseModel):
    """Batch query response."""
    status: str = "success"
    results: List[Dict[str, Any]]
    total: int


# ============ Document Schemas ============

class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    status: str = "success"
    document_id: str
    filename: str
    size_bytes: int
    message: str


class DocumentInfo(BaseModel):
    """Document information."""
    id: str
    filename: str
    original_filename: str
    size_bytes: int
    status: DocumentStatus
    chunk_count: int = 0
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class DocumentListResponse(BaseModel):
    """List of documents."""
    status: str = "success"
    documents: List[DocumentInfo]
    total: int


class DocumentDetailResponse(BaseModel):
    """Detailed document information."""
    status: str = "success"
    document: DocumentInfo
    chunks: Optional[List[Dict[str, Any]]] = None


# ============ Chunking Schemas ============

class ChunkingConfig(BaseModel):
    """Chunking configuration."""
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)


class ChunkingRequest(BaseModel):
    """Request to process documents."""
    document_ids: List[str] = Field(..., description="Document IDs to process")
    config: Optional[ChunkingConfig] = None
    auto_index: bool = Field(default=True, description="Auto-index after chunking")


class ChunkingJobResponse(BaseModel):
    """Response with job information."""
    status: str = "success"
    job_id: str
    message: str
    documents_queued: int


class ChunkingStatusResponse(BaseModel):
    """Chunking job status."""
    status: str = "success"
    job_id: str
    job_status: JobStatus
    progress: float = Field(description="Progress 0.0 to 1.0")
    documents_processed: int
    documents_total: int
    error_message: Optional[str] = None


class ChunkInfo(BaseModel):
    """Chunk information."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None


# ============ Knowledge Base Schemas ============

class KBStatsResponse(BaseModel):
    """Knowledge base statistics."""
    status: str = "success"
    total_documents: int
    total_chunks: int
    indexed_chunks: int
    vector_store_status: str
    bm25_index_status: str
    last_indexed: Optional[datetime] = None
    storage_size_mb: float


class ReindexRequest(BaseModel):
    """Reindex request."""
    rebuild_vectors: bool = True
    rebuild_bm25: bool = True
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific documents to reindex (None = all)"
    )


class ReindexResponse(BaseModel):
    """Reindex response."""
    status: str = "success"
    job_id: str
    message: str


class SearchChunksRequest(BaseModel):
    """Search chunks request."""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    search_type: str = Field(default="hybrid", description="vector, bm25, or hybrid")


class SearchChunksResponse(BaseModel):
    """Search chunks response."""
    status: str = "success"
    chunks: List[ChunkInfo]
    total: int


# ============ Health Schemas ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    vector_store: str
    bm25_index: str
    version: str
    timestamp: datetime


# ============ Error Schemas ============

class ErrorResponse(BaseModel):
    """Error response."""
    status: str = "error"
    message: str
    detail: Optional[str] = None
