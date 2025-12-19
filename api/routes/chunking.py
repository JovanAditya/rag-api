"""
Chunking Routes.

Endpoints for document processing and chunking.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from ..models.schemas import (
    ChunkingConfig,
    ChunkingRequest,
    ChunkingJobResponse,
    ChunkingStatusResponse,
    JobStatus,
    ErrorResponse,
)
from ..services.chunking_service import chunking_service

router = APIRouter(prefix="/api/chunking", tags=["Chunking"])


@router.post(
    "/process",
    response_model=ChunkingJobResponse,
    responses={400: {"model": ErrorResponse}}
)
async def process_documents(request: ChunkingRequest):
    """
    Process uploaded documents into chunks.
    
    - **document_ids**: List of document IDs to process
    - **config**: Optional chunking configuration
    - **auto_index**: Whether to auto-index after chunking (default: true)
    
    Returns job ID for tracking progress.
    """
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    # Get config
    chunk_size = None
    chunk_overlap = None
    if request.config:
        chunk_size = request.config.chunk_size
        chunk_overlap = request.config.chunk_overlap
    
    # Start processing
    job_id = chunking_service.process_documents(
        document_ids=request.document_ids,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        auto_index=request.auto_index
    )
    
    return ChunkingJobResponse(
        status="success",
        job_id=job_id,
        message="Processing started",
        documents_queued=len(request.document_ids)
    )


@router.get(
    "/status/{job_id}",
    response_model=ChunkingStatusResponse,
    responses={404: {"model": ErrorResponse}}
)
async def get_job_status(job_id: str):
    """
    Get chunking job status.
    
    - **job_id**: Job ID from process endpoint
    """
    job = chunking_service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ChunkingStatusResponse(
        status="success",
        job_id=job["id"],
        job_status=JobStatus(job["status"]),
        progress=job["progress"],
        documents_processed=job["documents_processed"],
        documents_total=job["documents_total"],
        error_message=job.get("error_message")
    )


@router.get("/config")
async def get_chunking_config():
    """
    Get current chunking configuration.
    """
    return {
        "status": "success",
        "config": {
            "chunk_size": chunking_service.default_chunk_size,
            "chunk_overlap": chunking_service.default_chunk_overlap
        }
    }


@router.put("/config")
async def update_chunking_config(config: ChunkingConfig):
    """
    Update default chunking configuration.
    
    - **chunk_size**: Default chunk size (100-5000)
    - **chunk_overlap**: Default overlap (0-1000)
    """
    chunking_service.update_config(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    return {
        "status": "success",
        "message": "Configuration updated",
        "config": {
            "chunk_size": chunking_service.default_chunk_size,
            "chunk_overlap": chunking_service.default_chunk_overlap
        }
    }


@router.post("/process-all")
async def process_all_documents(
    config: ChunkingConfig = None,
    auto_index: bool = True
):
    """
    Process all uploaded documents that haven't been processed.
    
    - **config**: Optional chunking configuration
    - **auto_index**: Whether to auto-index after chunking
    """
    from ..services.document_service import document_service
    
    # Get unprocessed documents
    docs = document_service.list_documents()
    unprocessed = [
        d["id"] for d in docs 
        if d.get("status") in ["uploaded", "error"]
    ]
    
    if not unprocessed:
        return {
            "status": "success",
            "message": "No documents to process",
            "documents_queued": 0
        }
    
    # Get config
    chunk_size = None
    chunk_overlap = None
    if config:
        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap
    
    # Start processing
    job_id = chunking_service.process_documents(
        document_ids=unprocessed,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        auto_index=auto_index
    )
    
    return {
        "status": "success",
        "job_id": job_id,
        "message": "Processing started",
        "documents_queued": len(unprocessed)
    }
