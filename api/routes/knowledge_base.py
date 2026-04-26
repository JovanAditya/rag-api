"""
Knowledge Base Routes.

Endpoints for knowledge base management.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime

from ..models.schemas import (
    KBStatsResponse,
    ReindexRequest,
    ReindexResponse,
    SearchChunksRequest,
    SearchChunksResponse,
    ChunkInfo,
    ErrorResponse,
)
from ..services.kb_service import kb_service

router = APIRouter(prefix="/v1/kb", tags=["Knowledge Base"])


@router.get(
    "/stats",
    response_model=KBStatsResponse
)
async def get_kb_stats():
    """
    Get knowledge base statistics.
    
    Returns counts, index status, and storage usage.
    """
    stats = kb_service.get_stats()
    
    return KBStatsResponse(
        status="success",
        total_documents=stats["total_documents"],
        total_chunks=stats["total_chunks"],
        indexed_chunks=stats["indexed_chunks"],
        vector_store_status=stats["vector_store_status"],
        bm25_index_status=stats["bm25_index_status"],
        last_indexed=stats.get("last_indexed"),
        storage_size_mb=stats["storage_size_mb"]
    )


@router.post(
    "/reindex",
    response_model=ReindexResponse
)
async def reindex_kb(request: ReindexRequest = None):
    """
    Trigger knowledge base reindexing.
    
    - **rebuild_vectors**: Rebuild vector index (default: true)
    - **rebuild_bm25**: Rebuild BM25 index (default: true)
    - **document_ids**: Specific documents to reindex (optional, None = all)
    """
    if request is None:
        request = ReindexRequest()
    
    job_id = kb_service.reindex(
        rebuild_vectors=request.rebuild_vectors,
        rebuild_bm25=request.rebuild_bm25,
        document_ids=request.document_ids
    )
    
    return ReindexResponse(
        status="success",
        job_id=job_id,
        message="Reindexing started"
    )


@router.get(
    "/reindex/status/{job_id}",
    responses={404: {"model": ErrorResponse}}
)
async def get_reindex_status(job_id: str):
    """
    Get reindex job status.
    
    - **job_id**: Job ID from reindex endpoint
    
    Returns job status, progress, and completion info.
    """
    job = kb_service.get_reindex_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "status": "success",
        "job": job
    }


@router.get("/jobs/active")
async def get_active_jobs():
    """
    Get currently active reindex/chunking jobs.
    
    Use this to determine if polling should continue.
    Returns has_active_jobs=true if any job is pending or running.
    """
    from ..services.chunking_service import chunking_service
    
    active_reindex = kb_service.get_active_jobs()
    active_chunking = chunking_service.get_active_jobs()
    
    has_active = len(active_reindex) > 0 or len(active_chunking) > 0
    
    return {
        "status": "success",
        "has_active_jobs": has_active,
        "reindex_jobs": active_reindex,
        "chunking_jobs": active_chunking,
        "total_active": len(active_reindex) + len(active_chunking)
    }

@router.get(
    "/search",
    response_model=SearchChunksResponse
)
async def search_chunks(
    query: str,
    limit: int = 10,
    search_type: str = "hybrid"
):
    """
    Search chunks in knowledge base.
    
    - **query**: Search query
    - **limit**: Maximum results (default: 10)
    - **search_type**: vector, bm25, or hybrid (default: hybrid)
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    results = kb_service.search_chunks(
        query=query,
        limit=limit,
        search_type=search_type
    )
    
    chunks = [
        ChunkInfo(
            id=r["id"],
            document_id=r["document_id"],
            content=r["content"],
            chunk_index=r["chunk_index"],
            metadata=r.get("metadata")
        )
        for r in results
    ]
    
    return SearchChunksResponse(
        status="success",
        chunks=chunks,
        total=len(chunks)
    )


@router.post("/search", response_model=SearchChunksResponse)
async def search_chunks_post(request: SearchChunksRequest):
    """
    Search chunks in knowledge base (POST version).
    
    - **query**: Search query
    - **limit**: Maximum results
    - **search_type**: vector, bm25, or hybrid
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    results = kb_service.search_chunks(
        query=request.query,
        limit=request.limit,
        search_type=request.search_type
    )
    
    chunks = [
        ChunkInfo(
            id=r["id"],
            document_id=r["document_id"],
            content=r["content"],
            chunk_index=r["chunk_index"],
            metadata=r.get("metadata")
        )
        for r in results
    ]
    
    return SearchChunksResponse(
        status="success",
        chunks=chunks,
        total=len(chunks)
    )


@router.delete(
    "/clear",
    responses={400: {"model": ErrorResponse}}
)
async def clear_kb(confirm: bool = False):
    """
    Clear all knowledge base data.
    
    **WARNING**: This will delete all documents, chunks, and indexes!
    
    - **confirm**: Must be true to proceed
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to clear knowledge base"
        )
    
    success = kb_service.clear_all()
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to clear knowledge base"
        )
    
    return {
        "status": "success",
        "message": "Knowledge base fully cleared (documents, chunks, and indexes)"
    }


@router.delete(
    "/clear-indexes",
    responses={400: {"model": ErrorResponse}}
)
async def clear_indexes(confirm: bool = False):
    """
    Clear only indexes (ChromaDB and BM25) without affecting document files or chunks.
    
    Use this before reindexing to avoid duplicate entries.
    
    - **confirm**: Must be true to proceed
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to clear indexes"
        )
    
    success = kb_service.clear_indexes()
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to clear indexes"
        )
    
    return {
        "status": "success",
        "message": "Indexes cleared (ChromaDB and BM25). Documents and chunks are preserved."
    }


@router.delete(
    "/clear-chunks",
    responses={400: {"model": ErrorResponse}}
)
async def clear_chunks(confirm: bool = False):
    """
    Clear only chunks file without affecting documents or indexes.
    
    Useful for re-chunking documents with different settings.
    Documents are reset to 'uploaded' status so they can be re-processed.
    
    - **confirm**: Must be true to proceed
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to clear chunks"
        )
    
    success = kb_service.clear_chunks()
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to clear chunks"
        )
    
    return {
        "status": "success",
        "message": "Chunks cleared. Documents reset to 'uploaded' status for re-processing."
    }

@router.get("/health")
async def kb_health():
    """
    Quick health check for knowledge base.
    """
    stats = kb_service.get_stats()
    
    healthy = (
        stats["vector_store_status"] in ["healthy", "unknown"] and
        stats["bm25_index_status"] in ["healthy", "unknown"]
    )
    
    return {
        "status": "healthy" if healthy else "degraded",
        "vector_store": stats["vector_store_status"],
        "bm25_index": stats["bm25_index_status"],
        "total_documents": stats["total_documents"],
        "total_chunks": stats["total_chunks"],
        "timestamp": datetime.now().isoformat()
    }
