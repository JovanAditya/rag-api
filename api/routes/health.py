"""
Health Check Routes - Updated.
"""

from datetime import datetime
from fastapi import APIRouter

from ..models.schemas import HealthResponse
from ..services.rag_service import rag_service

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and component health.
    """
    try:
        health = rag_service.health_check()
        
        return HealthResponse(
            status="healthy" if health.get("ready", False) else "degraded",
            model_loaded=health.get("model_loaded", False),
            vector_store=health.get("vector_store", "unknown"),
            bm25_index=health.get("bm25_index", "unknown"),
            version="0.1.0",
            timestamp=datetime.now()
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            model_loaded=False,
            vector_store="error",
            bm25_index="error",
            version="0.1.0",
            timestamp=datetime.now()
        )


@router.get("/health/detailed")
async def detailed_health():
    """
    Detailed health check with all components.
    """
    from ..services.kb_service import kb_service
    from ..services.document_service import document_service
    
    try:
        # RAG health
        rag_health = rag_service.health_check()
        
        # KB stats
        kb_stats = kb_service.get_stats()
        
        # Document stats
        doc_stats = document_service.get_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "rag_model": {
                    "status": "healthy" if rag_health.get("ready") else "degraded",
                    "model_loaded": rag_health.get("model_loaded", False)
                },
                "vector_store": {
                    "status": kb_stats["vector_store_status"]
                },
                "bm25_index": {
                    "status": kb_stats["bm25_index_status"]
                },
                "document_storage": {
                    "status": "healthy",
                    "total_documents": doc_stats["total_documents"],
                    "storage_mb": doc_stats["total_size_mb"]
                },
                "knowledge_base": {
                    "status": "healthy",
                    "total_chunks": kb_stats["total_chunks"],
                    "storage_mb": kb_stats["storage_size_mb"]
                }
            },
            "version": "0.1.0"
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": "0.1.0"
        }


@router.get("/")
async def root():
    """Root endpoint - redirects to docs info."""
    return {
        "message": "Academic RAG API",
        "docs": "/docs",
        "health": "/health"
    }
