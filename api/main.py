"""
Academic RAG API - Main Application.

FastAPI application for the Academic RAG system.
Provides endpoints for:
- Query/Chat
- Document Management
- Chunking
- Knowledge Base Management
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add academic-rag to path
ACADEMIC_RAG_PATH = Path(__file__).parent.parent.parent / "rag-model"
sys.path.insert(0, str(ACADEMIC_RAG_PATH))

# Import routers
from .routes.query import router as query_router
from .routes.health import router as health_router
from .routes.documents import router as documents_router
from .routes.chunking import router as chunking_router
from .routes.knowledge_base import router as kb_router

# Create FastAPI app
app = FastAPI(
    title="Academic RAG API",
    description="""
REST API for Academic RAG System.

## Features

- **Query API**: Question answering using RAG
- **Document API**: Upload and manage documents
- **Chunking API**: Process documents into searchable chunks
- **Knowledge Base API**: Manage indexes and search

## For Laravel Integration

```php
$response = Http::post('http://localhost:5001/api/query', [
    'question' => 'Apa syarat beasiswa prestasi?'
]);
```
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query_router)
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chunking_router)
app.include_router(kb_router)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Academic RAG API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "query": "/api/query",
            "documents": "/api/documents",
            "chunking": "/api/chunking",
            "knowledge_base": "/api/kb",
            "health": "/health"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Academic RAG API...")
    # Data directories are now managed by document_service (shared data/ folder)
    logger.info("Academic RAG API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Academic RAG API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
