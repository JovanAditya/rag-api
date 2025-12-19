"""
Document Management Routes.

Endpoints for uploading, viewing, and deleting documents.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List
from datetime import datetime

from ..models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentInfo,
    DocumentStatus,
    ErrorResponse,
)
from ..services.document_service import document_service

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOC, DOCX, TXT).
    
    - **file**: Document file to upload
    
    Returns document ID for tracking.
    """
    try:
        # Read file content
        content = await file.read()
        
        # Upload
        result = document_service.upload_document(content, file.filename)
        
        return DocumentUploadResponse(
            status="success",
            document_id=result["document_id"],
            filename=result["filename"],
            size_bytes=result["size_bytes"],
            message=result["message"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post(
    "/upload/batch",
    response_model=List[DocumentUploadResponse],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def upload_documents_batch(files: List[UploadFile] = File(...)):
    """
    Upload multiple documents at once.
    
    - **files**: List of document files to upload
    
    Returns list of document IDs.
    """
    results = []
    errors = []
    
    for file in files:
        try:
            content = await file.read()
            result = document_service.upload_document(content, file.filename)
            results.append(DocumentUploadResponse(
                status="success",
                document_id=result["document_id"],
                filename=result["filename"],
                size_bytes=result["size_bytes"],
                message=result["message"]
            ))
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
    
    if errors and not results:
        raise HTTPException(status_code=400, detail={"errors": errors})
    
    return results


@router.get(
    "",
    response_model=DocumentListResponse
)
async def list_documents(
    status: DocumentStatus = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    List all documents.
    
    - **status**: Filter by document status (optional)
    - **limit**: Maximum number of results
    - **offset**: Pagination offset
    """
    docs = document_service.list_documents()
    
    # Filter by status if specified
    if status:
        docs = [d for d in docs if d.get("status") == status.value]
    
    # Pagination
    total = len(docs)
    docs = docs[offset:offset + limit]
    
    # Convert to response model
    documents = []
    for d in docs:
        documents.append(DocumentInfo(
            id=d["id"],
            filename=d["filename"],
            original_filename=d["original_filename"],
            size_bytes=d["size_bytes"],
            status=DocumentStatus(d.get("status", "uploaded")),
            chunk_count=d.get("chunk_count", 0),
            uploaded_at=datetime.fromisoformat(d["uploaded_at"]),
            processed_at=datetime.fromisoformat(d["processed_at"]) if d.get("processed_at") else None,
            error_message=d.get("error_message")
        ))
    
    return DocumentListResponse(
        status="success",
        documents=documents,
        total=total
    )


@router.post(
    "/refresh",
    responses={500: {"model": ErrorResponse}}
)
async def refresh_documents():
    """
    Force refresh document metadata by scanning the documents folder.
    
    Useful after external file uploads (e.g., from Laravel).
    """
    try:
        count = document_service.refresh_metadata()
        return {
            "status": "success",
            "message": f"Metadata refreshed, {count} documents detected",
            "documents_count": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    responses={404: {"model": ErrorResponse}}
)
async def get_document(document_id: str, include_chunks: bool = False):
    """
    Get document details.
    
    - **document_id**: Document ID
    - **include_chunks**: Include chunk data (default: false)
    """
    doc = document_service.get_document(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = DocumentInfo(
        id=doc["id"],
        filename=doc["filename"],
        original_filename=doc["original_filename"],
        size_bytes=doc["size_bytes"],
        status=DocumentStatus(doc.get("status", "uploaded")),
        chunk_count=doc.get("chunk_count", 0),
        uploaded_at=datetime.fromisoformat(doc["uploaded_at"]),
        processed_at=datetime.fromisoformat(doc["processed_at"]) if doc.get("processed_at") else None,
        error_message=doc.get("error_message")
    )
    
    chunks = None
    if include_chunks:
        chunks = document_service.get_document_chunks(document_id)
    
    return DocumentDetailResponse(
        status="success",
        document=document,
        chunks=chunks
    )


@router.delete(
    "/{document_id}",
    responses={404: {"model": ErrorResponse}}
)
async def delete_document(document_id: str):
    """
    Delete a document.
    
    - **document_id**: Document ID to delete
    """
    success = document_service.delete_document(document_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "status": "success",
        "message": f"Document {document_id} deleted"
    }


@router.get(
    "/{document_id}/chunks"
)
async def get_document_chunks(document_id: str):
    """
    Get chunks for a document.
    
    - **document_id**: Document ID
    """
    doc = document_service.get_document(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = document_service.get_document_chunks(document_id)
    
    if chunks is None:
        return {
            "status": "success",
            "document_id": document_id,
            "chunks": [],
            "message": "Document not yet processed"
        }
    
    return {
        "status": "success",
        "document_id": document_id,
        "chunks": chunks,
        "total": len(chunks)
    }
