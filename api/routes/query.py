"""
Query Routes - Updated with batch support.
"""

import uuid
import time
from fastapi import APIRouter, HTTPException
from typing import List

from ..models.schemas import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    BatchQueryRequest,
    BatchQueryResponse,
    PipelineType,
    ErrorResponse,
)
from ..services.rag_service import rag_service

router = APIRouter(prefix="/api", tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}}
)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    - **question**: Question to answer
    - **pipeline_type**: 'baseline' or 'advanced' (default: advanced)
    - **max_results**: Maximum source documents (1-20)
    
    Returns answer with sources and confidence score.
    """
    start_time = time.time()
    
    try:
        result = rag_service.query(
            question=request.question,
            pipeline_type=request.pipeline_type.value,
            max_results=request.max_results
        )
        
        # Convert sources
        sources = []
        for source in result.get("sources", []):
            sources.append(SourceDocument(
                id=source.get("id", str(uuid.uuid4())),
                content=source.get("text", source.get("content", "")),
                score=source.get("score"),
                metadata=source.get("metadata")
            ))
        
        return QueryResponse(
            status="success",
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            sources=sources,
            query_id=result.get("query_id", f"q_{uuid.uuid4().hex[:12]}"),
            pipeline_used=request.pipeline_type.value,
            response_time=time.time() - start_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/query/batch",
    response_model=BatchQueryResponse,
    responses={500: {"model": ErrorResponse}}
)
async def batch_query(request: BatchQueryRequest):
    """
    Batch query multiple questions.
    
    - **questions**: List of questions
    - **pipeline_type**: Pipeline to use for all queries
    
    Returns list of results.
    """
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided")
    
    if len(request.questions) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 questions per batch")
    
    results = []
    
    for question in request.questions:
        try:
            result = rag_service.query(
                question=question,
                pipeline_type=request.pipeline_type.value
            )
            results.append({
                "question": question,
                "answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "question": question,
                "answer": "",
                "confidence": 0.0,
                "status": "error",
                "error": str(e)
            })
    
    return BatchQueryResponse(
        status="success",
        results=results,
        total=len(results)
    )


@router.post("/chat")
async def chat(question: str, pipeline_type: str = "advanced"):
    """
    Simple chat endpoint (convenience wrapper).
    
    - **question**: Question to answer
    - **pipeline_type**: Pipeline type (default: advanced)
    """
    try:
        result = rag_service.query(
            question=question,
            pipeline_type=pipeline_type
        )
        
        return {
            "status": "success",
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
