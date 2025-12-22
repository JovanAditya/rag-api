"""
RAG Service for API.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import uuid

# Add rag-model to path (submodule directory)
RAG_MODEL_PATH = Path(__file__).parent.parent.parent / "rag-model"
sys.path.insert(0, str(RAG_MODEL_PATH))

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service wrapper for Academic RAG model.
    
    Provides a clean interface for the API to interact with the RAG model.
    """
    
    def __init__(self):
        """Initialize RAG service with both pipeline types."""
        self._baseline_rag = None
        self._advanced_rag = None
        self._current_pipeline = "advanced"
        
        self._init_models()
    
    def _init_models(self):
        """Initialize RAG models."""
        try:
            from rag_model import AcademicRAG
            from rag_model.core.config import RAGConfig, RetrievalConfig
            
            # Initialize baseline (vector-only)
            logger.info("Initializing baseline RAG model...")
            # Create config with specific retrieval type
            baseline_retrieval = RetrievalConfig(pipeline_type="baseline")
            baseline_config = RAGConfig(retrieval=baseline_retrieval)
            
            self._baseline_rag = AcademicRAG(
                config=baseline_config,
                research_mode=False,
                response_format="api"
            )
            
            # Initialize advanced (hybrid + reranking)
            logger.info("Initializing advanced RAG model...")
            # Create config with specific retrieval type
            advanced_retrieval = RetrievalConfig(pipeline_type="advanced")
            advanced_config = RAGConfig(retrieval=advanced_retrieval)
            
            self._advanced_rag = AcademicRAG(
                config=advanced_config,
                research_mode=False,
                response_format="api"
            )
            
            logger.info("RAG models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG models: {e}")
            raise
    
    def refresh(self):
        """
        Refresh/reload RAG models after reindexing.
        This allows using new index without API restart.
        """
        logger.info("Refreshing RAG models...")
        self._baseline_rag = None
        self._advanced_rag = None
        self._init_models()
        logger.info("RAG models refreshed successfully")
    
    def is_ready(self) -> bool:
        """Check if service is ready to handle requests."""
        return self._baseline_rag is not None or self._advanced_rag is not None
    
    def health_check(self) -> Dict[str, Any]:
        """Run health check on RAG models."""
        result = {
            "ready": self.is_ready(),
            "model_loaded": self.is_ready(),
            "baseline_available": self._baseline_rag is not None,
            "advanced_available": self._advanced_rag is not None
        }
        
        # Check detailed health from model
        try:
            # Try advanced first
            rag = self._advanced_rag or self._baseline_rag
            if rag:
                model_health = rag.health_check()
                # Extract from components sub-dict
                components = model_health.get("components", {})
                vs_status = components.get("vector_store", {})
                bm25_status = components.get("bm25_index", {})
                
                # Handle both string and dict status formats
                if isinstance(vs_status, dict):
                    result["vector_store"] = vs_status.get("status", "unknown")
                else:
                    result["vector_store"] = vs_status if vs_status else "unknown"
                    
                if isinstance(bm25_status, dict):
                    result["bm25_index"] = bm25_status.get("status", "unknown")
                else:
                    result["bm25_index"] = bm25_status if bm25_status else "unknown"
                
                # LLM status
                llm_status = components.get("llm", {})
                if isinstance(llm_status, dict):
                    result["llm"] = llm_status.get("status", "unknown")
                else:
                    result["llm"] = llm_status if llm_status else "unknown"
                
                # Log for debugging
                logger.debug(f"Health components: {list(components.keys())}")
                logger.debug(f"BM25 raw status: {bm25_status}")
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            result["error"] = str(e)
        
        return result
    
    def query(
        self,
        question: str,
        pipeline_type: str = "advanced",
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: The question to answer
            pipeline_type: 'baseline' or 'advanced'
            max_results: Maximum number of source documents
        
        Returns:
            Dictionary with answer, confidence, sources, and metadata
        """
        # Select appropriate model
        if pipeline_type == "baseline":
            rag = self._baseline_rag
        else:
            rag = self._advanced_rag
        
        if not rag:
            raise RuntimeError(f"RAG model for pipeline '{pipeline_type}' not available")
        
        # Generate query ID
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        
        try:
            # Execute query
            result = rag.query(question)
            
            # Format response
            return {
                "query_id": query_id,
                "answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "sources": result.get("sources", [])[:max_results],
                "metadata": {
                    "pipeline_type": pipeline_type,
                    "retrieval_time": result.get("metadata", {}).get("retrieval_time", 0),
                    "generation_time": result.get("metadata", {}).get("generation_time", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise


# Global service instance
rag_service = RAGService()
