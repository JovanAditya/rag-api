"""
Knowledge Base Service for Academic RAG API.

Handles knowledge base management, statistics, and reindexing.
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Add academic-rag to path
ACADEMIC_RAG_PATH = Path(__file__).parent.parent.parent.parent / "academic-rag"
sys.path.insert(0, str(ACADEMIC_RAG_PATH))

from .document_service import document_service, PROCESSED_DIR, DATA_DIR, DOCUMENTS_DIR

logger = logging.getLogger(__name__)

# Reindex job storage
_reindex_jobs: Dict[str, Dict[str, Any]] = {}

# Stats cache (10 second TTL)
_stats_cache: Dict[str, Any] = {}
_stats_cache_time: float = 0
_STATS_CACHE_TTL: float = 10.0  # seconds


class KnowledgeBaseService:
    """Service for knowledge base management."""
    
    def __init__(self):
        """Initialize KB service."""
        self._rag = None
    
    def _get_rag(self):
        """Get or create RAG instance."""
        if self._rag is None:
            try:
                from rag_model import AcademicRAG
                self._rag = AcademicRAG()
            except Exception as e:
                logger.error(f"Failed to initialize RAG: {e}")
        return self._rag
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Statistics dictionary
        """
        global _stats_cache, _stats_cache_time
        
        import time
        now = time.time()
        
        # Return cached stats if still valid
        if _stats_cache and (now - _stats_cache_time) < _STATS_CACHE_TTL:
            return _stats_cache
        
        # Document stats
        doc_stats = document_service.get_stats()
        
        # Chunk stats - prefer chunks.json (manual), fallback to all_chunks.json
        chunks_file = PROCESSED_DIR / "chunks.json"
        if not chunks_file.exists():
            chunks_file = PROCESSED_DIR / "all_chunks.json"
        total_chunks = 0
        if chunks_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    total_chunks = len(chunks)
            except Exception as e:
                logger.error(f"Failed to load chunks: {e}")
        
        # Storage size
        storage_size = 0
        for dir_path in [PROCESSED_DIR, DOCUMENTS_DIR]:
            if dir_path.exists():
                for f in dir_path.glob("**/*"):
                    if f.is_file():
                        storage_size += f.stat().st_size
        
        # RAG status - check directly from disk for accurate status
        vector_status = "unknown"
        bm25_status = "unknown"
        last_indexed = None
        indexed_chunks = 0
        
        # Check ChromaDB directly - query actual collection count
        # Skip if auto-indexing is in progress to avoid blocking
        try:
            from .chunking_service import _auto_index_in_progress, _auto_index_lock
            
            # Check if indexing is in progress
            with _auto_index_lock:
                indexing_active = _auto_index_in_progress
            
            if indexing_active:
                # Skip ChromaDB check during indexing to avoid lock contention
                vector_status = "indexing"
                logger.debug("Skipping ChromaDB check - indexing in progress")
            else:
                chroma_dir = DATA_DIR / "chroma_db"
                if chroma_dir.exists():
                    import chromadb
                    client = chromadb.PersistentClient(path=str(chroma_dir))
                    collections = client.list_collections()
                    
                    if collections:
                        # Check if any collection has documents
                        total_docs_in_chroma = 0
                        for col in collections:
                            try:
                                total_docs_in_chroma += col.count()
                            except:
                                pass
                        
                        if total_docs_in_chroma > 0:
                            vector_status = "healthy"
                            indexed_chunks = total_docs_in_chroma
                        else:
                            vector_status = "empty"
                    else:
                        vector_status = "empty"
                    
                    del client
                else:
                    vector_status = "empty"
        except Exception as e:
            logger.debug(f"ChromaDB check: {e}")
            vector_status = "empty"
        
        # Check BM25 cache directly  
        try:
            cache_dir = DATA_DIR / "cache"
            bm25_file = cache_dir / "bm25_academic_docs.pkl.gz"
            if bm25_file.exists() and bm25_file.stat().st_size > 1000:  # > 1KB means has data
                bm25_status = "healthy"
            else:
                bm25_status = "empty"
        except Exception as e:
            logger.error(f"BM25 check failed: {e}")
            bm25_status = "error"
        
        
        result = {
            "total_documents": doc_stats["total_documents"],
            "total_chunks": total_chunks,
            "indexed_chunks": indexed_chunks,
            "vector_store_status": vector_status,
            "bm25_index_status": bm25_status,
            "last_indexed": last_indexed,
            "storage_size_mb": round(storage_size / (1024 * 1024), 2)
        }
        
        # Update cache
        _stats_cache = result
        _stats_cache_time = now
        
        return result
    
    def reindex(
        self,
        rebuild_vectors: bool = True,
        rebuild_bm25: bool = True,
        document_ids: List[str] = None
    ) -> str:
        """
        Trigger reindexing.
        
        Args:
            rebuild_vectors: Rebuild vector index
            rebuild_bm25: Rebuild BM25 index
            document_ids: Specific documents (None = all)
            
        Returns:
            Job ID
        """
        import threading
        import uuid
        from datetime import datetime
        
        job_id = f"reindex_{uuid.uuid4().hex[:12]}"
        
        # Initialize job tracking
        _reindex_jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "progress": 0.0,
            "chunks_indexed": 0,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "error_message": None
        }
        
        thread = threading.Thread(
            target=self._reindex_worker,
            args=(job_id, rebuild_vectors, rebuild_bm25, document_ids)
        )
        thread.start()
        
        return job_id
    
    def get_reindex_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get reindex job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status or None
        """
        return _reindex_jobs.get(job_id)
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of active reindex/indexing jobs.
        
        Returns:
            List of active jobs with pending or running status
        """
        active = []
        for job_id, job in _reindex_jobs.items():
            if job.get("status") in ["pending", "running"]:
                active.append(job)
        return active
    
    def _reindex_worker(
        self,
        job_id: str,
        rebuild_vectors: bool,
        rebuild_bm25: bool,
        document_ids: List[str]
    ):
        """Background worker for reindexing."""
        from datetime import datetime
        
        try:
            logger.info(f"Starting reindex job: {job_id}")
            _reindex_jobs[job_id]["status"] = "running"
            
            # Load chunks - prefer chunks.json (manual), fallback to all_chunks.json
            chunks_file = PROCESSED_DIR / "chunks.json"
            if not chunks_file.exists():
                chunks_file = PROCESSED_DIR / "all_chunks.json"
            if not chunks_file.exists():
                logger.warning("No chunks file found")
                _reindex_jobs[job_id]["status"] = "completed"
                _reindex_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                return
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Filter by document IDs if specified
            if document_ids:
                chunks = [c for c in chunks if c.get("document_id") in document_ids]
            
            if not chunks:
                logger.warning("No chunks to index")
                _reindex_jobs[job_id]["status"] = "completed"
                _reindex_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                return
            
            logger.info(f"Indexing {len(chunks)} chunks")
            _reindex_jobs[job_id]["progress"] = 0.2
            
            # Try to use academic-rag indexing
            try:
                from rag_model.indexing.unified_index_manager import UnifiedIndexManager
                from rag_model.core.config import IndexConfig, BM25Config
                
                # Get config objects and convert to dict format expected by UnifiedIndexManager
                index_cfg = IndexConfig()
                bm25_cfg = BM25Config()
                
                # CLEAR existing collection first to avoid duplicates
                try:
                    import chromadb
                    chroma_client = chromadb.PersistentClient(path=str(index_cfg.chroma_dir))
                    existing_collections = [c.name for c in chroma_client.list_collections()]
                    
                    if index_cfg.chroma_collection in existing_collections:
                        chroma_client.delete_collection(index_cfg.chroma_collection)
                        logger.info(f"Cleared existing collection: {index_cfg.chroma_collection}")
                    
                    del chroma_client
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Could not clear existing collection: {e}")
                
                _reindex_jobs[job_id]["progress"] = 0.3
                
                # UnifiedIndexManager expects dicts with specific keys
                vector_config = {
                    "collection_name": index_cfg.chroma_collection,
                    "persist_directory": index_cfg.chroma_dir,
                    "embedding_model": "indobenchmark/indobert-base-p2"
                }
                
                bm25_config = {
                    "k1": bm25_cfg.k1,
                    "b": bm25_cfg.b,
                    "ngram_range": (bm25_cfg.ngram_range_min, bm25_cfg.ngram_range_max)
                }
                
                manager = UnifiedIndexManager(
                    vector_config=vector_config,
                    bm25_config=bm25_config,
                    cache_dir=index_cfg.cache_dir
                )
                
                _reindex_jobs[job_id]["progress"] = 0.4
                
                # Prepare documents for indexing
                documents = [
                    {
                        "text": c.get("content") or c.get("text", ""),
                        "metadata": {
                            "chunk_id": c.get("id") or c.get("chunk_id", ""),
                            "document_id": c.get("document_id", ""),
                            "chunk_index": c.get("chunk_index", 0),
                            **c.get("metadata", {})
                        }
                    }
                    for c in chunks
                ]
                
                # Index
                if rebuild_vectors or rebuild_bm25:
                    manager.index_documents(documents)
                
                _reindex_jobs[job_id]["chunks_indexed"] = len(documents)
                _reindex_jobs[job_id]["progress"] = 0.9
                
                logger.info(f"Reindex completed: {len(documents)} documents")
                
                # Refresh the main RAG instance to see the new indexes
                rag = self._get_rag()
                if rag:
                    try:
                        rag.refresh_indexes()
                        logger.info("RAG pipeline indexes refreshed successfully")
                    except Exception as e:
                        logger.warning(f"Failed to refresh RAG indexes: {e}")
                
                # Also refresh the global rag_service used by query endpoint
                try:
                    from .rag_service import rag_service
                    rag_service.refresh()
                    logger.info("Global RAG service refreshed successfully")
                except Exception as e:
                    logger.warning(f"Failed to refresh global RAG service: {e}")
                
                _reindex_jobs[job_id]["progress"] = 1.0
                _reindex_jobs[job_id]["status"] = "completed"
                _reindex_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"RAG indexing failed: {e}")
                _reindex_jobs[job_id]["status"] = "failed"
                _reindex_jobs[job_id]["error_message"] = str(e)
                _reindex_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Reindex job failed: {e}")
            _reindex_jobs[job_id]["status"] = "failed"
            _reindex_jobs[job_id]["error_message"] = str(e)
            _reindex_jobs[job_id]["completed_at"] = datetime.now().isoformat()
    
    def search_chunks(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Search chunks in knowledge base.
        
        Args:
            query: Search query
            limit: Max results
            search_type: vector, bm25, or hybrid
            
        Returns:
            List of matching chunks
        """
        results = []
        
        # Try RAG search first
        rag = self._get_rag()
        if rag:
            try:
                # Use appropriate pipeline
                pipeline_type = "baseline" if search_type == "vector" else "advanced"
                result = rag.query(query, pipeline_type=pipeline_type)
                
                sources = result.get("sources", [])
                for i, source in enumerate(sources[:limit]):
                    results.append({
                        "id": source.get("id", f"result_{i}"),
                        "document_id": source.get("metadata", {}).get("document_id", "unknown"),
                        "content": source.get("text", source.get("content", "")),
                        "chunk_index": source.get("metadata", {}).get("chunk_index", i),
                        "score": source.get("score", 0),
                        "metadata": source.get("metadata", {})
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"RAG search failed: {e}")
        
        # Fallback: simple text search in chunks file
        chunks_file = PROCESSED_DIR / "chunks.json"
        if not chunks_file.exists():
            chunks_file = PROCESSED_DIR / "all_chunks.json"
        if chunks_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                query_lower = query.lower()
                for chunk in chunks:
                    content = chunk.get("content", "")
                    if query_lower in content.lower():
                        results.append({
                            "id": chunk.get("id"),
                            "document_id": chunk.get("document_id"),
                            "content": content,
                            "chunk_index": chunk.get("chunk_index", 0),
                            "metadata": chunk.get("metadata", {})
                        })
                        if len(results) >= limit:
                            break
                            
            except Exception as e:
                logger.error(f"Fallback search failed: {e}")
        
        return results
    
    def clear_all(self) -> bool:
        """
        Clear all knowledge base data including documents, chunks, and indexes.
        
        Returns:
            True if successful
        """
        try:
            import shutil
            import gc
            import time
            
            logger.info("Starting knowledge base clear...")
            
            # Step 1: Reset all Python references to RAG
            self._rag = None
            
            from .rag_service import rag_service
            rag_service._advanced_rag = None
            rag_service._baseline_rag = None
            
            # Force garbage collection
            gc.collect()
            time.sleep(0.5)
            
            # Step 2: Clear ChromaDB collections (don't try to delete files - Windows lock issue)
            chroma_dir = DATA_DIR / "chroma_db"
            if chroma_dir.exists():
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=str(chroma_dir))
                    
                    # Delete all collections (data, not files)
                    for col in client.list_collections():
                        try:
                            client.delete_collection(col.name)
                            logger.info(f"Deleted collection: {col.name}")
                        except Exception as e:
                            logger.warning(f"Failed to delete collection {col.name}: {e}")
                    
                    del client
                    gc.collect()
                    logger.info("ChromaDB collections cleared")
                except Exception as e:
                    logger.warning(f"ChromaDB clear failed: {e}")
            
            # Step 3: Clear documents folder
            if DOCUMENTS_DIR.exists():
                for item in DOCUMENTS_DIR.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                logger.info("Documents folder contents cleared")
            
            # Step 4: Clear processed folder (chunks, metadata, cache)
            if PROCESSED_DIR.exists():
                for item in PROCESSED_DIR.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                logger.info("Processed folder contents cleared")
            
            # Step 5: Clear BM25 cache
            cache_dir = DATA_DIR / "cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
                logger.info("BM25 cache cleared")
            
            # Step 6: Reinitialize document service
            document_service._metadata = {"documents": {}}
            document_service._save_metadata()
            
            # Invalidate stats cache
            global _stats_cache, _stats_cache_time
            _stats_cache = {}
            _stats_cache_time = 0
            
            logger.info("Knowledge base fully cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear KB: {e}")
            return False
    
    def clear_indexes(self) -> bool:
        """
        Clear only the indexes (ChromaDB and BM25) without affecting documents or chunks files.
        Use this before reindexing to avoid duplicates.
        
        Returns:
            True if successful
        """
        try:
            import shutil
            
            # Clear ChromaDB using the collection API (avoids file lock issues)
            rag = self._get_rag()
            if rag:
                try:
                    # Try to delete via unified_index_manager or vector_store
                    if hasattr(rag, '_unified_index_manager') and rag._unified_index_manager:
                        if hasattr(rag._unified_index_manager, 'vector_store') and rag._unified_index_manager.vector_store:
                            rag._unified_index_manager.vector_store.delete_collection()
                            logger.info("ChromaDB collection deleted via API")
                    elif hasattr(rag, '_vector_store') and rag._vector_store:
                        rag._vector_store.delete_collection()
                        logger.info("ChromaDB collection deleted via baseline API")
                except Exception as e:
                    logger.warning(f"Could not delete collection via API: {e}")
                    # Fallback: try direct deletion after closing
                    try:
                        import chromadb
                        chroma_dir = DATA_DIR / "chroma_db"
                        if chroma_dir.exists():
                            # Create new client to force release
                            client = chromadb.PersistentClient(path=str(chroma_dir))
                            try:
                                client.delete_collection("academic_docs")
                            except:
                                pass
                            del client
                            shutil.rmtree(chroma_dir, ignore_errors=True)
                            logger.info("ChromaDB cleared via direct client")
                    except Exception as e2:
                        logger.warning(f"Fallback ChromaDB clear failed: {e2}")
            else:
                # No RAG loaded, try direct file deletion
                chroma_dir = DATA_DIR / "chroma_db"
                if chroma_dir.exists():
                    shutil.rmtree(chroma_dir, ignore_errors=True)
                    logger.info("ChromaDB directory removed")
            
            # Clear BM25 cache (both locations for compatibility)
            for cache_path in [DATA_DIR / "cache", PROCESSED_DIR / "cache"]:
                if cache_path.exists():
                    shutil.rmtree(cache_path, ignore_errors=True)
                    logger.info(f"Cleared cache: {cache_path}")
            
            # Reset and reinitialize RAG service
            from .rag_service import rag_service
            rag_service._advanced_rag = None
            rag_service._baseline_rag = None
            rag_service._init_models()  # Reinitialize with fresh indexes
            
            logger.info("Indexes cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear indexes: {e}")
            return False
    
    def clear_chunks(self) -> bool:
        """
        Clear only chunks file without affecting documents or indexes.
        Useful for re-chunking documents with different settings.
        
        Returns:
            True if successful
        """
        try:
            # Delete chunks.json
            chunks_file = PROCESSED_DIR / "chunks.json"
            if chunks_file.exists():
                chunks_file.unlink()
                logger.info("Cleared chunks.json")
            
            # Also clear all_chunks.json if exists
            all_chunks_file = PROCESSED_DIR / "all_chunks.json"
            if all_chunks_file.exists():
                all_chunks_file.unlink()
                logger.info("Cleared all_chunks.json")
            
            # Reset document status to 'uploaded' so they can be re-processed
            for doc_id, doc_info in document_service._metadata.get("documents", {}).items():
                doc_info["status"] = "uploaded"
                doc_info["chunk_count"] = 0
                doc_info["processed_at"] = None
            document_service._save_metadata()
            
            logger.info("Chunks cleared, document status reset to 'uploaded'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear chunks: {e}")
            return False


# Singleton instance
kb_service = KnowledgeBaseService()
