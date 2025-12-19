"""
Chunking Service for Academic RAG API.

Handles document processing and chunking.
"""

import os
import sys
import uuid
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Add academic-rag to path
ACADEMIC_RAG_PATH = Path(__file__).parent.parent.parent.parent / "academic-rag"
sys.path.insert(0, str(ACADEMIC_RAG_PATH))

from .document_service import document_service, PROCESSED_DIR

logger = logging.getLogger(__name__)

# Job storage
_jobs: Dict[str, Dict[str, Any]] = {}

# Lock to prevent concurrent auto-indexing
_auto_index_lock = threading.Lock()
_auto_index_in_progress = False
_pending_reindex = False  # Flag to trigger reindex after current one completes


class ChunkingService:
    """Service for document chunking."""
    
    def __init__(self):
        """Initialize chunking service."""
        self.default_chunk_size = 1000
        self.default_chunk_overlap = 200
    
    def process_documents(
        self,
        document_ids: List[str],
        chunk_size: int = None,
        chunk_overlap: int = None,
        auto_index: bool = True
    ) -> str:
        """
        Process documents into chunks (async).
        
        Args:
            document_ids: List of document IDs
            chunk_size: Chunk size in characters
            chunk_overlap: Overlap between chunks
            auto_index: Whether to auto-index after chunking
            
        Returns:
            Job ID for tracking
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        # Create job
        _jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "progress": 0.0,
            "documents_total": len(document_ids),
            "documents_processed": 0,
            "document_ids": document_ids,
            "created_at": datetime.now().isoformat(),
            "error_message": None
        }
        
        # Process in background
        thread = threading.Thread(
            target=self._process_worker,
            args=(
                job_id,
                document_ids,
                chunk_size or self.default_chunk_size,
                chunk_overlap or self.default_chunk_overlap,
                auto_index
            )
        )
        thread.start()
        
        return job_id
    
    def _process_worker(
        self,
        job_id: str,
        document_ids: List[str],
        chunk_size: int,
        chunk_overlap: int,
        auto_index: bool
    ):
        """Background worker for processing documents."""
        try:
            _jobs[job_id]["status"] = "running"
            
            # Collect all chunks in memory
            all_processed_chunks = []
            
            for i, doc_id in enumerate(document_ids):
                try:
                    # Update progress
                    _jobs[job_id]["progress"] = i / len(document_ids)
                    
                    # Get document
                    doc = document_service.get_document(doc_id)
                    if not doc:
                        logger.warning(f"Document not found: {doc_id}")
                        continue
                    
                    # Update status
                    document_service.update_document_status(doc_id, "processing")
                    
                    # Get file path
                    file_path = document_service.get_file_path(doc_id)
                    if not file_path or not file_path.exists():
                        document_service.update_document_status(
                            doc_id, "error", error_message="File not found"
                        )
                        continue
                    
                    # Extract text
                    text = self._extract_text(file_path)
                    if not text:
                        document_service.update_document_status(
                            doc_id, "error", error_message="Failed to extract text"
                        )
                        continue
                    
                    # Create chunks (pass document filename for metadata)
                    original_filename = doc.get("original_filename") or doc.get("filename") or file_path.name
                    chunks = self._create_chunks(
                        doc_id, text, chunk_size, chunk_overlap, original_filename
                    )
                    
                    # Collect chunks in memory (no per-document file)
                    all_processed_chunks.extend(chunks)
                    
                    # Update status
                    document_service.update_document_status(
                        doc_id, "indexed", chunk_count=len(chunks)
                    )
                    
                    _jobs[job_id]["documents_processed"] = i + 1
                    
                    logger.info(f"Processed document {doc_id}: {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing {doc_id}: {e}")
                    document_service.update_document_status(
                        doc_id, "error", error_message=str(e)
                    )
            
            # Auto-index if requested (pass collected chunks directly)
            if auto_index and all_processed_chunks:
                self._rebuild_index(document_ids, all_processed_chunks)
            
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 1.0
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error_message"] = str(e)
    
    def _extract_text(self, file_path: Path) -> Optional[str]:
        """
        Extract text from document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Extracted text
        """
        ext = file_path.suffix.lower()
        
        try:
            if ext == ".txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif ext == ".pdf":
                try:
                    import pdfplumber
                    text_parts = []
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)
                    return "\n\n".join(text_parts)
                except ImportError:
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(file_path)
                        text_parts = []
                        for page in reader.pages:
                            text_parts.append(page.extract_text())
                        return "\n\n".join(text_parts)
                    except ImportError:
                        logger.error("PDF libraries not available")
                        return None
            
            elif ext in [".doc", ".docx"]:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text_parts = []
                    for para in doc.paragraphs:
                        if para.text.strip():
                            text_parts.append(para.text)
                    return "\n\n".join(text_parts)
                except ImportError:
                    logger.error("python-docx not available")
                    return None
            
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return None
                
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return None
    
    def _create_chunks(
        self,
        doc_id: str,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        original_filename: str = None
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from text.
        
        Args:
            doc_id: Document ID
            text: Full text
            chunk_size: Chunk size
            chunk_overlap: Overlap size
            original_filename: Original document filename for metadata
            
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk_text.strip():
                chunks.append({
                    "id": f"{doc_id}_chunk_{chunk_index}",
                    "document_id": doc_id,
                    "content": chunk_text.strip(),
                    "chunk_index": chunk_index,
                    "metadata": {
                        "start_char": start,
                        "end_char": end,
                        "chunk_size": len(chunk_text),
                        "original_filename": original_filename or doc_id,
                        "source": original_filename or doc_id
                    }
                })
                chunk_index += 1
            
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _rebuild_index(self, document_ids: List[str], chunks: List[Dict[str, Any]] = None):
        """
        Rebuild search indexes after chunking.
        
        Args:
            document_ids: List of document IDs that were chunked
            chunks: Optional list of chunks (if provided, use directly instead of loading from files)
        """
        try:
            # Use provided chunks or load from main chunks.json
            all_chunks = chunks if chunks else []
            
            if not all_chunks and document_ids:
                # Fallback: load from chunks.json and filter by document_ids
                chunks_file = PROCESSED_DIR / "chunks.json"
                if chunks_file.exists():
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        all_loaded = json.load(f)
                    all_chunks = [c for c in all_loaded if c.get("document_id") in document_ids]
            
            if not all_chunks:
                logger.warning("No chunks to index")
                return
            
            logger.info(f"Indexing {len(all_chunks)} chunks from {len(document_ids)} documents")
            
            # Save combined chunks to chunks.json (with file locking to prevent corruption)
            combined_file = PROCESSED_DIR / "chunks.json"
            
            # Cross-platform file locking
            import sys
            lock_file = PROCESSED_DIR / "chunks.lock"
            lock = None
            
            try:
                lock = open(lock_file, 'w')
                
                # Platform-specific locking
                if sys.platform == 'win32':
                    # Windows: use msvcrt
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Unix/Linux: use fcntl
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                
                # Load existing chunks if any
                existing_chunks = []
                if combined_file.exists():
                    try:
                        with open(combined_file, 'r', encoding='utf-8') as f:
                            existing_chunks = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Corrupted chunks.json, starting fresh: {e}")
                        existing_chunks = []
                
                # Merge (remove old chunks from same documents)
                existing_doc_ids = set(document_ids)
                filtered_chunks = [
                    c for c in existing_chunks
                    if c.get("document_id") not in existing_doc_ids
                ]
                filtered_chunks.extend(all_chunks)
                
                # Save updated chunks file
                with open(combined_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_chunks, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved {len(filtered_chunks)} total chunks. Starting auto-indexing...")
                
            finally:
                # Release lock and close file
                if lock:
                    try:
                        if sys.platform == 'win32':
                            import msvcrt
                            try:
                                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                            except:
                                pass
                        else:
                            import fcntl
                            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    except:
                        pass
                    lock.close()

            # Call KB service to perform actual indexing
            # We import here to avoid circular dependencies
            from .kb_service import kb_service
            
            # Check if auto-indexing is already in progress
            # If so, set pending flag - chunks are saved to file, will be indexed when current one finishes
            global _auto_index_in_progress, _pending_reindex
            with _auto_index_lock:
                if _auto_index_in_progress:
                    _pending_reindex = True
                    logger.info("Auto-indexing already in progress, marking pending. Will reindex after current one completes.")
                    return
                _auto_index_in_progress = True
                _pending_reindex = False  # Clear pending flag as we're starting
            
            try:
                sub_job_id = f"auto_index_{uuid.uuid4().hex[:8]}"
                
                # Initialize job entry (required by _reindex_worker)
                from .kb_service import _reindex_jobs
                from datetime import datetime
                _reindex_jobs[sub_job_id] = {
                    "id": sub_job_id,
                    "status": "pending",
                    "progress": 0.0,
                    "chunks_indexed": 0,
                    "started_at": datetime.now().isoformat(),
                    "completed_at": None,
                    "error_message": None
                }
                
                # Run reindex in background thread to avoid blocking API
                def run_reindex():
                    global _auto_index_in_progress, _pending_reindex
                    try:
                        kb_service._reindex_worker(
                            job_id=sub_job_id,
                            rebuild_vectors=True,
                            rebuild_bm25=True,
                            document_ids=None  # Reindex all from chunks.json
                        )
                        logger.info("Auto-indexing completed successfully")
                    finally:
                        # Check if there's a pending reindex request
                        should_reindex_again = False
                        with _auto_index_lock:
                            _auto_index_in_progress = False
                            if _pending_reindex:
                                should_reindex_again = True
                                _pending_reindex = False
                        
                        # Trigger another reindex if pending
                        if should_reindex_again:
                            logger.info("Pending reindex detected, triggering another reindex...")
                            # Small delay to ensure any file writes are complete
                            import time
                            time.sleep(1)
                            # Recursively call _rebuild_index (will start a new thread)
                            chunking_service._rebuild_index([], [])
                
                import threading
                thread = threading.Thread(target=run_reindex)
                thread.start()
                
                # Return immediately - don't wait for indexing to complete
                return
                
            except Exception as e:
                with _auto_index_lock:
                    _auto_index_in_progress = False
                raise e

            
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status or None
        """
        return _jobs.get(job_id)
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of active chunking jobs.
        
        Returns:
            List of active jobs with pending or running status
        """
        active = []
        for job_id, job in _jobs.items():
            if job.get("status") in ["pending", "running"]:
                active.append(job)
        
        # Also include auto-indexing as an active job if in progress
        if _auto_index_in_progress:
            active.append({
                "id": "auto_indexing",
                "status": "running",
                "type": "auto_indexing",
                "message": "Auto-indexing in progress"
            })
        
        return active
    
    def is_auto_indexing(self) -> bool:
        """Check if auto-indexing is currently in progress."""
        return _auto_index_in_progress
    
    def update_config(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Update default chunking config.
        
        Args:
            chunk_size: New default chunk size
            chunk_overlap: New default overlap
        """
        if chunk_size:
            self.default_chunk_size = chunk_size
        if chunk_overlap:
            self.default_chunk_overlap = chunk_overlap


# Singleton instance
chunking_service = ChunkingService()
