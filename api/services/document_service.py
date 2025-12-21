"""
Document Service for Academic RAG API.

Handles document upload, storage, and management.
Now with auto-detection of manual processing output.
"""

import os
import uuid
import json
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Storage paths - use DATA_PATH env var or fallback to project-relative
def _get_data_dir() -> Path:
    """Get data directory from env or calculate from project root."""
    env_path = os.environ.get("DATA_PATH")
    if env_path:
        return Path(env_path).resolve()
    
    # Fallback: go up from this file to rag-deploy root, then into data/
    # File: rag-api/api/services/document_service.py
    # rag-api root: 3 levels up (rag-api/api/services -> rag-api)
    # rag-deploy root: 4 levels up (rag-api is submodule inside rag-deploy)
    rag_api_root = Path(__file__).resolve().parent.parent.parent
    
    # Check if we're inside rag-deploy (submodule) or standalone
    parent_data = rag_api_root.parent / "data"
    local_data = rag_api_root / "data"
    
    # Prefer parent rag-deploy/data if exists, else use rag-api/data
    if parent_data.exists():
        return parent_data.resolve()
    return local_data.resolve()

DATA_DIR = _get_data_dir()
DOCUMENTS_DIR = (DATA_DIR / "documents").resolve()
PROCESSED_DIR = (DATA_DIR / "processed").resolve()
METADATA_FILE = PROCESSED_DIR / "documents_metadata.json"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

# Log paths on startup for debugging
logger.info(f"DATA_DIR: {DATA_DIR}")
logger.info(f"DOCUMENTS_DIR: {DOCUMENTS_DIR}")

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def sanitize_doc_id(filename: str) -> str:
    """Create a clean document ID from filename."""
    name = Path(filename).stem
    name = re.sub(r'[^a-zA-Z0-9]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.lower().strip('_')
    return f"doc_{name[:50]}"


class DocumentService:
    """Service for managing documents with auto-detection of manual processing."""
    
    def __init__(self):
        """Initialize document service."""
        self._metadata = self._load_metadata()
        self._chunks_cache = None
        self._chunks_cache_mtime = None
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load documents metadata.
        
        First tries to load from metadata file.
        If empty or missing, auto-generates from documents folder and saves.
        """
        metadata = {"documents": {}}
        
        # Try to load existing metadata
        if METADATA_FILE.exists():
            try:
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        
        # If no documents in metadata, auto-scan folder and save
        if not metadata.get("documents"):
            metadata = self._auto_generate_metadata()
            # Save immediately so file exists for next startup
            self._metadata = metadata
            self._save_metadata()
        
        return metadata
    
    def _auto_generate_metadata(self) -> Dict[str, Any]:
        """Auto-generate metadata by scanning documents folder."""
        metadata = {"documents": {}}
        
        if not DOCUMENTS_DIR.exists():
            return metadata
        
        # Log scanning activity
        logger.info(f"Auto-generating metadata from {DOCUMENTS_DIR}")
        
        for file in DOCUMENTS_DIR.iterdir():
            if file.is_file() and file.suffix.lower() in ALLOWED_EXTENSIONS:
                doc_id = sanitize_doc_id(file.name)
                
                # Count chunks for this document
                chunk_count = self._count_chunks_for_source(file.name)
                
                metadata["documents"][doc_id] = {
                    "id": doc_id,
                    "filename": file.name,
                    "original_filename": file.name,
                    "size_bytes": file.stat().st_size,
                    "status": "indexed" if chunk_count > 0 else "uploaded",
                    "chunk_count": chunk_count,
                    "uploaded_at": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    "processed_at": datetime.fromtimestamp(file.stat().st_mtime).isoformat() if chunk_count > 0 else None,
                    "error_message": None
                }
        
        logger.info(f"Auto-detected {len(metadata['documents'])} documents")
        return metadata
    
    def _count_chunks_for_source(self, filename: str) -> int:
        """Count chunks that match a given source filename."""
        chunks = self._load_chunks_file()
        if not chunks:
            return 0
        return sum(1 for c in chunks if c.get("metadata", {}).get("source") == filename)
    
    def _load_chunks_file(self) -> List[Dict[str, Any]]:
        """
        Load chunks.json with caching.
        
        Returns cached version if file hasn't changed.
        """
        if not CHUNKS_FILE.exists():
            return []
        
        try:
            current_mtime = CHUNKS_FILE.stat().st_mtime
            
            # Check cache validity
            if self._chunks_cache is not None and self._chunks_cache_mtime == current_mtime:
                return self._chunks_cache
            
            # Load fresh
            with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                self._chunks_cache = json.load(f)
                self._chunks_cache_mtime = current_mtime
            
            return self._chunks_cache
        except Exception as e:
            logger.error(f"Failed to load chunks.json: {e}")
            return []
    
    def _save_metadata(self):
        """Save documents metadata to file."""
        try:
            logger.info(f"Saving metadata to: {METADATA_FILE}")
            with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Metadata saved successfully. Documents: {len(self._metadata.get('documents', {}))}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {METADATA_FILE}: {e}")
    
    def refresh_metadata(self):
        """Force refresh metadata from folder scan."""
        logger.info(f"Refreshing metadata from {DOCUMENTS_DIR}")
        self._metadata = self._auto_generate_metadata()
        self._save_metadata()
        doc_count = len(self._metadata.get("documents", {}))
        logger.info(f"Refresh complete. Found {doc_count} documents.")
        return doc_count
    
    def upload_document(
        self,
        file_content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """
        Upload a document.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Upload result with document_id
        """
        # Validate extension
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File type '{ext}' not allowed. "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Validate size
        if len(file_content) > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Generate document ID
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        # Create safe filename
        safe_filename = f"{doc_id}{ext}"
        file_path = DOCUMENTS_DIR / safe_filename
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Save metadata
        self._metadata["documents"][doc_id] = {
            "id": doc_id,
            "filename": safe_filename,
            "original_filename": filename,
            "size_bytes": len(file_content),
            "status": "uploaded",
            "chunk_count": 0,
            "uploaded_at": datetime.now().isoformat(),
            "processed_at": None,
            "error_message": None
        }
        self._save_metadata()
        
        logger.info(f"Document uploaded: {doc_id} ({filename})")
        
        return {
            "document_id": doc_id,
            "filename": safe_filename,
            "size_bytes": len(file_content),
            "message": "Document uploaded successfully"
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents.
        
        Returns:
            List of document info
        """
        return list(self._metadata["documents"].values())
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document info or None
        """
        return self._metadata["documents"].get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted, False if not found
        """
        doc = self._metadata["documents"].get(doc_id)
        if not doc:
            return False
        
        # Delete file
        file_path = DOCUMENTS_DIR / doc["filename"]
        if file_path.exists():
            file_path.unlink()
        
        # Delete per-document chunks file if exists (legacy)
        chunks_file = PROCESSED_DIR / f"{doc_id}_chunks.json"
        if chunks_file.exists():
            chunks_file.unlink()
        
        # Remove from metadata
        del self._metadata["documents"][doc_id]
        self._save_metadata()
        
        logger.info(f"Document deleted: {doc_id}")
        return True
    
    def get_document_chunks(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get chunks for a document by filtering chunks.json.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunks or None
        """
        doc = self.get_document(doc_id)
        if not doc:
            return None
        
        filename = doc.get("filename") or doc.get("original_filename")
        
        # Load and filter chunks
        all_chunks = self._load_chunks_file()
        
        # Try filtering by document_id first (API-created chunks)
        matching_chunks = [
            c for c in all_chunks 
            if c.get("document_id") == doc_id
        ]
        
        # Fallback: filter by source in metadata (manual chunks)
        if not matching_chunks and filename:
            matching_chunks = [
                c for c in all_chunks 
                if c.get("metadata", {}).get("source") == filename
            ]
        
        return matching_chunks if matching_chunks else None
    
    def update_document_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int = 0,
        error_message: str = None
    ):
        """
        Update document status.
        
        Args:
            doc_id: Document ID
            status: New status
            chunk_count: Number of chunks
            error_message: Error message if any
        """
        if doc_id in self._metadata["documents"]:
            self._metadata["documents"][doc_id]["status"] = status
            self._metadata["documents"][doc_id]["chunk_count"] = chunk_count
            if status == "indexed":
                self._metadata["documents"][doc_id]["processed_at"] = \
                    datetime.now().isoformat()
            if error_message:
                self._metadata["documents"][doc_id]["error_message"] = error_message
            self._save_metadata()
    
    def get_file_path(self, doc_id: str) -> Optional[Path]:
        """
        Get file path for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Path to file or None
        """
        doc = self.get_document(doc_id)
        if doc:
            return DOCUMENTS_DIR / doc["filename"]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get document statistics.
        
        Returns:
            Statistics dictionary
        """
        # Refresh metadata if empty
        if not self._metadata.get("documents"):
            self._metadata = self._auto_generate_metadata()
        
        docs = self._metadata["documents"].values()
        total_size = sum(d.get("size_bytes", 0) for d in docs)
        
        status_counts = {}
        for doc in docs:
            status = doc.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_documents": len(docs),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "status_counts": status_counts
        }


# Singleton instance
document_service = DocumentService()
