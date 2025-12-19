"""Routes Package."""

from .query import router as query_router
from .health import router as health_router
from .documents import router as documents_router
from .chunking import router as chunking_router
from .knowledge_base import router as kb_router

__all__ = [
    "query_router",
    "health_router",
    "documents_router",
    "chunking_router",
    "kb_router",
]
