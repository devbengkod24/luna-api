"""RAG adapters (retriever + vector store).

Organized by *function* (rag) instead of framework/vendor.
Current implementations are re-exported from existing modules.
"""

from src.adapters.llamaindex.rag_adapter import RAGAdapter
from src.adapters.qdrant.rag_store import RAGStore

__all__ = [
    "RAGAdapter",
    "RAGStore",
]

