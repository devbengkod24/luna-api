"""Storage type entities."""
from enum import Enum


class StorageType(Enum):
    """Storage type enumeration."""
    RAG = "rag"  # RAG documents
    FACES = "faces"  # Face images
