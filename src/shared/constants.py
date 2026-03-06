"""Application constants."""
from enum import Enum

class UserRole(Enum):
    """User role - represents the role of the user in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"


class Character(str, Enum):
    RUSSELL = "russell"
    FREDRICKSEN = "fredricksen"

class StorageType(Enum):
    """Storage type enumeration."""
    RAG = "rag"  # RAG documents
    FACES = "faces"  # Face images
