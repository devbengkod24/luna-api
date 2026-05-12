"""Application constants."""

from enum import Enum


class MessageType(Enum):
    USER = "user"
    TOOL = "tool"
    ASSISTANT = "assistant"


class Character(int, Enum):
    VENDRA = 1
    SHAZIA = 0


class StorageType(Enum):
    """Storage type enumeration."""

    RAG = "rag"
    FACES = "faces"


class FileStatus(str, Enum):
    """File processing status."""

    UPLOADING = "uploading"  # File sedang di-upload ke MinIO
    UPLOADED = "uploaded"  # File sudah aman di MinIO
    INDEXING = "indexing"  # Sedang chunking, embedding, dan simpan ke Qdrant
    COMPLETED = "completed"  # Pengetahuan sudah aktif
    FAILED = "failed"  # Terjadi kesalahan
