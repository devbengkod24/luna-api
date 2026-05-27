"""Application constants."""

from enum import Enum


class MessageType(Enum):
    USER = "user"
    TOOL = "tool"
    ASSISTANT = "assistant"


class Character(int, Enum):
    # Active character is LUNA. VENDRA is reserved as a second slot for future
    # multi-persona development; nothing in production currently selects it.
    LUNA = 0

    @property
    def slug(self) -> str:
        """Lowercase identifier used for voice-clone asset paths (luna.wav, vendra.wav, ...)."""
        return self.name.lower()


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
