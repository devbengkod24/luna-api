"""Application constants."""

from enum import Enum

# The shape of every id this service mints (conversations, messages, files are
# UUIDv7). An id of any other shape is not ours, and checking at the boundary
# keeps it from reaching the database as a type error and coming back as a 500.
UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"


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


class FileStatus(str, Enum):
    """File processing status."""

    UPLOADING = "uploading"  # File sedang di-upload ke MinIO
    UPLOADED = "uploaded"  # File sudah aman di MinIO
    INDEXING = "indexing"  # Sedang chunking, embedding, dan simpan ke Qdrant
    COMPLETED = "completed"  # Pengetahuan sudah aktif
    FAILED = "failed"  # Terjadi kesalahan
