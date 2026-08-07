"""Service port interfaces (external services / non-DB adapters)."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, ClassVar

from src.domain.entities import Message, UserContext
from src.domain.entities.face import FaceData
from src.domain.events import StreamEvent


class IAgentService(ABC):
    """Interface for AI Agent adapter (with tools)."""

    @abstractmethod
    async def generate_stream(
        self,
        user_input: str,
        messages: list[Message],
        context: UserContext | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        ...

    @abstractmethod
    async def preprocess_tts(self, text: str, system_prompt: str) -> str:
        ...


class IRAGService(ABC):
    """Interface for RAG retrieval and document indexing."""

    @abstractmethod
    async def index_document(
        self,
        file_id: str,
        file_path: str,
        tags: list[str] | None = None,
    ) -> bool:
        """Index document into vector store."""
        ...

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve relevant documents."""
        ...

    @abstractmethod
    async def delete_by_file_id(self, file_id: str) -> bool:
        """Delete all vectors for a file."""
        ...


class IStorageService(ABC):
    """Storage service interface."""

    @abstractmethod
    async def upload_file(self, file_content: bytes, filename: str, content_type: str) -> str:
        """Upload file and return storage path."""
        ...

    @abstractmethod
    async def download_file(self, storage_path: str) -> bytes:
        """Download file bytes from storage."""
        ...

    @abstractmethod
    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from storage."""
        ...


class IFaceService(ABC):
    """Interface for face vector store operations (Qdrant)."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store collection."""
        ...

    @abstractmethod
    async def add_face(self, face: FaceData) -> str:
        """Store face data and return face_id."""
        ...

    @abstractmethod
    async def search_face(
        self, embedding: list[float], threshold: float = 0.6
    ) -> tuple[FaceData, float] | None:
        """Find closest matching face. Returns (face, score) or None."""
        ...

    @abstractmethod
    async def get_all(self) -> list[FaceData]:
        """List all registered faces (without embeddings)."""
        ...

    @abstractmethod
    async def get_all_cursor(
        self, limit: int, cursor: str | None = None
    ) -> tuple[list[FaceData], str | None]:
        """List faces with cursor-based pagination. Returns (faces, next_cursor)."""
        ...

    @abstractmethod
    async def get_by_id(self, face_id: str) -> FaceData | None:
        """Get a face by its ID."""
        ...

    @abstractmethod
    async def get_by_nim(self, nim: str) -> FaceData | None:
        """Get a face by student NIM."""
        ...

    @abstractmethod
    async def delete(self, face_id: str) -> bool:
        """Delete a face by ID. Returns True if deleted."""
        ...


class ISTTService(ABC):
    """Speech-to-Text service interface."""

    @abstractmethod
    async def transcribe(self, speech_base64: str) -> str:
        """Transcribe a base64-encoded audio file (WAV/MP3/OGG/etc)."""
        ...

    @abstractmethod
    async def transcribe_pcm(self, pcm16: bytes, sample_rate: int) -> str:
        """Transcribe raw 16-bit mono PCM directly. Skips WAV/base64 wrapping."""
        ...


class ITTSService(ABC):
    """Text-to-Speech service interface."""

    supports_streaming: ClassVar[bool] = False

    @abstractmethod
    async def synthesize(self, text: str, character: int) -> str:
        """Synthesize text and return base64-encoded WAV."""
        ...

    async def synthesize_stream(
        self, text: str, character: int
    ) -> AsyncGenerator[bytes, None]:
        """Yield raw PCM16 byte chunks. Override when supports_streaming=True."""
        raise NotImplementedError
        yield b""  # satisfy the async generator protocol
