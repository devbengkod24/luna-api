"""Service port interfaces (external services / non-DB adapters)."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from src.domain.entities import UserContext
from src.domain.entities.face import FaceData


class IAgentService(ABC):
    """Interface for AI Agent adapter (with tools)."""

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        context: Optional[UserContext] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        ...

    @abstractmethod
    async def generate_title(self, title_prompt: str) -> str:
        ...


class IRAGService(ABC):
    """Interface for RAG retrieval and document indexing."""

    @abstractmethod
    async def index_document(
        self,
        file_id: str,
        file_path: str,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Index document into vector store."""
        ...

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[dict]:
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
        self, embedding: List[float], threshold: float = 0.6
    ) -> Optional[Tuple[FaceData, float]]:
        """Find closest matching face. Returns (face, score) or None."""
        ...

    @abstractmethod
    async def get_all(self) -> List[FaceData]:
        """List all registered faces (without embeddings)."""
        ...

    @abstractmethod
    async def get_all_cursor(
        self, limit: int, cursor: Optional[str] = None
    ) -> Tuple[List[FaceData], Optional[str]]:
        """List faces with cursor-based pagination. Returns (faces, next_cursor)."""
        ...

    @abstractmethod
    async def get_by_id(self, face_id: str) -> Optional[FaceData]:
        """Get a face by its ID."""
        ...

    @abstractmethod
    async def get_by_nim(self, nim: str) -> Optional[FaceData]:
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
        ...


class ITTSService(ABC):
    """Text-to-Speech service interface."""

    @abstractmethod
    async def synthesize(self, text: str, character: int) -> str:
        ...
