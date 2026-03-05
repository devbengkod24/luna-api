"""Service port interfaces (external services / non-DB adapters).

Naming is kept compatible with existing code (`*Adapter`, `*Store`) while the
module structure follows the `luna-api` style: repositories vs services.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.domain.entities.chat import UserContext
from src.domain.entities.knowledge import RetrievalResultEntity


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


class IRAGService(ABC):
    """Interface for RAG retrieval (read/search)."""

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 5) -> list[RetrievalResultEntity]:
        ...

    @abstractmethod
    async def add_document(self, file_path: Path) -> int:
        ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        ...

    @abstractmethod
    async def get_all_document_ids(self) -> list[str]:
        ...

    @abstractmethod
    async def get_all_documents_metadata(self) -> list[dict]:
        ...
    

class IStorageService(ABC):
    """Storage service interface - simple file operations only."""

    @abstractmethod
    async def save_file(self, file_content: bytes, filename: str) -> Path:
        ...

    @abstractmethod
    async def delete_file(self, filename: str) -> bool:
        ...

    @abstractmethod
    async def file_exists(self, filename: str) -> bool:
        ...

    @abstractmethod
    async def list_files(self) -> List[str]:
        ...

    @abstractmethod
    async def read_file(self, filename: str) -> bytes:
        ...


class ISTTService(ABC):
    """Speech-to-Text service interface."""

    @abstractmethod
    async def transcribe(self, speech_base64: str) -> str:
        ...


class ITTSService(ABC):
    """Text-to-Speech service interface."""

    @abstractmethod
    async def synthesize(self, text: str) -> str:
        ...

