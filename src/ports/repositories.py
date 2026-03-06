"""Repository port interfaces (persistence layer contracts)."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.chat import ChatSession, Message


class IChatSessionRepository(ABC):
    """Interface for chat repository."""

    @abstractmethod
    async def create(self, chat_session: ChatSession) -> ChatSession:
        """Create a new chat session."""
        ...

    @abstractmethod
    async def get_session_by_id(self, id: str) -> Optional[ChatSession]:
        """Get chat session by UUID."""
        ...

    @abstractmethod
    async def list_sessions(self) -> List[ChatSession]:
        """List all chat sessions."""
        ...

    @abstractmethod
    async def update_session_title(self, id: str, title: str) -> bool:
        """Update session title."""
        ...

class IMessageRepository(ABC):
    """Interface for message repository."""

    @abstractmethod
    async def create(self, message: Message) -> Message:
        """Create a new message."""
        ...

    @abstractmethod
    async def get_message_by_id(self, id: str) -> Optional[Message]:
        """Get message by UUID."""
        ...