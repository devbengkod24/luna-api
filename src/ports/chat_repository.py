"""Repository port interfaces."""
from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.chat import ChatSessionEntity, MessageEntity

class IChatRepository(ABC):
    """Interface for chat repository."""
    
    @abstractmethod
    async def create_session(self, uuid: str, title: str) -> ChatSessionEntity:
        """Create a new chat session."""
        ...
    
    @abstractmethod
    async def get_session_by_uuid(self, uuid: str) -> Optional[ChatSessionEntity]:
        """Get chat session by UUID."""
        ...
    
    @abstractmethod
    async def list_sessions(self) -> List[ChatSessionEntity]:
        """List all chat sessions."""
        ...
    
    @abstractmethod
    async def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        context: Optional[dict] = None,
    ) -> MessageEntity:
        """Add a message to a session."""
        ...
    
    @abstractmethod
    async def update_session_title(self, uuid: str, title: str) -> bool:
        """Update session title."""
        ...

