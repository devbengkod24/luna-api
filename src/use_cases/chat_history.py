"""Chat history use cases."""
from typing import List, Optional

from src.ports.chat_repository import IChatRepository
from src.domain.entities.chat import ChatSessionEntity


class GetChatHistoryUseCase:
    """Use case for getting chat history list."""
    
    def __init__(self, chat_repository: IChatRepository):
        """Initialize use case with dependencies.
        
        Args:
            chat_repository: Repository for chat persistence
        """
        self.chat_repository = chat_repository
    
    async def execute(self) -> List[ChatSessionEntity]:
        """Get list of all chat sessions.
        
        Returns:
            List[ChatSessionEntity]: List of chat sessions
        """
        return await self.chat_repository.list_sessions()


class GetChatSessionUseCase:
    """Use case for getting a specific chat session with messages."""
    
    def __init__(self, chat_repository: IChatRepository):
        """Initialize use case with dependencies.
        
        Args:
            chat_repository: Repository for chat persistence
        """
        self.chat_repository = chat_repository
    
    async def execute(self, session_uuid: str) -> Optional[ChatSessionEntity]:
        """Get a specific chat session by UUID.
        
        Args:
            session_uuid: Session unique identifier
            
        Returns:
            Optional[ChatSessionEntity]: Session entity or None if not found
        """
        return await self.chat_repository.get_session_by_uuid(session_uuid)

