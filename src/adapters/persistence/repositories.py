"""Database repository implementation."""
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.adapters.postgres.models import ChatSession, Message
from src.domain.entities.chat import ChatSessionEntity, MessageEntity
from src.ports.repositories import IChatRepository


class ChatRepository(IChatRepository):
    """Repository for chat-related database operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def create_session(self, uuid: str, title: str) -> ChatSessionEntity:
        """Create a new chat session.
        
        Args:
            uuid: Unique identifier for the session
            title: Session title
            
        Returns:
            ChatSessionEntity: Created session entity
        """
        session = ChatSession(uuid=uuid, title=title)
        self.session.add(session)
        await self.session.flush()
        await self.session.refresh(session)
        
        return ChatSessionEntity(
            id=session.id,
            uuid=session.uuid,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            messages=[],
        )
    
    async def get_session_by_uuid(self, uuid: str) -> Optional[ChatSessionEntity]:
        """Get chat session by UUID with all messages.
        
        Args:
            uuid: Session unique identifier
            
        Returns:
            Optional[ChatSessionEntity]: Session entity or None if not found
        """
        stmt = select(ChatSession).where(ChatSession.uuid == uuid).options(selectinload(ChatSession.messages))
        result = await self.session.execute(stmt)
        session = result.scalar_one_or_none()
        
        if not session:
            return None
        
        messages = [
            MessageEntity(
                id=msg.id,
                session_id=msg.session_id,
                role=msg.role,
                content=msg.content,
                context=msg.context,
                created_at=msg.created_at,
            )
            for msg in session.messages
        ]
        
        return ChatSessionEntity(
            id=session.id,
            uuid=session.uuid,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            messages=messages,
        )
    
    async def list_sessions(self) -> List[ChatSessionEntity]:
        """List all chat sessions.
        
        Returns:
            List[ChatSessionEntity]: List of session entities
        """
        stmt = select(ChatSession).order_by(ChatSession.updated_at.desc())
        result = await self.session.execute(stmt)
        sessions = result.scalars().all()
        
        return [
            ChatSessionEntity(
                id=session.id,
                uuid=session.uuid,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                messages=[],
            )
            for session in sessions
        ]
    
    async def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        context: Optional[dict] = None,
    ) -> MessageEntity:
        """Add a message to a session.
        
        Args:
            session_id: Session database ID
            role: Message role ('user' or 'assistant')
            content: Message content
            context: Optional context (emotion, name, etc.)
            
        Returns:
            MessageEntity: Created message entity
        """
        message = Message(
            session_id=session_id,
            role=role,
            content=content,
            context=context,
        )
        self.session.add(message)
        await self.session.flush()
        await self.session.refresh(message)
        
        return MessageEntity(
            id=message.id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            context=message.context,
            created_at=message.created_at,
        )
    
    async def update_session_title(self, uuid: str, title: str) -> bool:
        """Update session title.
        
        Args:
            uuid: Session unique identifier
            title: New title
            
        Returns:
            bool: True if updated, False if not found
        """
        stmt = select(ChatSession).where(ChatSession.uuid == uuid)
        result = await self.session.execute(stmt)
        session = result.scalar_one_or_none()
        
        if not session:
            return False
        
        session.title = title
        await self.session.flush()
        return True

