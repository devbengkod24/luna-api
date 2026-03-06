"""Database repository implementation."""
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.adapters.persistence.models import ChatSessionModel, MessageModel, FileModel
from src.domain.entities import ChatSession, Message, File
from src.ports.repositories import IChatSessionRepository, IMessageRepository, IFileRepository
from src.adapters.persistence.mappers import ChatSessionMapper, MessageMapper, FileMapper


class ChatSessionRepository(IChatSessionRepository):    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, chat_session: ChatSession) -> ChatSession:
        model = ChatSessionMapper.to_model(chat_session)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        
        return ChatSessionMapper.to_entity(model)
    
    async def get_session_by_id(self, id: str) -> ChatSession | None:
        stmt = select(ChatSessionModel).where(ChatSessionModel.id == id).options(selectinload(ChatSessionModel.messages))
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if not model:
            return None
        return ChatSessionMapper.to_entity(model)
        
    async def list_sessions(self) -> list[ChatSession]:
        stmt = select(ChatSessionModel).order_by(ChatSessionModel.updated_at.desc())
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [ChatSessionMapper.to_entity(model) for model in models]
    
    async def update_session_title(self, id: str, title: str) -> bool:
        stmt = select(ChatSessionModel).where(ChatSessionModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if not model:
            return False
        model.title = title
        await self.session.flush()
        return True

class MessageRepository(IMessageRepository):
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, message: Message) -> Message:
        model = MessageMapper.to_model(message)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return MessageMapper.to_entity(model)
    
    async def get_message_by_id(self, id: str) -> Message | None:
        stmt = select(MessageModel).where(MessageModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if not model:
            return None
        return MessageMapper.to_entity(model)

class FileRepository(IFileRepository):
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, file: File) -> File:
        model = FileMapper.to_model(file)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return FileMapper.to_entity(model)
    
    async def get_file_by_id(self, id: str) -> File | None:
        stmt = select(FileModel).where(FileModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if not model:
            return None
        return FileMapper.to_entity(model)

    async def list_files(self) -> list[File]:
        stmt = select(FileModel).order_by(FileModel.updated_at.desc())
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [FileMapper.to_entity(model) for model in models]
    
    async def update(self, file: File) -> File:
        stmt = (
            update(FileModel)  
            .where(FileModel.id == file.id)
            .values(
                filename=file.filename,
                size_bytes=file.size_bytes,
                content_type=file.content_type,
                storage_path=file.storage_path,
                status=file.status,
                tags=file.tags or [],
                indexed_at=file.indexed_at,
            )
        )
        await self.session.execute(stmt)
        await self.session.commit()
        return await self.get_file_by_id(file.id)