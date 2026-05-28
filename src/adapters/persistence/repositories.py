"""Database repository implementation."""
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.persistence.models import ConversationModel, MessageModel, FileModel
from src.domain.entities import Conversation, Message, File
from src.ports.repositories import IConversationRepository, IMessageRepository, IFileRepository
from src.adapters.persistence.mappers import ConversationMapper, MessageMapper, FileMapper


class ConversationRepository(IConversationRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, conversation: Conversation) -> Conversation:
        model = ConversationMapper.to_model(conversation)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return ConversationMapper.to_entity(model)

    async def get_by_id(self, id: str) -> Conversation | None:
        stmt = select(ConversationModel).where(ConversationModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if not model:
            return None
        return ConversationMapper.to_entity(model)

    async def list_cursor(
        self,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[list[Conversation], str | None]:
        limit_fetch = limit + 1
        stmt = select(ConversationModel)
        if cursor:
            try:
                cursor_uuid = UUID(cursor)
            except (ValueError, TypeError):
                cursor_uuid = None
            if cursor_uuid is not None:
                stmt = stmt.where(ConversationModel.id < cursor_uuid)

        stmt = stmt.order_by(ConversationModel.id.desc()).limit(limit_fetch)
        result = await self.session.execute(stmt)
        models = result.scalars().all()

        has_next = len(models) == limit_fetch
        if has_next:
            models = models[:limit]

        conversations = [ConversationMapper.to_entity(m) for m in models]

        next_cursor: str | None = None
        if has_next and models:
            last = models[-1]
            next_cursor = str(last.id)

        return conversations, next_cursor

    async def update_title(self, id: str, title: str) -> bool:
        stmt = (
            update(ConversationModel)
            .where(ConversationModel.id == id)
            .values(title=title)
        )
        result = await self.session.execute(stmt)
        return (result.rowcount or 0) > 0

    async def delete(self, id: str) -> bool:
        # ON DELETE CASCADE on messages.conversation_id handles child rows.
        stmt = delete(ConversationModel).where(ConversationModel.id == id)
        result = await self.session.execute(stmt)
        return (result.rowcount or 0) > 0


class MessageRepository(IMessageRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, message: Message) -> Message:
        model = MessageMapper.to_model(message)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return MessageMapper.to_entity(model)

    async def get_by_id(self, id: str) -> Message | None:
        stmt = select(MessageModel).where(MessageModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if not model:
            return None
        return MessageMapper.to_entity(model)

    async def get_by_conversation_id(self, conversation_id: str) -> list[Message]:
        stmt = (
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.created_at.asc())
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [MessageMapper.to_entity(m) for m in models]

    async def get_recent_by_conversation_id(
        self, conversation_id: str, limit: int
    ) -> list[Message]:
        stmt = (
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        models = list(result.scalars().all())
        models.reverse()  # return ascending by created_at
        return [MessageMapper.to_entity(m) for m in models]

    async def get_by_conversation_id_cursor(
        self,
        conversation_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[list[Message], str | None]:
        limit_fetch = limit + 1
        stmt = select(MessageModel).where(MessageModel.conversation_id == conversation_id)

        if cursor:
            try:
                cursor_uuid = UUID(cursor)
            except (ValueError, TypeError):
                cursor_uuid = None
            if cursor_uuid is not None:
                stmt = stmt.where(MessageModel.id > cursor_uuid)

        stmt = stmt.order_by(MessageModel.id.asc()).limit(limit_fetch)
        result = await self.session.execute(stmt)
        models = result.scalars().all()

        has_next = len(models) == limit_fetch
        if has_next:
            models = models[:limit]

        messages = [MessageMapper.to_entity(m) for m in models]
        next_cursor = None
        if has_next and models:
            last = models[-1]
            next_cursor = str(last.id)

        return messages, next_cursor

    async def delete(self, id: str) -> bool:
        stmt = delete(MessageModel).where(MessageModel.id == id)
        result = await self.session.execute(stmt)
        return (result.rowcount or 0) > 0


class FileRepository(IFileRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, file: File) -> File:
        model = FileMapper.to_model(file)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return FileMapper.to_entity(model)

    async def get_by_id(self, id: str) -> File | None:
        stmt = select(FileModel).where(FileModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if not model:
            return None
        return FileMapper.to_entity(model)

    async def list(self) -> list[File]:
        stmt = select(FileModel).order_by(FileModel.updated_at.desc())
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [FileMapper.to_entity(model) for model in models]

    async def update(self, file: File) -> File:
        from src.shared.constants import FileStatus
        stmt = (
            update(FileModel)
            .where(FileModel.id == file.id)
            .values(
                filename=file.filename,
                size_bytes=file.size_bytes,
                content_type=file.content_type,
                storage_path=file.storage_path,
                status=file.status.value if isinstance(file.status, FileStatus) else file.status,
                tags=file.tags or [],
                indexed_at=file.indexed_at,
            )
        )
        await self.session.execute(stmt)
        await self.session.flush()
        return file

    async def delete(self, id: str) -> bool:
        stmt = delete(FileModel).where(FileModel.id == id)
        result = await self.session.execute(stmt)
        return (result.rowcount or 0) > 0
