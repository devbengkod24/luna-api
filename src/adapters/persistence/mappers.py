"""Mappers to convert between ORM models and domain entities."""

from src.domain.entities import ChatSession, Message, File
from src.adapters.persistence.models import (
    ChatSessionModel,
    MessageModel,
    FileModel,
)

class ChatSessionMapper:
    @staticmethod
    def to_entity(model: ChatSessionModel) -> ChatSession:
        return ChatSession(
            id=str(model.id),
            title=model.title,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def to_model(entity: ChatSession) -> ChatSessionModel:
        kwargs: dict = dict(
            title=entity.title,
        )
        if entity.id is not None:
            kwargs["id"] = entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        if entity.updated_at is not None:
            kwargs["updated_at"] = entity.updated_at
        return ChatSessionModel(**kwargs)


class MessageMapper:
    @staticmethod
    def to_entity(model: MessageModel) -> Message:
        return Message(
            id=str(model.id),
            session_id=str(model.session_id),
            character=model.character,
            base64_audio=model.base64_audio,
            context=model.context,
            content=model.content,
            created_at=model.created_at,
        )

    @staticmethod
    def to_model(entity: Message) -> MessageModel:
        kwargs: dict = dict(
            session_id=entity.session_id,
            character=entity.character,
            base64_audio=entity.base64_audio,
            context=entity.context,
            content=entity.content,
        )
        if entity.id is not None:
            kwargs["id"] = entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        if entity.updated_at is not None:
            kwargs["updated_at"] = entity.updated_at
        return MessageModel(**kwargs)


class FileMapper:
    @staticmethod
    def to_entity(model: FileModel) -> File:
        return File(
            id=str(model.id),
            filename=model.filename,
            size_bytes=model.size_bytes,
            content_type=model.content_type,
            storage_path=model.storage_path,
            status=model.status,
            tags=model.tags or [],
            indexed_at=model.indexed_at,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def to_model(entity: File) -> FileModel:
        kwargs: dict = dict(
            filename=entity.filename,
            size_bytes=entity.size_bytes,
            content_type=entity.content_type,
            storage_path=entity.storage_path,
            status=entity.status,
            tags=entity.tags or [],
            indexed_at=entity.indexed_at,
        )
        if entity.id is not None:
            kwargs["id"] = entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        if entity.updated_at is not None:
            kwargs["updated_at"] = entity.updated_at
        return FileModel(**kwargs)
