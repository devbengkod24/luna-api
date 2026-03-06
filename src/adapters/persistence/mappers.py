"""Mappers to convert between ORM models and domain entities."""
import uuid
from typing import Any, Optional

from src.domain.entities import User, Project, ApiKey, File, Conversation, ConversationItem, Response, Assistant, Usage
from src.shared.constants import ItemStatus, ItemType, ResponseStatus, Role
from src.adapters.persistence.models import (
    UserModel,
    ProjectModel,
    ApiKeyModel,
    FileModel,
    AssistantModel,
    ConversationModel,
    ConversationItemModel,
    ResponseModel,
)


class UserMapper:
    @staticmethod
    def to_entity(model: UserModel) -> User:
        return User(
            id=str(model.id),
            email=model.email,
            hashed_password=model.hashed_password,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def to_model(entity: User) -> UserModel:
        kwargs: dict = dict(
            email=entity.email,
            hashed_password=entity.hashed_password,
            is_active=entity.is_active,
        )
        if entity.id is not None:
            kwargs["id"] = entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        if entity.updated_at is not None:
            kwargs["updated_at"] = entity.updated_at
        return UserModel(**kwargs)


class ProjectMapper:
    @staticmethod
    def to_entity(model: ProjectModel) -> Project:
        return Project(
            id=str(model.id),
            name=model.name,
            slug=model.slug,
            created_by=str(model.created_by),
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def to_model(entity: Project) -> ProjectModel:
        kwargs: dict = dict(
            name=entity.name,
            slug=entity.slug,
            created_by=entity.created_by,
            is_active=entity.is_active,
        )
        if entity.id is not None:
            kwargs["id"] = entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        if entity.updated_at is not None:
            kwargs["updated_at"] = entity.updated_at
        return ProjectModel(**kwargs)


class ApiKeyMapper:
    @staticmethod
    def to_entity(model: ApiKeyModel) -> ApiKey:
        return ApiKey(
            id=str(model.id),
            project_id=str(model.project_id),
            label=model.label,
            hashed_key=model.hashed_key,
            masked_key=model.masked_key,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def to_model(entity: ApiKey) -> ApiKeyModel:
        kwargs: dict = dict(
            project_id=entity.project_id,
            label=entity.label,
            hashed_key=entity.hashed_key,
            masked_key=entity.masked_key,
        )
        if entity.id is not None:
            kwargs["id"] = entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        if entity.updated_at is not None:
            kwargs["updated_at"] = entity.updated_at
        return ApiKeyModel(**kwargs)


class AssistantMapper:
    @staticmethod
    def to_entity(model: AssistantModel) -> Assistant:
        return Assistant(
            id=str(model.id),
            project_id=str(model.project_id),
            name=model.name,
            instruction=model.instruction,
            tags=model.tags,
            temperature=model.temperature,
            top_p=model.top_p,
            max_output_tokens=model.max_output_tokens,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def to_model(entity: Assistant) -> AssistantModel:
        kwargs: dict = dict(
            project_id=entity.project_id,
            name=entity.name,
            instruction=entity.instruction,
            tags=entity.tags,
            temperature=entity.temperature,
            top_p=entity.top_p,
            max_output_tokens=entity.max_output_tokens,
        )
        if entity.id is not None:
            kwargs["id"] = uuid.UUID(entity.id) if isinstance(entity.id, str) else entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        if entity.updated_at is not None:
            kwargs["updated_at"] = entity.updated_at
        return AssistantModel(**kwargs)


class ConversationMapper:
    @staticmethod
    def to_entity(model: ConversationModel) -> Conversation:
        # metadata = getattr(model, "metadata_", None) or {}
        return Conversation(
            id=str(model.id),
            project_id=str(model.project_id),
            assistant_id=str(model.assistant_id) if model.assistant_id else None,
            title=model.title,
            # metadata=metadata,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def to_model(entity: Conversation) -> ConversationModel:
        kwargs: dict = dict(
            project_id=entity.project_id,
            assistant_id=entity.assistant_id,
            title=entity.title or "",
        )
        if entity.id is not None:
            kwargs["id"] = uuid.UUID(entity.id) if isinstance(entity.id, str) else entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        if entity.updated_at is not None:
            kwargs["updated_at"] = entity.updated_at
        return ConversationModel(**kwargs)


class ConversationItemMapper:
    """Mapper for ConversationItem entity ↔ ConversationItemModel."""

    @staticmethod
    def to_entity(model: ConversationItemModel) -> ConversationItem:
        return ConversationItem(
            conversation_id=str(model.conversation_id),
            item_type=ItemType(model.item_type) if isinstance(model.item_type, str) else model.item_type,
            status=ItemStatus(model.status) if isinstance(model.status, str) else model.status,
            message=model.message,
            created_at=model.created_at,
            id=str(model.id),
        )

    @staticmethod
    def to_model(
        entity: ConversationItem
    ) -> ConversationItemModel:
        kwargs: dict = dict(
            conversation_id=entity.conversation_id,
            item_type=entity.item_type.value if hasattr(entity.item_type, "value") else entity.item_type,
            status=entity.status.value if hasattr(entity.status, "value") else entity.status,
            message=entity.message,
        )
        if entity.id is not None:
            kwargs["id"] = uuid.UUID(entity.id) if isinstance(entity.id, str) else entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        return ConversationItemModel(**kwargs)


class ResponseMapper:
    """Mapper for Response entity ↔ ResponseModel."""

    @staticmethod
    def to_entity(model: ResponseModel) -> Response:
        assistant = model.assistant
        temperature = assistant.temperature if assistant else 0
        top_p = assistant.top_p if assistant else 0
        max_output_tokens = assistant.max_output_tokens if assistant else 0
        return Response(
            id=str(model.id),
            assistant_id=str(model.assistant_id) if model.assistant_id else None,
            status=ResponseStatus(model.status) if isinstance(model.status, str) else model.status,
            usage=Usage(
                input_tokens=model.input_tokens,
                output_tokens=model.output_tokens,
                total_tokens=model.total_tokens,
            ),
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            output=model.output,
            error=model.error,
            created_at=model.created_at,
            completed_at=model.completed_at,
        )

    @staticmethod
    def to_model(entity: Response) -> ResponseModel:
        kwargs: dict = dict(
            assistant_id=entity.assistant_id,
            status=entity.status.value if hasattr(entity.status, "value") else entity.status,
            input_tokens=entity.usage.input_tokens,
            output_tokens=entity.usage.output_tokens,
            total_tokens=entity.usage.total_tokens,
            output=entity.output,
            error=entity.error,
            completed_at=entity.completed_at,
        )
        if entity.id is not None:
            kwargs["id"] = uuid.UUID(entity.id) if isinstance(entity.id, str) else entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        return ResponseModel(**kwargs)


class FileMapper:
    @staticmethod
    def to_entity(model: FileModel) -> File:
        return File(
            id=str(model.id),
            project_id=str(model.project_id),
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
            project_id=entity.project_id,
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
