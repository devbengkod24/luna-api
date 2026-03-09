"""Mappers to convert between ORM models and domain entities."""

import json
from datetime import datetime

from src.shared.constants import FileStatus
from src.domain.entities import (
    Conversation,
    Message,
    File,
    UserMessage,
    ToolMessage,
    AIMessage,
    ToolsCall,
    Function,
    ResponseMetadata,
    UsageMetadata,
    UserContext,
)
from src.shared.constants import Character, MessageType
from src.adapters.persistence.models import (
    ConversationModel,
    MessageModel,
    FileModel,
)

class ConversationMapper:
    @staticmethod
    def to_entity(model: ConversationModel) -> Conversation:
        return Conversation(
            id=str(model.id),
            title=model.title,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def to_model(entity: Conversation) -> ConversationModel:
        kwargs: dict = dict(
            title=entity.title,
        )
        if entity.id is not None:
            kwargs["id"] = entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
        if entity.updated_at is not None:
            kwargs["updated_at"] = entity.updated_at
        return ConversationModel(**kwargs)


def _body_to_dict(body: UserMessage | ToolMessage | AIMessage) -> dict:
    """Serialize message content (domain) to JSONB dict."""
    if isinstance(body, UserMessage):
        return {"type": MessageType.USER.value, "content": body.content}
    elif isinstance(body, ToolMessage):
        return {
            "type": MessageType.TOOL.value,
            "content": body.content,
            "tool_call_id": body.tool_call_id,
            "name": body.name,
        }
    elif isinstance(body, AIMessage):
        tool_calls = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (body.additional_kwargs or [])
        ]
        rm = body.response_metadata
        return {
            "type": MessageType.ASSISTANT.value,
            "content": body.content,
            "additional_kwargs": {"tool_calls": tool_calls},
            "response_metadata": {
                "model": rm.model,
                "done_reason": rm.done_reason,
            },
            "usage_metadata": {
                "input_tokens": body.usage_metadata.input_tokens,
                "output_tokens": body.usage_metadata.output_tokens,
                "total_tokens": body.usage_metadata.total_tokens,
            },
        }
    return {"type": MessageType.USER.value, "content": ""}


def _dict_to_body(d: dict) -> UserMessage | ToolMessage | AIMessage:
    """Deserialize JSONB dict to domain message content."""
    if not d:
        return UserMessage(content="")
    msg_type = (d.get("type") or MessageType.USER.value).lower()
    content = d.get("content", "")

    if msg_type == MessageType.USER.value:
        return UserMessage(content=content)
    elif msg_type == MessageType.TOOL.value:
        return ToolMessage(content=content, tool_call_id=d.get("tool_call_id", ""), name=d.get("name", ""))
    elif msg_type == MessageType.ASSISTANT.value:
        return _dict_to_ai_message(d)
    else:
        return UserMessage(content=content)


def _dict_to_ai_message(d: dict) -> AIMessage:
    """Build AIMessage from JSONB dict."""
    rm = d.get("response_metadata") or {}
    um = d.get("usage_metadata") or {}
    tool_calls_raw = (d.get("additional_kwargs") or {}).get("tool_calls", [])
    tool_calls = []
    for tc in tool_calls_raw:
        fn = tc.get("function") or {}
        args = fn.get("arguments", "")
        if isinstance(args, dict):
            args = json.dumps(args)
        tool_calls.append(
            ToolsCall(
                id=tc.get("id", ""),
                type=tc.get("type", ""),
                function=Function(name=fn.get("name", ""), arguments=args or ""),
            )
        )
    return AIMessage(
        content=d.get("content", ""),
        additional_kwargs=tool_calls,
        response_metadata=ResponseMetadata(
            model=rm.get("model", ""),
            done_reason=rm.get("done_reason", ""),
        ),
        usage_metadata=UsageMetadata(
            input_tokens=um.get("input_tokens", 0),
            output_tokens=um.get("output_tokens", 0),
            total_tokens=um.get("total_tokens", 0),
        ),
    )


class MessageMapper:
    @staticmethod
    def to_entity(model: MessageModel) -> Message:
        context = None
        if model.context:
            context = UserContext(
                emotion=model.context.get("emotion"),
                name=model.context.get("name"),
                nim=model.context.get("nim"),
            )
        body = model.body if isinstance(model.body, dict) else {}
        return Message(
            id=str(model.id),
            conversation_id=str(model.conversation_id),
            character=Character(int(model.character)),
            context=context,
            body=_dict_to_body(body),
            created_at=model.created_at,
        )

    @staticmethod
    def to_model(entity: Message) -> MessageModel:
        context_dict = None
        if entity.context is not None:
            context_dict = {
                "emotion": entity.context.emotion,
                "name": entity.context.name,
                "nim": entity.context.nim,
            }
        kwargs: dict = dict(
            conversation_id=entity.conversation_id,
            character=entity.character,
            context=context_dict,
            body=_body_to_dict(entity.body),
        )
        if entity.id is not None:
            kwargs["id"] = entity.id
        if entity.created_at is not None:
            kwargs["created_at"] = entity.created_at
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
            status=FileStatus(model.status),
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
            status=entity.status.value if isinstance(entity.status, FileStatus) else entity.status,
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
