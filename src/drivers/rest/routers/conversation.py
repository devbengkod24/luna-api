"""Conversation REST routers."""

import logging
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Path, Query

from src.drivers import dependencies
from src.drivers.rest.schemas import (
    ConversationResponse,
    ListConversationsPaginatedResponse,
    ListMessagesPaginatedResponse,
    MessageResponse,
    UpdateConversationRequest,
    UserContextSchema,
)
from src.shared.constants import UUID_PATTERN
from src.use_cases import (
    DeleteConversationUseCase,
    DeleteMessageUseCase,
    ListConversationMessagesUseCase,
    ListConversationsUseCase,
    UpdateConversationUseCase,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversation", tags=["conversation"])

ConversationId = Annotated[str, Path(pattern=UUID_PATTERN)]
MessageId = Annotated[str, Path(pattern=UUID_PATTERN)]


def _to_conversation(c) -> ConversationResponse:
    return ConversationResponse(
        id=c.id or "",
        title=c.title,
        created_at=c.created_at.isoformat() if c.created_at else "",
        updated_at=c.updated_at.isoformat() if c.updated_at else "",
    )


@router.get("", response_model=ListConversationsPaginatedResponse)
async def list_conversations(
    limit: int = Query(20, ge=1, le=100, description="Number of conversations per page"),
    cursor: str | None = Query(None, description="Opaque cursor from previous response for next page"),
    use_case: ListConversationsUseCase = Depends(dependencies.get_list_conversations_use_case),
):
    """List conversations with cursor-based pagination."""
    conversations, next_cursor = await use_case.execute(limit=limit, cursor=cursor)
    return ListConversationsPaginatedResponse(
        items=[_to_conversation(c) for c in conversations],
        next_cursor=next_cursor,
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: ConversationId,
    request: UpdateConversationRequest,
    use_case: UpdateConversationUseCase = Depends(dependencies.get_update_conversation_use_case),
):
    """Update conversation title."""
    conversation = await use_case.execute(conversation_id=conversation_id, title=request.title)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _to_conversation(conversation)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: ConversationId,
    use_case: DeleteConversationUseCase = Depends(dependencies.get_delete_conversation_use_case),
):
    """Delete a conversation and all its messages."""
    deleted = await use_case.execute(conversation_id=conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/{conversation_id}/message", response_model=ListMessagesPaginatedResponse)
async def list_messages(
    conversation_id: ConversationId,
    limit: int = Query(20, ge=1, le=100, description="Number of messages per page"),
    cursor: str | None = Query(None, description="Opaque cursor from previous response for next page"),
    use_case: ListConversationMessagesUseCase = Depends(dependencies.get_list_conversation_messages_use_case),
):
    """List messages of a conversation with cursor-based pagination."""
    messages, next_cursor = await use_case.execute(
        conversation_id=conversation_id, limit=limit, cursor=cursor
    )
    return ListMessagesPaginatedResponse(
        items=[
            MessageResponse(
                id=msg.id or "",
                character=msg.character.value,
                body=asdict(msg.body) if not isinstance(msg.body, dict) else msg.body,
                context=UserContextSchema(
                    emotion=msg.context.emotion,
                    name=msg.context.name,
                    nim=msg.context.nim,
                ) if msg.context else None,
                created_at=msg.created_at.isoformat() if msg.created_at else "",
            )
            for msg in messages
        ],
        next_cursor=next_cursor,
    )


@router.delete("/{conversation_id}/message/{message_id}", status_code=204)
async def delete_message(
    conversation_id: ConversationId,
    message_id: MessageId,
    use_case: DeleteMessageUseCase = Depends(dependencies.get_delete_message_use_case),
):
    """Delete a message from a conversation."""
    deleted = await use_case.execute(conversation_id=conversation_id, message_id=message_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Message not found")
