"""Conversation REST routers."""
import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Query

from src.drivers import dependencies
from src.drivers.rest.schemas import (
    ConversationResponse,
    ListConversationsPaginatedResponse,
    UpdateConversationRequest,
    MessageResponse,
    ListMessagesPaginatedResponse,
    UserContextSchema,
)
from src.use_cases import (
    ListConversationsUseCase,
    UpdateConversationUseCase,
    DeleteConversationUseCase,
    ListConversationMessagesUseCase,
    DeleteMessageUseCase,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.get("", response_model=ListConversationsPaginatedResponse)
async def list_conversations(
    limit: int = Query(20, ge=1, le=100, description="Number of conversations per page"),
    cursor: str | None = Query(None, description="Opaque cursor from previous response for next page"),
    use_case: ListConversationsUseCase = Depends(dependencies.get_list_conversations_use_case),
):
    """List conversations with cursor-based pagination."""
    try:
        conversations, next_cursor = await use_case.execute(limit=limit, cursor=cursor)
        return ListConversationsPaginatedResponse(
            items=[
                ConversationResponse(
                    id=c.id or "",
                    title=c.title,
                    created_at=c.created_at.isoformat() if c.created_at else "",
                    updated_at=c.updated_at.isoformat() if c.updated_at else "",
                )
                for c in conversations
            ],
            next_cursor=next_cursor,
        )
    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    use_case: UpdateConversationUseCase = Depends(dependencies.get_update_conversation_use_case),
):
    """Update conversation title."""
    try:
        conversation = await use_case.execute(conversation_id=conversation_id, title=request.title)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return ConversationResponse(
            id=conversation.id or "",
            title=conversation.title,
            created_at=conversation.created_at.isoformat() if conversation.created_at else "",
            updated_at=conversation.updated_at.isoformat() if conversation.updated_at else "",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    use_case: DeleteConversationUseCase = Depends(dependencies.get_delete_conversation_use_case),
):
    """Delete a conversation and all its messages."""
    try:
        deleted = await use_case.execute(conversation_id=conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{conversation_id}/message", response_model=ListMessagesPaginatedResponse)
async def list_messages(
    conversation_id: str,
    limit: int = Query(20, ge=1, le=100, description="Number of messages per page"),
    cursor: str | None = Query(None, description="Opaque cursor from previous response for next page"),
    use_case: ListConversationMessagesUseCase = Depends(dependencies.get_list_conversation_messages_use_case),
):
    """List messages of a conversation with cursor-based pagination."""
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{conversation_id}/message/{message_id}", status_code=204)
async def delete_message(
    conversation_id: str,
    message_id: str,
    use_case: DeleteMessageUseCase = Depends(dependencies.get_delete_message_use_case),
):
    """Delete a message from a conversation."""
    try:
        deleted = await use_case.execute(message_id=message_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Message not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
