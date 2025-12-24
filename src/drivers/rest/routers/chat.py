"""Chat REST routers."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Query
from fastapi.responses import StreamingResponse

from src.domain.entities.chat import UserContext
from src.drivers import dependencies
from src.drivers.rest.schemas.chat import (
    ChatAudioRequest,
    ChatHistoryItem,
    ChatSessionDetail,
    ChatTextRequest,
    MessageItem,
    SSEEvent,
)
from src.use_cases.chat_history import GetChatHistoryUseCase, GetChatSessionUseCase
from src.use_cases.chat_speech import ChatSpeechUseCase
from src.use_cases.chat_text import ChatTextUseCase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/chat", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def make_sse_stream(use_case, execute_args: tuple, logger_prefix: str):
    """Create SSE event generator from use case."""

    async def event_generator():
        try:
            event_count = 0
            async for event in use_case.execute(*execute_args):
                if not isinstance(event, dict) or "type" not in event:
                    logger.warning(f"{logger_prefix} invalid event: {event}")
                    continue
                try:
                    yield SSEEvent(**event).to_sse()
                    event_count += 1
                except Exception as e:
                    logger.error(f"{logger_prefix} serialize error: {e}", exc_info=True)
                    yield SSEEvent(
                        type="error", content="Error serializing event"
                    ).to_sse()
                    break
            logger.info(f"{logger_prefix} completed - {event_count} events sent")
        except ValueError as e:
            logger.warning(f"{logger_prefix} validation: {e}")
            yield SSEEvent(type="error", content=f"Validation error: {e}").to_sse()
            yield SSEEvent(type="done", finish_reason="error").to_sse()
        except Exception as e:
            logger.error(f"{logger_prefix} stream error: {e}", exc_info=True)
            yield SSEEvent(
                type="error", content="An error occurred while processing your request"
            ).to_sse()
            yield SSEEvent(type="done", finish_reason="error").to_sse()

    return event_generator


@router.post("/text")
async def chat_text(
    request: ChatTextRequest,
    use_case: ChatTextUseCase = Depends(dependencies.get_chat_text_use_case),
):
    """Handle text-based chat with streaming response."""
    context = UserContext(
        emotion=request.context.emotion,
        name=request.context.name,
        nim=request.context.nim,
        character=request.context.character,
    )

    logger.info(
        f"Chat text request - UUID: {request.uuid}, "
        f"Prompt length: {len(request.prompt)}, "
        f"Emotion: {context.emotion}"
    )

    event_generator = make_sse_stream(
        use_case, (request.uuid, request.prompt, context), "Chat text"
    )
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/audio")
async def chat_audio(
    request: ChatAudioRequest,
    use_case: ChatSpeechUseCase = Depends(dependencies.get_chat_speech_use_case),
):
    """Handle audio-based chat with streaming response."""
    context = UserContext(
        emotion=request.context.emotion,
        name=request.context.name,
        nim=request.context.nim,
        character=request.context.character,
    )

    logger.info(
        f"Chat audio request - UUID: {request.uuid}, "
        f"Audio size: {len(request.audio)} bytes, "
        f"Emotion: {context.emotion}"
    )

    event_generator = make_sse_stream(
        use_case, (request.uuid, request.audio, context), "Chat audio"
    )
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/history", response_model=list[ChatHistoryItem])
async def get_chat_history(
    limit: int | None = Query(
        None, ge=1, le=1000, description="Maximum number of sessions to return"
    ),
    offset: int | None = Query(0, ge=0, description="Number of sessions to skip"),
    use_case: GetChatHistoryUseCase = Depends(dependencies.get_history_use_case),
):
    """Get list of all chat sessions."""
    try:
        sessions = await use_case.execute()

        if limit is not None:
            total = len(sessions)
            sessions = sessions[offset : offset + limit]
            logger.info(
                f"Listed {len(sessions)} session(s) "
                f"(offset={offset}, limit={limit}, total={total})"
            )
        else:
            logger.info(f"Listed {len(sessions)} session(s)")

        return [
            ChatHistoryItem(
                id=session.id,
                uuid=session.uuid,
                title=session.title,
                created_at=session.created_at.isoformat() if session.created_at else "",
                updated_at=session.updated_at.isoformat() if session.updated_at else "",
            )
            for session in sessions
        ]

    except Exception as e:
        logger.error(f"Error getting chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/history/{session_uuid}", response_model=ChatSessionDetail)
async def get_chat_session(
    session_uuid: str,
    use_case: GetChatSessionUseCase = Depends(dependencies.get_session_use_case),
):
    """Get specific chat session with all messages."""
    try:
        session = await use_case.execute(session_uuid)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        logger.info(
            f"Retrieved session: {session_uuid} ({len(session.messages)} messages)"
        )

        return ChatSessionDetail(
            id=session.id,
            uuid=session.uuid,
            title=session.title,
            messages=[
                MessageItem(
                    role=msg.role,
                    content=msg.content,
                    context=msg.context,
                )
                for msg in session.messages
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
