"""Chat REST routers."""
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.params import Query
from typing import Optional
from src.use_cases.chat_text import ChatTextUseCase
from src.use_cases.chat_speech import ChatSpeechUseCase
from src.use_cases.chat_history import GetChatHistoryUseCase, GetChatSessionUseCase
from src.domain.entities.chat import UserContext
from src.drivers import dependencies
from src.drivers.rest.schemas.chat import (
    ChatTextRequest, 
    ChatAudioRequest, 
    ChatHistoryItem, 
    ChatSessionDetail, 
    MessageItem,
    SSEEvent,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/chat", tags=["chat"])

MAX_PROMPT_LENGTH = 10000
MAX_AUDIO_BASE64_SIZE = 10 * 1024 * 1024  # 10MB

def validate_base64(data: str) -> bool:
    """Validate base64 string."""
    import base64
    try:
        if not data or not isinstance(data, str):
            return False
        if len(data) > MAX_AUDIO_BASE64_SIZE:
            return False
        base64.b64decode(data, validate=True)
        return True
    except Exception:
        return False

@router.post("/text")
async def chat_text(
    request: ChatTextRequest,
    use_case: ChatTextUseCase = Depends(dependencies.get_chat_text_use_case),
):
    """Handle text-based chat with streaming response.
    
    Returns Server-Sent Events (SSE) stream.
    
    Response format:
    - Log events: {"type": "log", "content": "..."}
    - Token events: {"type": "token", "content": "..."}
    - Done events: {"type": "done", "finish_reason": "stop"}
    - Error events: {"type": "error", "content": "..."}
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty"
        )
    
    if len(request.prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long. Maximum {MAX_PROMPT_LENGTH} characters"
        )
    
    context = UserContext(
        emotion=request.context.emotion.lower(),
        name=request.context.name,
        nim=request.context.nim,
    )

    logger.info(
        f"Chat text request - UUID: {request.uuid}, "
        f"Prompt length: {len(request.prompt)}, "
        f"Emotion: {context.emotion}"
    )
    
    async def event_generator():
        """Generate SSE events."""
        try:
            event_count = 0
            async for event in use_case.execute(request.uuid, request.prompt.strip(), context):
                # Validate event structure
                if not isinstance(event, dict) or "type" not in event:
                    logger.warning(f"Invalid event structure: {event}")
                    continue
                
                try:
                    yield SSEEvent(**event).to_sse()
                    event_count += 1
                except Exception as e:
                    logger.error(f"Error serializing SSE event: {e}", exc_info=True)
                    yield SSEEvent(
                        type="error",
                        content="Error serializing event"
                    ).to_sse()
                    break
            
            logger.info(f"Chat text completed - {event_count} events sent")
                    
        except ValueError as e:
            logger.warning(f"Validation error in chat: {e}")
            yield SSEEvent(type="error", content=f"Validation error: {str(e)}").to_sse()
            yield SSEEvent(type="done", finish_reason="error").to_sse()
        except Exception as e:
            logger.error(f"Error in chat text stream: {e}", exc_info=True)
            yield SSEEvent(
                type="error",
                content="An error occurred while processing your request"
            ).to_sse()
            yield SSEEvent(type="done", finish_reason="error").to_sse()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/audio")
async def chat_audio(
    request: ChatAudioRequest,
    use_case: ChatSpeechUseCase = Depends(dependencies.get_chat_speech_use_case),
):
    """Handle audio-based chat with streaming response.
    
    Returns Server-Sent Events (SSE) stream in OpenAI-compatible format.
    
    Response format:
    - Log events: {"type": "log", "content": "..."}
    - Transcript events: {"type": "transcript", "content": "..."}
    - Token events: {"type": "token", "content": "..."}
    - Done events: {"type": "done", "finish_reason": "stop"}
    - Error events: {"type": "error", "content": "..."}
    """
    if not request.audio or not request.audio.strip():
        raise HTTPException(
            status_code=400,
            detail="Audio data cannot be empty"
        )
    
    if not validate_base64(request.audio):
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 audio data or audio too large"
        )

    context = UserContext(
        emotion=request.context.emotion,
        name=request.context.name,
        nim=request.context.nim,
    )

    logger.info(
        f"Chat audio request - UUID: {request.uuid}, "
        f"Audio size: {len(request.audio)} bytes, "
        f"Emotion: {context.emotion}"
    )
    
    async def event_generator():
        """Generate SSE events."""
        try:
            event_count = 0
            async for event in use_case.execute(request.uuid, request.audio, context):
                # Validate event structure
                if not isinstance(event, dict) or "type" not in event:
                    logger.warning(f"Invalid event structure: {event}")
                    continue
                
                try:
                    yield SSEEvent(**event).to_sse()
                    event_count += 1
                except Exception as e:
                    logger.error(f"Error serializing SSE event: {e}", exc_info=True)
                    yield SSEEvent(
                        type="error",
                        content="Error serializing event"
                    ).to_sse()
                    break
            
            logger.info(f"Chat audio completed - {event_count} events sent")
                    
        except ValueError as e:
            logger.warning(f"Validation error in chat audio: {e}")
            yield SSEEvent(type="error", content=f"Validation error: {str(e)}").to_sse()
            yield SSEEvent(type="done", finish_reason="error").to_sse()
        except Exception as e:
            logger.error(f"Error in chat audio stream: {e}", exc_info=True)
            yield SSEEvent(
                type="error",
                content="An error occurred while processing your request"
            ).to_sse()
            yield SSEEvent(type="done", finish_reason="error").to_sse()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/history", response_model=list[ChatHistoryItem])
async def get_chat_history(
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum number of sessions to return"),
    offset: Optional[int] = Query(0, ge=0, description="Number of sessions to skip"),
    use_case: GetChatHistoryUseCase = Depends(dependencies.get_history_use_case),
):
    """Get list of all chat sessions."""
    try:
        sessions = await use_case.execute()
        
        # Apply pagination if requested
        if limit is not None:
            total = len(sessions)
            sessions = sessions[offset:offset + limit]
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
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


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
        
        logger.info(f"Retrieved session: {session_uuid} ({len(session.messages)} messages)")
        
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
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

