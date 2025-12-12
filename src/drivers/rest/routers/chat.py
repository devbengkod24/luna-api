"""Chat REST routers."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

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

router = APIRouter(prefix="/v1/chat", tags=["chat"])

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
    context = UserContext(
        emotion=request.context.emotion,
        name=request.context.name,
        nim=request.context.nim,
    )
    
    async def event_generator():
        """Generate SSE events."""
        try:
            async for event in use_case.execute(request.uuid, request.prompt, context):
                yield SSEEvent(**event).to_sse()
                    
        except Exception as e:
            yield SSEEvent(type="error", content=f"Error: {str(e)}").to_sse()
            yield SSEEvent(type="done", finish_reason="stop").to_sse()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
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
    context = UserContext(
        emotion=request.context.emotion,
        name=request.context.name,
        nim=request.context.nim,
    )
    
    async def event_generator():
        """Generate SSE events."""
        try:
            async for event in use_case.execute(request.uuid, request.audio, context):
                yield SSEEvent(**event).to_sse()
                    
        except Exception as e:
            yield SSEEvent(type="error", content=f"Error: {str(e)}").to_sse()
            yield SSEEvent(type="done", finish_reason="stop").to_sse()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history", response_model=list[ChatHistoryItem])
async def get_chat_history(
    use_case: GetChatHistoryUseCase = Depends(dependencies.get_history_use_case),
):
    """Get list of all chat sessions."""
    sessions = await use_case.execute()
    
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


@router.get("/history/{session_uuid}", response_model=ChatSessionDetail)
async def get_chat_session(
    session_uuid: str,
    use_case: GetChatSessionUseCase = Depends(dependencies.get_session_use_case),
):
    """Get specific chat session with all messages."""
    session = await use_case.execute(session_uuid)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
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

