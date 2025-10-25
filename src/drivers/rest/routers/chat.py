"""Chat REST routers."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.database.connection import DatabaseConnection
from src.adapters.database.repository import ChatRepository
from src.adapters.llm.llm_adapter import LLMAdapter
from src.adapters.qdrant.rag_adapter import QdrantRAGAdapter
from src.adapters.audio.stt_adapter import STTAdapter
from src.use_cases.chat_text import ChatTextUseCase
from src.use_cases.chat_audio import ChatAudioUseCase
from src.use_cases.chat_history import GetChatHistoryUseCase, GetChatSessionUseCase
from src.domain.entities.context import UserContext

router = APIRouter(prefix="/v1/chat", tags=["chat"])


# Request/Response models
class ChatTextRequest(BaseModel):
    """Request model for text chat."""
    uuid: Optional[str] = None
    prompt: str
    context: dict  # {"emotion": "happy", "name": "User_ABC"}


class ChatAudioRequest(BaseModel):
    """Request model for audio chat."""
    uuid: Optional[str] = None
    audio: str  # Base64-encoded audio
    context: dict  # {"emotion": "happy", "name": "User_ABC"}


class ChatHistoryItem(BaseModel):
    """Response model for chat history item."""
    id: int
    uuid: str
    title: str
    created_at: str
    updated_at: str


class MessageItem(BaseModel):
    """Response model for message."""
    role: str
    content: str
    context: Optional[dict] = None


class ChatSessionDetail(BaseModel):
    """Response model for chat session detail."""
    id: int
    uuid: str
    title: str
    messages: list[MessageItem]


# Dependency injection (will be properly configured in main.py)
async def get_chat_text_use_case() -> ChatTextUseCase:
    """Get chat text use case with dependencies."""
    # This is a placeholder - actual DI will be set up in main.py
    from src.drivers.rest.dependencies import get_db_session, get_llm_adapter, get_rag_adapter
    
    # These will be injected properly
    raise NotImplementedError("Use dependency injection from main.py")


async def get_chat_audio_use_case() -> ChatAudioUseCase:
    """Get chat audio use case with dependencies."""
    raise NotImplementedError("Use dependency injection from main.py")


async def get_history_use_case() -> GetChatHistoryUseCase:
    """Get chat history use case with dependencies."""
    raise NotImplementedError("Use dependency injection from main.py")


async def get_session_use_case() -> GetChatSessionUseCase:
    """Get chat session use case with dependencies."""
    raise NotImplementedError("Use dependency injection from main.py")


# Routes
@router.post("/text")
async def chat_text(
    request: ChatTextRequest,
    use_case: ChatTextUseCase = Depends(get_chat_text_use_case),
):
    """Handle text-based chat with streaming response.
    
    Returns Server-Sent Events (SSE) stream.
    """
    context = UserContext(
        emotion=request.context.get("emotion", "neutral"),
        name=request.context.get("name"),
    )
    
    async def event_generator():
        """Generate SSE events."""
        try:
            async for event in use_case.execute(request.uuid, request.prompt, context):
                # Format as SSE
                yield f"data: {event}\n\n"
        except Exception as e:
            yield f"data: {{'type': 'error', 'content': '{str(e)}'}}\n\n"
    
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
    use_case: ChatAudioUseCase = Depends(get_chat_audio_use_case),
):
    """Handle audio-based chat with streaming response.
    
    Returns Server-Sent Events (SSE) stream.
    """
    context = UserContext(
        emotion=request.context.get("emotion", "neutral"),
        name=request.context.get("name"),
    )
    
    async def event_generator():
        """Generate SSE events."""
        try:
            async for event in use_case.execute(request.uuid, request.audio, context):
                # Format as SSE
                yield f"data: {event}\n\n"
        except Exception as e:
            yield f"data: {{'type': 'error', 'content': '{str(e)}'}}\n\n"
    
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
    use_case: GetChatHistoryUseCase = Depends(get_history_use_case),
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
    use_case: GetChatSessionUseCase = Depends(get_session_use_case),
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

