from pydantic import BaseModel, Field
from typing import Literal

class UserContextSchema(BaseModel):
    """User context schema."""
    emotion: str = Field(default="neutral", description="User emotion state")
    name: str | None = Field(default=None, description="User name")
    nim: str | None = Field(default=None, description="User NIM")


class ChatTextRequest(BaseModel):
    """Request model for text chat."""
    uuid: str | None = None
    prompt: str
    context: UserContextSchema


class ChatAudioRequest(BaseModel):
    """Request model for audio chat."""
    uuid: str | None = None
    audio: str  # Base64-encoded audio
    context: UserContextSchema


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
    context: UserContextSchema | None = None

# chat/history Response Schema
class ChatSessionDetail(BaseModel):
    """Response model for chat session detail."""
    id: int
    uuid: str
    title: str
    messages: list[MessageItem]

class SSEEvent(BaseModel):
    """Event for SSE."""
    type: Literal["log", "token", "done", "error", "transcript"]
    content: str | None = None
    session_uuid: str | None = None
    finish_reason: str | None = None
    
    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json(exclude_none=True)}\n\n"