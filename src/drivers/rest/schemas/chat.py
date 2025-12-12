from pydantic import BaseModel, Field
from typing import Optional, Literal

class UserContextSchema(BaseModel):
    """User context schema."""
    emotion: str = Field(default="neutral", description="User emotion state")
    name: Optional[str] = Field(default=None, description="User name")
    nim: Optional[str] = Field(default=None, description="User NIM")


class ChatTextRequest(BaseModel):
    """Request model for text chat."""
    uuid: Optional[str] = None
    prompt: str
    context: UserContextSchema


class ChatAudioRequest(BaseModel):
    """Request model for audio chat."""
    uuid: Optional[str] = None
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
    context: Optional[UserContextSchema] = None

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
    content: Optional[str] = None
    session_uuid: Optional[str] = None
    finish_reason: Optional[str] = None
    
    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json(exclude_none=True)}\n\n"