import base64
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Any

from src.config import settings
from src.domain.entities.chat import Character

class UserContextSchema(BaseModel):
    """User context schema."""
    emotion: str | None = Field(
        default=None,
        description="User emotion state (null if neutral/no emotion detected)",
        min_length=1,
        max_length=20,
    )
    name: str | None = Field(
        default=None,
        description="User name",
        min_length=1,
        max_length=20,
    )
    nim: str | None = Field(
        default=None,
        description="User NIM",
        min_length=14,
        max_length=14,
    )


class ChatTextRequest(BaseModel):
    """Request model for text chat with validation."""
    uuid: str | None = None
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=settings.chat_max_prompt_length,
        description="User prompt",
    )
    character: Character = Field(
        ...,
        description="Persona character to use for responses (russell|fredricksen)",
    )
    context: UserContextSchema | None = Field(
        default=None,
        description="User context (emotion, name, nim). Can be null if all fields are null.",
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and normalize prompt."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()


class ChatAudioRequest(BaseModel):
    """Request model for audio chat with validation."""
    uuid: str | None = None
    audio: str = Field(
        ...,
        min_length=1,
        max_length=settings.chat_max_audio_base64_size,
        description="Base64-encoded audio",
    )
    character: Character = Field(
        ...,
        description="Persona character to use for responses (russell|fredricksen)",
    )
    context: UserContextSchema | None = Field(
        default=None,
        description="User context (emotion, name, nim). Can be null if all fields are null.",
    )
    enable_tts: bool = Field(
        default=True,
        description="Enable text-to-speech synthesis for the response",
    )

    @field_validator('audio')
    @classmethod
    def validate_audio(cls, v: str) -> str:
        """Validate audio data is valid base64 and normalize."""
        if not v or not v.strip():
            raise ValueError("Audio data cannot be empty")
        
        audio_data = v.strip()
        
        try:
            base64.b64decode(audio_data, validate=True)
        except Exception:
            raise ValueError("Invalid base64 audio data")
        
        return audio_data
    
class ChatHistoryItem(BaseModel):
    """Response model for chat history item."""
    id: int
    uuid: str
    title: str
    created_at: str
    updated_at: str


class MessageItem(BaseModel):
    """Response model for message."""
    id: int
    role: Literal["user", "assistant"]
    content: str
    context: UserContextSchema | None = None
    created_at: str


class ChatSessionDetail(BaseModel):
    """Response model for chat session detail."""
    id: int
    uuid: str
    title: str
    messages: list[MessageItem]


class SSEEvent(BaseModel):
    """Event for SSE."""
    type: Literal["log", "token", "done", "error", "transcript", "audio", "tool_call", "tool_result"]
    content: str | None = None
    session_uuid: str | None = None
    finish_reason: str | None = None
    tool: str | None = None
    input: Any | None = None
    result: Any | None = None
    
    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json(exclude_none=True)}\n\n"