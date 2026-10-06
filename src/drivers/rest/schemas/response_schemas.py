import base64
from pydantic import BaseModel, Field, field_validator

from src.config import settings
from src.shared.constants import UUID_PATTERN, Character


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
        max_length=100,
    )
    nim: str | None = Field(
        default=None,
        description="User NIM",
        min_length=1,
        max_length=20,
    )


class ResponseTextRequest(BaseModel):
    """Request model for text-based response."""

    conversation_id: str | None = Field(default=None, pattern=UUID_PATTERN)
    user_input: str = Field(
        ...,
        min_length=1,
        max_length=settings.chat_max_prompt_length,
        description="User input",
    )
    character: Character = Field(
        ...,
        description="Persona character code (0 = LUNA, 1 = VENDRA reserved)",
    )
    context: UserContextSchema | None = Field(
        default=None,
        description="User context (emotion, name, nim). Can be null if all fields are null.",
    )
    enable_tts: bool = Field(
        default=True,
        description="Enable text-to-speech synthesis for the response",
    )

    @field_validator("user_input")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("User input cannot be empty")
        return v.strip()


class ResponseAudioRequest(BaseModel):
    """Request model for audio-based response."""

    conversation_id: str | None = Field(default=None, pattern=UUID_PATTERN)
    audio_input: str = Field(
        ...,
        min_length=1,
        max_length=settings.chat_max_audio_base64_size,
        description="Base64-encoded audio",
    )
    character: Character = Field(
        ...,
        description="Persona character to use for responses (0 = LUNA, 1 = VENDRA reserved)",
    )
    context: UserContextSchema | None = Field(
        default=None,
        description="User context (emotion, name, nim). Can be null if all fields are null.",
    )
    enable_tts: bool = Field(
        default=True,
        description="Enable text-to-speech synthesis for the response",
    )

    @field_validator("audio_input")
    @classmethod
    def validate_audio_input(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Audio input cannot be empty")
        audio_input = v.strip()
        try:
            base64.b64decode(audio_input, validate=True)
        except Exception:
            raise ValueError("Invalid base64 audio input")
        return audio_input
