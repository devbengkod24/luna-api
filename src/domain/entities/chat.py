"""Chat domain entities."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class UserRole(Enum):
    """User role - represents the role of the user in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"


class Character(str, Enum):
    RUSSELL = "russell"
    FREDRICKSEN = "fredricksen"


@dataclass
class UserContext:
    """User context - stores user state for conversation."""

    emotion: str | None = None
    name: str | None = None
    nim: str | None = None

@dataclass
class ChatSession:
    """Chat session - represents a chat session."""

    title: str
    created_at: datetime
    updated_at: datetime
    id: Optional[str] = None

@dataclass
class Message:
    """Message - represents a message in a chat session."""

    base64_audio: str | None = None
    character: str
    content: dict
    context: UserContext | None = None
    created_at: datetime
    id: Optional[str] = None
    session_id: str