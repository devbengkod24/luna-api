"""Chat domain entities."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from src.shared.constants import Character


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

    character: Character
    content: dict
    session_id: str
    created_at: datetime
    base64_audio: str | None = None
    context: UserContext | None = None
    id: Optional[str] = None