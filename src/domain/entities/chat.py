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

    emotion: str = "neutral"
    name: str | None = None
    nim: str | None = None
    character: Character = Character.RUSSELL

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "emotion": self.emotion,
            "name": self.name,
            "nim": self.nim,
            "character": self.character.value,
        }


@dataclass
class MessageEntity:
    """Message entity - represents a single message in a chat."""

    session_id: int
    role: UserRole
    content: str
    id: int | None = None
    context: UserContext | None = None
    created_at: datetime | None = None


@dataclass
class ChatSessionEntity:
    """Chat session entity - represents a conversation session."""

    uuid: str
    title: str
    messages: list[MessageEntity]
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
