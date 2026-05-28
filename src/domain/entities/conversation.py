"""Conversation domain entities — used by conversation and message management."""

from dataclasses import dataclass
from datetime import datetime
from src.shared.constants import Character
from src.domain.entities.response import UserMessage, ToolMessage, AIMessage, UserContext


@dataclass
class Conversation:
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    id: str | None = None


@dataclass
class Message:
    character: Character
    body: UserMessage | ToolMessage | AIMessage
    conversation_id: str
    context: UserContext | None = None
    created_at: datetime | None = None
    id: str | None = None
