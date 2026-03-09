"""Conversation domain entities — used by conversation and message management."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

from src.shared.constants import Character
from src.domain.entities.response import UserMessage, ToolMessage, AIMessage, UserContext


@dataclass
class Conversation:
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[str] = None


@dataclass
class Message:
    character: Character
    body: Union[UserMessage, ToolMessage, AIMessage]
    conversation_id: str
    context: UserContext | None = None
    created_at: datetime | None = None
    id: Optional[str] = None
