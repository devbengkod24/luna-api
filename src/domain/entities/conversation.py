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
    content: Union[UserMessage, ToolMessage, AIMessage]
    conversation_id: str
    created_at: datetime
    context: UserContext | None = None
    id: Optional[str] = None
