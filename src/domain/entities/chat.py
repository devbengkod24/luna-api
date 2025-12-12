"""Chat domain entities."""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

class UserRole(Enum):
    """User role - represents the role of the user in the conversation."""
    
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class UserContext:
    """User context - stores user state for conversation."""
    
    emotion: str = "neutral"
    name: Optional[str] = None
    nim: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "emotion": self.emotion,
            "name": self.name,
            "nim": self.nim,
        }

@dataclass
class MessageEntity:
    """Message entity - represents a single message in a chat."""
    
    id: Optional[int]
    session_id: int
    role: UserRole
    content: str
    context: Optional[UserContext]
    created_at: Optional[datetime]


@dataclass
class ChatSessionEntity:
    """Chat session entity - represents a conversation session."""
    
    id: Optional[int]
    uuid: str
    title: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    messages: List[MessageEntity]

