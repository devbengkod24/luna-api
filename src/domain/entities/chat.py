"""Chat domain entities."""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class MessageEntity:
    """Message entity - represents a single message in a chat."""
    
    id: Optional[int]
    session_id: int
    role: str  # 'user' or 'assistant'
    content: str
    context: Optional[Dict[str, Any]]
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

