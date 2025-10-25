"""Context domain entities."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class UserContext:
    """User context - stores user state for conversation."""
    
    emotion: str
    name: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "emotion": self.emotion,
            "name": self.name,
        }

