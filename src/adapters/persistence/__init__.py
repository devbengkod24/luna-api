"""Persistence adapters (database, repositories, models).

This package is organized by *function* (persistence) instead of vendor/framework.
Implementations currently live in `src.adapters.postgres` and are re-exported here.
"""

from src.adapters.postgres.connection import DatabaseConnection
from src.adapters.postgres.chat_repository import ChatRepository
from src.adapters.postgres.models import Base, ChatSession, Message

__all__ = [
    "DatabaseConnection",
    "ChatRepository",
    "Base",
    "ChatSession",
    "Message",
]

