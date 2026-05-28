"""Repository port interfaces (persistence layer contracts)."""

from abc import ABC, abstractmethod

from src.domain.entities import Conversation, Message, File


class IConversationRepository(ABC):
    """Interface for conversation repository."""

    @abstractmethod
    async def create(self, conversation: Conversation) -> Conversation:
        """Create a new conversation."""
        ...

    @abstractmethod
    async def get_by_id(self, id: str) -> Conversation | None:
        """Get conversation by UUID."""
        ...

    @abstractmethod
    async def list_cursor(
        self,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[list[Conversation], str | None]:
        """List conversations with cursor-based pagination. Returns (conversations, next_cursor)."""
        ...

    @abstractmethod
    async def update_title(self, id: str, title: str) -> bool:
        """Update conversation title."""
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a conversation by id."""
        ...


class IMessageRepository(ABC):
    """Interface for message repository."""

    @abstractmethod
    async def create(self, message: Message) -> Message:
        """Create a new message."""
        ...

    @abstractmethod
    async def get_by_id(self, id: str) -> Message | None:
        """Get message by UUID."""
        ...

    @abstractmethod
    async def get_by_conversation_id(self, conversation_id: str) -> list[Message]:
        """Get all messages for a conversation, ordered by created_at."""
        ...

    @abstractmethod
    async def get_recent_by_conversation_id(
        self, conversation_id: str, limit: int
    ) -> list[Message]:
        """Get the most recent messages for a conversation, ordered by created_at."""
        ...

    @abstractmethod
    async def get_by_conversation_id_cursor(
        self,
        conversation_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[list[Message], str | None]:
        """Get messages for a conversation with cursor-based pagination. Returns (messages, next_cursor)."""
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a message by id."""
        ...


class IFileRepository(ABC):
    """File repository interface."""

    @abstractmethod
    async def create(self, file: File) -> File:
        ...

    @abstractmethod
    async def get_by_id(self, id: str) -> File | None:
        ...

    @abstractmethod
    async def list(self) -> list[File]:
        ...

    @abstractmethod
    async def update(self, file: File) -> File:
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        ...
