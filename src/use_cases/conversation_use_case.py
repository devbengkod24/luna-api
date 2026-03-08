"""Conversation use cases — CRUD for conversations and message list/delete."""

from typing import Optional

from src.domain.entities import Conversation, Message
from src.ports.repositories import IConversationRepository, IMessageRepository


class ListConversationsUseCase:
    def __init__(self, conversation_repo: IConversationRepository):
        self.conversation_repo = conversation_repo

    async def execute(
        self,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> tuple[list[Conversation], Optional[str]]:
        return await self.conversation_repo.list_cursor(limit=limit, cursor=cursor)


class UpdateConversationUseCase:
    def __init__(self, conversation_repo: IConversationRepository):
        self.conversation_repo = conversation_repo

    async def execute(self, conversation_id: str, title: str) -> Optional[Conversation]:
        updated = await self.conversation_repo.update_title(id=conversation_id, title=title)
        if not updated:
            return None
        return await self.conversation_repo.get_by_id(conversation_id)


class DeleteConversationUseCase:
    def __init__(self, conversation_repo: IConversationRepository):
        self.conversation_repo = conversation_repo

    async def execute(self, conversation_id: str) -> bool:
        return await self.conversation_repo.delete(id=conversation_id)


class ListConversationMessagesUseCase:
    def __init__(self, message_repo: IMessageRepository):
        self.message_repo = message_repo

    async def execute(
        self,
        conversation_id: str,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> tuple[list[Message], Optional[str]]:
        return await self.message_repo.get_by_conversation_id_cursor(
            conversation_id=conversation_id,
            limit=limit,
            cursor=cursor,
        )


class DeleteMessageUseCase:
    def __init__(self, message_repo: IMessageRepository):
        self.message_repo = message_repo

    async def execute(self, message_id: str) -> bool:
        return await self.message_repo.delete(id=message_id)
