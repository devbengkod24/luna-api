"""Unit tests for conversation CRUD use cases."""

import pytest

from src.domain.entities import Conversation, Message, UserMessage
from src.shared.constants import Character
from src.use_cases.conversation_use_case import (
    DeleteConversationUseCase,
    ListConversationMessagesUseCase,
    ListConversationsUseCase,
    UpdateConversationUseCase,
)

from .fakes import FakeConversationRepo, FakeMessageRepo


@pytest.mark.asyncio
async def test_list_conversations_returns_stored_items():
    repo = FakeConversationRepo()
    await repo.create(Conversation(title="One"))
    await repo.create(Conversation(title="Two"))

    use_case = ListConversationsUseCase(repo)
    items, cursor = await use_case.execute(limit=10)

    assert {c.title for c in items} == {"One", "Two"}
    assert cursor is None


@pytest.mark.asyncio
async def test_update_conversation_returns_updated_entity():
    repo = FakeConversationRepo()
    created = await repo.create(Conversation(title="Old"))

    use_case = UpdateConversationUseCase(repo)
    updated = await use_case.execute(conversation_id=created.id, title="New")

    assert updated is not None
    assert updated.title == "New"


@pytest.mark.asyncio
async def test_update_conversation_returns_none_when_missing():
    repo = FakeConversationRepo()
    use_case = UpdateConversationUseCase(repo)

    result = await use_case.execute(conversation_id="missing", title="x")
    assert result is None


@pytest.mark.asyncio
async def test_delete_conversation_returns_true_when_deleted():
    repo = FakeConversationRepo()
    created = await repo.create(Conversation(title="x"))

    use_case = DeleteConversationUseCase(repo)
    assert await use_case.execute(conversation_id=created.id) is True
    assert await repo.get_by_id(created.id) is None


@pytest.mark.asyncio
async def test_list_messages_filters_by_conversation_id():
    repo = FakeMessageRepo()
    await repo.create(
        Message(
            conversation_id="conv-a",
            character=Character.LUNA,
            body=UserMessage(content="hi"),
        )
    )
    await repo.create(
        Message(
            conversation_id="conv-b",
            character=Character.LUNA,
            body=UserMessage(content="bye"),
        )
    )

    use_case = ListConversationMessagesUseCase(repo)
    items, _ = await use_case.execute(conversation_id="conv-a", limit=10)

    assert [m.body.content for m in items] == ["hi"]
