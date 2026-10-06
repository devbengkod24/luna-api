"""Unit tests for the text response use case (streaming orchestrator)."""

import pytest

from src.domain.entities import (
    AIMessage,
    Conversation,
    ResponseMetadata,
    UsageMetadata,
)
from src.domain.events import (
    AssistantEvent,
    AudioEvent,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
)
from src.shared.constants import Character
from src.use_cases.response_use_case import ResponseTextUseCase

from .fakes import (
    RESPONSE_OPTIONS,
    FakeAgentService,
    FakeConversationRepo,
    FakeMessageRepo,
    FakePromptRepository,
    FakeSttService,
    FakeTtsService,
)


def _assistant_message(content: str) -> AIMessage:
    return AIMessage(
        content=content,
        additional_kwargs=[],
        response_metadata=ResponseMetadata(model="m", done_reason="stop"),
        usage_metadata=UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3),
    )


@pytest.mark.asyncio
async def test_response_text_emits_done_and_persists_messages():
    conv_repo = FakeConversationRepo()
    msg_repo = FakeMessageRepo()
    agent = FakeAgentService(
        events=[
            TokenEvent(content="hello"),
            AssistantEvent(message=_assistant_message("hello world")),
        ]
    )
    tts = FakeTtsService()

    use_case = ResponseTextUseCase(
        conversation_repo=conv_repo,
        message_repo=msg_repo,
        agent_service=agent,
        tts_service=tts,
        prompt_repo=FakePromptRepository(),
        options=RESPONSE_OPTIONS,
    )

    events = []
    async for ev in use_case.execute(
        conversation_id=None,
        user_input="hi",
        character=Character.LUNA,
        context=None,
        enable_tts=True,
    ):
        events.append(ev)

    # First message of new conversation
    types = [type(e).__name__ for e in events]
    assert "DoneEvent" in types
    assert any(isinstance(e, AssistantEvent) for e in events)
    # Audio synthesized because final assistant content was non-empty
    assert any(isinstance(e, AudioEvent) for e in events)

    # User + assistant messages persisted (no tool messages emitted)
    bodies = [m.body for m in msg_repo._items]
    assert len(bodies) == 2  # user + assistant


@pytest.mark.asyncio
async def test_response_text_emits_error_for_unknown_conversation():
    use_case = ResponseTextUseCase(
        conversation_repo=FakeConversationRepo(),
        message_repo=FakeMessageRepo(),
        agent_service=FakeAgentService(events=[]),
        tts_service=FakeTtsService(),
        prompt_repo=FakePromptRepository(),
        options=RESPONSE_OPTIONS,
    )

    events = []
    async for ev in use_case.execute(
        conversation_id="not-found",
        user_input="hi",
        character=Character.LUNA,
        context=None,
        enable_tts=False,
    ):
        events.append(ev)

    assert any(isinstance(e, ErrorEvent) for e in events)
    assert not any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.asyncio
async def test_response_text_passes_domain_messages_to_agent():
    conv_repo = FakeConversationRepo()
    conv = await conv_repo.create(Conversation(title="seed"))

    agent = FakeAgentService(events=[AssistantEvent(message=_assistant_message(""))])
    use_case = ResponseTextUseCase(
        conversation_repo=conv_repo,
        message_repo=FakeMessageRepo(),
        agent_service=agent,
        tts_service=FakeTtsService(),
        prompt_repo=FakePromptRepository(),
        options=RESPONSE_OPTIONS,
    )

    async for _ in use_case.execute(
        conversation_id=conv.id,
        user_input="hi",
        character=Character.LUNA,
        context=None,
        enable_tts=False,
    ):
        pass

    # Agent receives a list of domain Message objects (not dicts)
    assert isinstance(agent.received_messages, list)
    for m in agent.received_messages:
        assert hasattr(m, "body")


# --- speech path and layering -------------------------------------------------

class _FailingStt(FakeSttService):
    def __init__(self, error=None, transcript=""):
        self.error = error
        self.transcript = transcript

    async def transcribe(self, speech_base64: str) -> str:
        if self.error is not None:
            raise self.error
        return self.transcript


async def _speech_events(stt):
    from src.use_cases.response_use_case import ResponseSpeechUseCase

    use_case = ResponseSpeechUseCase(
        conversation_repo=FakeConversationRepo(),
        message_repo=FakeMessageRepo(),
        agent_service=FakeAgentService(events=[]),
        tts_service=None,
        stt_service=stt,
        prompt_repo=FakePromptRepository(),
        options=RESPONSE_OPTIONS,
    )
    return [e async for e in use_case.execute(None, "AAAA", Character.LUNA, None, enable_tts=False)]


@pytest.mark.asyncio
async def test_a_failed_transcription_ends_the_stream_with_an_error_event():
    # It used to raise out of the SSE stream, which cut the response off with no
    # event: the page waited for a reply that had already failed.
    for stt in (_FailingStt(error=RuntimeError("down")), _FailingStt(error=ValueError("bad")), _FailingStt()):
        events = await _speech_events(stt)
        assert isinstance(events[-1], ErrorEvent), [type(e).__name__ for e in events]
        assert not any(isinstance(e, DoneEvent) for e in events)


def test_use_cases_take_their_settings_from_the_composition_root():
    from pathlib import Path

    import ast

    for path in Path("src/use_cases").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [node.module or ""] if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            )
            assert not any(n == "src.config" or n.startswith("src.config.") for n in names), (
                f"{path} imports configuration directly"
            )
