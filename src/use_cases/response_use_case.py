"""Response use cases — text and speech AI response with streaming."""

import asyncio
import logging
import re
from typing import AsyncGenerator, Awaitable, Callable

from src.config import settings
from src.domain.entities import (
    Conversation,
    Message,
    UserContext,
    UserMessage,
)
from src.domain.events import (
    AssistantEvent,
    AudioChunkEvent,
    AudioEvent,
    DoneEvent,
    ErrorEvent,
    LogEvent,
    StreamEvent,
    TokenEvent,
    ToolResultEvent,
    TranscriptEvent,
)
from src.ports.prompts import IPromptRepository
from src.ports.repositories import IConversationRepository, IMessageRepository
from src.ports.services import IAgentService, ISTTService, ITTSService
from src.shared.constants import Character

logger = logging.getLogger(__name__)

# A title runner generates+persists a conversation title using its OWN db
# session, detached from the request. Injected by the driver/DI layer so this
# use case stays free of any session/infra knowledge.
TitleRunner = Callable[[str, str], Awaitable[None]]

# Hold references to fire-and-forget tasks so the event loop doesn't garbage-
# collect them mid-flight; they self-remove on completion.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro: Awaitable[None]) -> None:
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


class GenerateTitleUseCase:
    """Generate and persist a title for a brand-new conversation, asynchronously."""

    def __init__(
        self,
        conversation_repo: IConversationRepository,
        agent_service: IAgentService,
        prompt_repo: IPromptRepository,
    ):
        self.conversation_repo = conversation_repo
        self.agent_service = agent_service
        self.prompt_repo = prompt_repo

    async def execute(self, conversation_id: str, user_input: str) -> None:
        try:
            title_prompt = self.prompt_repo.build_title_prompt(user_input)
            title = await self.agent_service.generate_title(title_prompt)
            if title and title.strip():
                await self.conversation_repo.update_title(conversation_id, title.strip())
        except Exception:
            logger.exception(
                "Failed to generate/update conversation title (conversation_id=%s)",
                conversation_id,
            )


class SynthesizeTtsUseCase:
    """Preprocess assistant text and synthesize speech via TTS."""

    def __init__(
        self,
        agent_service: IAgentService,
        tts_service: ITTSService,
        prompt_repo: IPromptRepository,
    ):
        self.agent_service = agent_service
        self.tts_service = tts_service
        self.prompt_repo = prompt_repo

    async def stream_chunk(
        self, text: str, character: Character
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield AudioChunkEvent(s) for streaming adapters, or one AudioEvent.

        When the TTS service supports streaming (e.g. Cartesia), yields multiple
        AudioChunkEvent objects as PCM16 bytes arrive. For non-streaming adapters
        (MMS, Chatterbox, F5) falls back to a single AudioEvent with the full WAV.
        """
        import base64 as _b64

        if not text or not text.strip():
            return

        if self.tts_service.supports_streaming:
            async for pcm in self.tts_service.synthesize_stream(text, character):
                yield AudioChunkEvent(
                    content=_b64.b64encode(pcm).decode(),
                    sample_rate=getattr(self.tts_service, "sample_rate", 22050),
                )
        else:
            audio_b64 = await self.tts_service.synthesize(text, character)
            if audio_b64:
                yield AudioEvent(content=audio_b64)


class ResponseTextUseCase:
    """Use case for text-based AI response with streaming."""

    def __init__(
        self,
        conversation_repo: IConversationRepository,
        message_repo: IMessageRepository,
        agent_service: IAgentService,
        tts_service: ITTSService,
        prompt_repo: IPromptRepository,
        title_runner: TitleRunner | None = None,
    ):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.agent_service = agent_service
        self.prompt_repo = prompt_repo
        self.title_runner = title_runner
        self.title_use_case = GenerateTitleUseCase(
            conversation_repo, agent_service, prompt_repo
        )
        self.tts_use_case = SynthesizeTtsUseCase(agent_service, tts_service, prompt_repo)

    async def execute(
        self,
        conversation_id: str | None,
        user_input: str,
        character: Character,
        context: UserContext | None,
        enable_tts: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        async for event in _stream_conversation(
            conversation_repo=self.conversation_repo,
            message_repo=self.message_repo,
            agent_service=self.agent_service,
            tts_use_case=self.tts_use_case,
            title_use_case=self.title_use_case,
            title_runner=self.title_runner,
            prompt_repo=self.prompt_repo,
            conversation_id=conversation_id,
            user_input=user_input,
            character=character,
            context=context,
            enable_tts=enable_tts,
        ):
            yield event


class ResponseSpeechUseCase:
    """Use case for speech-based AI response with streaming."""

    def __init__(
        self,
        conversation_repo: IConversationRepository,
        message_repo: IMessageRepository,
        agent_service: IAgentService,
        tts_service: ITTSService,
        stt_service: ISTTService,
        prompt_repo: IPromptRepository,
        title_runner: TitleRunner | None = None,
    ):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.agent_service = agent_service
        self.stt_service = stt_service
        self.prompt_repo = prompt_repo
        self.title_runner = title_runner
        self.title_use_case = GenerateTitleUseCase(
            conversation_repo, agent_service, prompt_repo
        )
        self.tts_use_case = SynthesizeTtsUseCase(agent_service, tts_service, prompt_repo)

    async def execute(
        self,
        conversation_id: str | None,
        speech_base64: str,
        character: Character,
        context: UserContext | None,
        enable_tts: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        yield LogEvent(content="Mentranskripsi speech...")
        transcript = await self.stt_service.transcribe(speech_base64)
        yield TranscriptEvent(content=transcript)

        async for event in _stream_conversation(
            conversation_repo=self.conversation_repo,
            message_repo=self.message_repo,
            agent_service=self.agent_service,
            tts_use_case=self.tts_use_case,
            title_use_case=self.title_use_case,
            title_runner=self.title_runner,
            prompt_repo=self.prompt_repo,
            conversation_id=conversation_id,
            user_input=transcript,
            character=character,
            context=context,
            enable_tts=enable_tts,
        ):
            yield event


# Sentence-boundary matcher: a run of non-terminators ending in ., !, ?, or \n.
_SENTENCE_RE = re.compile(r"[^.!?\n]*[.!?\n]+")
# Force-flush a terminator-free buffer this long so a monologue still starts
# speaking instead of stalling.
_TTS_FORCE_FLUSH_CHARS = 220


def _drain_sentences(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences out of an accumulating token buffer.

    Returns (complete_sentences, remainder). The remainder is the trailing
    partial sentence still being streamed; the caller keeps accumulating into it.
    """
    sentences: list[str] = []
    cut = 0
    for match in _SENTENCE_RE.finditer(buffer):
        chunk = match.group(0).strip()
        if chunk:
            sentences.append(chunk)
        cut = match.end()
    remainder = buffer[cut:]
    if len(remainder) >= _TTS_FORCE_FLUSH_CHARS:
        tail = remainder.strip()
        if tail:
            sentences.append(tail)
        remainder = ""
    return sentences, remainder


async def _stream_conversation(
    *,
    conversation_repo: IConversationRepository,
    message_repo: IMessageRepository,
    agent_service: IAgentService,
    tts_use_case: SynthesizeTtsUseCase,
    title_use_case: GenerateTitleUseCase,
    title_runner: TitleRunner | None,
    prompt_repo: IPromptRepository,
    conversation_id: str | None,
    user_input: str,
    character: Character,
    context: UserContext | None,
    enable_tts: bool,
) -> AsyncGenerator[StreamEvent, None]:
    """Core streaming logic shared by text and speech use cases.

    The LLM token stream and TTS synthesis run concurrently: as the agent emits
    tokens, complete sentences are handed to a TTS worker that synthesizes them
    one at a time (preserving order) while the model keeps generating. Audio for
    the first sentence therefore starts playing well before the full reply is
    written — the dominant latency win over synthesizing the whole reply at the
    end. Both the voice WS and chat SSE paths flow through here.
    """
    conversation = await _resolve_conversation(conversation_repo, conversation_id)
    if conversation is None:
        yield ErrorEvent(content="Conversation not found")
        return

    if conversation_id is None:
        yield LogEvent(content="Membuat sesi baru...")
        yield LogEvent(content="Sesi dibuat: Obrolan Baru")
        # Title generation runs detached. The injected runner uses its OWN db
        # session; sharing the request session races with the persistence below.
        if title_runner is not None:
            _spawn_background(title_runner(conversation.id, user_input))
        else:
            _spawn_background(title_use_case.execute(conversation.id, user_input))

    system_prompt = prompt_repo.get_system_prompt(character, context)

    yield LogEvent(content="Menganalisis konteks...")
    history = await message_repo.get_recent_by_conversation_id(
        conversation.id, settings.chat_history_limit
    )

    # Persist current user message AFTER loading history to avoid duplicating
    # user_input in the LLM context (agent already receives user_input separately).
    await message_repo.create(
        Message(
            conversation_id=conversation.id,
            character=character,
            body=UserMessage(content=user_input),
            context=context,
        )
    )

    # LLM producer and TTS worker both feed one ordered output queue, each
    # posting a sentinel when done; the consumer drains until all sentinels seen.
    out_q: asyncio.Queue = asyncio.Queue()
    sentence_q: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    async def _tts_worker() -> None:
        try:
            while True:
                text = await sentence_q.get()
                if text is None:  # producer finished feeding sentences
                    return
                try:
                    async for event in tts_use_case.stream_chunk(text, character):
                        await out_q.put(event)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(
                        "Chunk TTS failed (conversation_id=%s): %s", conversation.id, e
                    )
                    continue
        finally:
            await out_q.put(sentinel)

    async def _llm_producer() -> None:
        tts_buffer = ""
        try:
            gen = agent_service.generate_stream(
                user_input=user_input,
                messages=history,
                context=context,
                system_prompt=system_prompt,
            )
            while True:
                try:
                    async with asyncio.timeout(settings.agent_idle_timeout_seconds):
                        event = await anext(gen)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    await out_q.put(ErrorEvent(content="Agent timeout"))
                    return

                await out_q.put(event)

                if isinstance(event, TokenEvent):
                    if enable_tts and event.content:
                        tts_buffer += event.content
                        sentences, tts_buffer = _drain_sentences(tts_buffer)
                        for sentence in sentences:
                            await sentence_q.put(sentence)
                elif isinstance(event, AssistantEvent):
                    await message_repo.create(
                        Message(
                            conversation_id=conversation.id,
                            character=character,
                            body=event.message,
                            context=context,
                        )
                    )
                elif isinstance(event, ToolResultEvent):
                    await message_repo.create(
                        Message(
                            conversation_id=conversation.id,
                            character=character,
                            body=_tool_event_to_message_body(event),
                            context=context,
                        )
                    )
                    # Drop any pre-tool buffer so only the final post-tool answer
                    # is spoken (pre-tool assistant text is typically empty).
                    tts_buffer = ""

            # Flush the trailing partial sentence once the stream ends.
            if enable_tts:
                tail = tts_buffer.strip()
                if tail:
                    await sentence_q.put(tail)
        except Exception:
            logger.exception(
                "Response streaming failed (conversation_id=%s)", conversation.id
            )
            await out_q.put(ErrorEvent(content="Internal server error"))
        finally:
            if enable_tts:
                await sentence_q.put(None)  # signal the worker to stop
            await out_q.put(sentinel)

    producers = [asyncio.create_task(_llm_producer(), name="voice-llm-producer")]
    if enable_tts:
        producers.append(asyncio.create_task(_tts_worker(), name="voice-tts-worker"))

    errored = False
    remaining = len(producers)
    try:
        while remaining > 0:
            item = await out_q.get()
            if item is sentinel:
                remaining -= 1
                continue
            if isinstance(item, ErrorEvent):
                errored = True
            yield item
    finally:
        for task in producers:
            if not task.done():
                task.cancel()
        for task in producers:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    if not errored:
        yield DoneEvent(conversation_id=conversation.id)


async def _resolve_conversation(
    repo: IConversationRepository, conversation_id: str | None
) -> Conversation | None:
    if conversation_id:
        return await repo.get_by_id(conversation_id)
    return await repo.create(Conversation(title="Obrolan Baru"))


def _tool_event_to_message_body(event: ToolResultEvent):
    """Convert a tool-result event to a domain ToolMessage."""
    from src.domain.entities import ToolMessage

    return ToolMessage(
        content=event.content,
        tool_call_id=event.tool_call_id,
        name=event.name,
    )
