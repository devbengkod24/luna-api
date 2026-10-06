"""Response use cases — text and speech AI response with streaming."""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


@dataclass(frozen=True)
class ResponseOptions:
    """The knobs a reply runs under, handed in by the composition root.

    Passed rather than read from `src.config`, so the use case depends on the
    values and not on where they are configured.
    """

    history_limit: int
    agent_idle_timeout_s: float
    timezone: str


def _new_conversation_title(timezone: str) -> str:
    """Title for a fresh conversation: the local wall-clock time it started.

    Replaces an LLM-generated title, which cost an extra cloud round-trip per
    conversation and could not be persisted anyway — it ran detached, in its own
    transaction, while the conversation row was still uncommitted on the request
    session, so its UPDATE matched zero rows ~86% of the time. A timestamp is
    computed inline and committed with the request, so it always lands.
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(
            "Unknown timezone %r — falling back to UTC for conversation titles", timezone
        )
        now = datetime.now(ZoneInfo("UTC"))
    return now.strftime("%Y-%m-%d %H:%M:%S")


class SynthesizeTtsUseCase:
    """Synthesize one sentence of the reply."""

    def __init__(self, tts_service: ITTSService):
        self.tts_service = tts_service

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
                    sample_rate=self.tts_service.sample_rate,
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
        tts_service: ITTSService | None,
        prompt_repo: IPromptRepository,
        options: ResponseOptions,
    ):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.agent_service = agent_service
        self.prompt_repo = prompt_repo
        self.options = options
        self.tts_use_case = SynthesizeTtsUseCase(tts_service) if tts_service else None

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
            prompt_repo=self.prompt_repo,
            conversation_id=conversation_id,
            user_input=user_input,
            character=character,
            context=context,
            enable_tts=enable_tts and self.tts_use_case is not None,
            options=self.options,
        ):
            yield event


class ResponseSpeechUseCase:
    """Use case for speech-based AI response with streaming."""

    def __init__(
        self,
        conversation_repo: IConversationRepository,
        message_repo: IMessageRepository,
        agent_service: IAgentService,
        tts_service: ITTSService | None,
        stt_service: ISTTService,
        prompt_repo: IPromptRepository,
        options: ResponseOptions,
    ):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.agent_service = agent_service
        self.stt_service = stt_service
        self.prompt_repo = prompt_repo
        self.options = options
        self.tts_use_case = SynthesizeTtsUseCase(tts_service) if tts_service else None

    async def execute(
        self,
        conversation_id: str | None,
        speech_base64: str,
        character: Character,
        context: UserContext | None,
        enable_tts: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        yield LogEvent(content="Mentranskripsi speech...")
        try:
            transcript = await self.stt_service.transcribe(speech_base64)
        except ValueError:
            # Unreadable audio: the client's to fix, and not a server error.
            yield ErrorEvent(content="Audio tidak valid")
            return
        except Exception:
            # Raised out of the stream, this used to cut the SSE response off
            # mid-way with no event at all; the page waited for a reply that
            # had already failed.
            logger.exception("Transcription failed")
            yield ErrorEvent(content="Transkripsi gagal")
            return
        if not transcript or not transcript.strip():
            yield ErrorEvent(content="Tidak ada suara yang terdengar")
            return
        yield TranscriptEvent(content=transcript)

        async for event in _stream_conversation(
            conversation_repo=self.conversation_repo,
            message_repo=self.message_repo,
            agent_service=self.agent_service,
            tts_use_case=self.tts_use_case,
            prompt_repo=self.prompt_repo,
            conversation_id=conversation_id,
            user_input=transcript,
            character=character,
            context=context,
            enable_tts=enable_tts and self.tts_use_case is not None,
            options=self.options,
        ):
            yield event


# A run of sentence punctuation.
_TERMINATORS = re.compile(r"[.!?\n]+")
# The word just before a dot, to tell "Prof." from the end of a sentence.
_LAST_WORD = re.compile(r"(\w+)$")
# Words a dot follows without ending the sentence: titles before a name, and
# the abbreviations that sit mid-sentence. Cutting at them split one name into
# three TTS requests with a pause in each gap. Deliberately not "dll."/"dsb.",
# which end sentences as often as not.
_NO_BREAK_AFTER = frozenset({
    "prof", "dr", "drs", "dra", "ir", "h", "hj", "st", "sh", "se", "si", "mm",
    "mt", "mkom", "kom", "pd", "ag", "psi", "ked", "sp", "no", "nomor", "jl",
    "jln", "telp", "tel", "hlm", "yth", "sdr", "sdri", "bpk", "kel", "kec",
    "kab", "prov", "mr", "mrs", "ms", "vs", "ph", "phd",
})
# Force-flush a terminator-free buffer this long so a monologue still starts
# speaking instead of stalling.
_TTS_FORCE_FLUSH_CHARS = 220


def _drain_sentences(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences out of an accumulating token buffer.

    Returns (complete_sentences, remainder). The remainder is the trailing
    partial sentence still being streamed; the caller keeps accumulating into it.

    A `.`, `!` or `?` ends a sentence only when WHITESPACE follows it, so the
    decision waits for the next character. Cutting at any dot, as this did,
    split "Rp 7.500.000" into "Rp 7." / "500." / "000" — which the normalizer
    then read, correctly for each fragment, as "tujuh rupiah / lima ratus / nol"
    — and "dinus.ac.id", "3.25" and "Prof. Dr." the same way. A newline always
    ends one.
    """
    sentences: list[str] = []
    start = 0
    for match in _TERMINATORS.finditer(buffer):
        end = match.end()
        run = match.group(0)
        if "\n" not in run:
            if end >= len(buffer):
                break  # the next token decides
            if not buffer[end].isspace():
                continue  # 7.500, dinus.ac.id, 3.25
            if set(run) == {"."}:
                word = _LAST_WORD.search(buffer[start:match.start()])
                if word and (
                    word.group(1).lower() in _NO_BREAK_AFTER
                    or (len(word.group(1)) == 1 and word.group(1).isalpha())
                ):
                    continue  # Prof. Dr. Pulung, A. Rahman
        chunk = buffer[start:end].strip()
        if chunk:
            sentences.append(chunk)
        start = end
    remainder = buffer[start:]
    if len(remainder) >= _TTS_FORCE_FLUSH_CHARS:
        # At the last space, never inside a word or a number.
        cut = remainder.rstrip().rfind(" ")
        if cut <= 0:
            cut = len(remainder)
        head = remainder[:cut].strip()
        if head:
            sentences.append(head)
        remainder = remainder[cut:]
    return sentences, remainder


async def _stream_conversation(
    *,
    conversation_repo: IConversationRepository,
    message_repo: IMessageRepository,
    agent_service: IAgentService,
    tts_use_case: SynthesizeTtsUseCase | None,
    prompt_repo: IPromptRepository,
    conversation_id: str | None,
    user_input: str,
    character: Character,
    context: UserContext | None,
    enable_tts: bool,
    options: ResponseOptions,
) -> AsyncGenerator[StreamEvent, None]:
    """Core streaming logic shared by text and speech use cases.

    The LLM token stream and TTS synthesis run concurrently: as the agent emits
    tokens, complete sentences are handed to a TTS worker that synthesizes them
    one at a time (preserving order) while the model keeps generating. Audio for
    the first sentence therefore starts playing well before the full reply is
    written — the dominant latency win over synthesizing the whole reply at the
    end. Both the voice WS and chat SSE paths flow through here.
    """
    conversation = await _resolve_conversation(conversation_repo, conversation_id, options.timezone)
    if conversation is None:
        yield ErrorEvent(content="Conversation not found")
        return

    if conversation_id is None:
        yield LogEvent(content="Membuat sesi baru...")
        yield LogEvent(content=f"Sesi dibuat: {conversation.title}")

    system_prompt = prompt_repo.get_system_prompt(character, context)

    yield LogEvent(content="Menganalisis konteks...")
    history = await message_repo.get_recent_by_conversation_id(
        conversation.id, options.history_limit
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
                    async with asyncio.timeout(options.agent_idle_timeout_s):
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
    repo: IConversationRepository, conversation_id: str | None, timezone: str
) -> Conversation | None:
    if conversation_id:
        return await repo.get_by_id(conversation_id)
    return await repo.create(Conversation(title=_new_conversation_title(timezone)))


def _tool_event_to_message_body(event: ToolResultEvent):
    """Convert a tool-result event to a domain ToolMessage."""
    from src.domain.entities import ToolMessage

    return ToolMessage(
        content=event.content,
        tool_call_id=event.tool_call_id,
        name=event.name,
    )
