"""Turn WebSocket — the cloud half of the edge/cloud split.

One persistent connection per kiosk, carrying one turn at a time. The kiosk has
already decided *who* is speaking and *whether* they are speaking to Luna; this
endpoint is deliberately incapable of second-guessing that. There is no VAD here,
no active-speaker model, no facing score and no video: segmentation and the
device-directed decision need millisecond-scale alignment between lips and audio,
and that alignment is exactly what a network hop destroys.

What is left is a plain audio-in / text-and-audio-out service:

    turn_begin  -> speaker identity, affect and attention metadata (~500 bytes)
    <binary>    -> PCM-16, streamed from the moment the floor was granted
    turn_end    -> transcribe what arrived, then LLM + RAG + TTS
    cancel      -> drop the in-flight reply (barge-in, or the speaker left)

Wire framing is shared with `/session/stream`: [type:1][ts:uint64 BE][payload],
type 0x01 = audio. Only 0x01 is accepted here — video has no reason to exist on
this socket.

Every event about a turn carries that turn's `turn_id`, errors and
`interrupted` included. The kiosk acts only on events that name the turn it
holds; the socket's own chatter (`ping`, `ack`) names none. A missing id used to
mean "the current turn", and a late `error` from a superseded turn then closed
the next one.
"""

import asyncio
import json
import logging
import re
import struct
from contextlib import aclosing

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.config import settings
from src.domain.entities import UserContext
from src.domain.events import DoneEvent, TranscriptEvent
from src.drivers import dependencies
from src.drivers.rest.presenters import stream_event_to_dict
from src.drivers.security import token_accepted
from src.shared.constants import UUID_PATTERN, Character

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/turn", tags=["turn"])

_TYPE_AUDIO = 0x01
_HEADER = struct.Struct(">BQ")
_UUID = re.compile(UUID_PATTERN)


def _parse_character(value: str | None) -> Character:
    if value is None:
        return Character.LUNA
    try:
        return Character(int(value))
    except (ValueError, TypeError):
        return Character.LUNA


def _conversation_id(value) -> str | None:
    """A conversation id this service could have minted, or None.

    Anything else reached the database lookup and failed there, taking every
    turn on the socket down with an "Internal server error" until a reset.
    """
    return value if isinstance(value, str) and _UUID.match(value) else None


def _context_from(payload: dict) -> UserContext:
    """Build the LLM's user context from what the edge observed.

    The kiosk has already voted identity and emotion over the whole utterance,
    so this is a straight read rather than another aggregation. Missing fields
    are normal — an unrecognised visitor still gets answered, just generically.
    """
    speaker = payload.get("speaker") or {}
    identity = speaker.get("identity") or {}
    affect = payload.get("affect") or {}
    name = identity.get("name")
    return UserContext(
        name=name if name and name != "Unknown" else None,
        nim=identity.get("nim"),
        emotion=affect.get("emotion"),
    )


def _typed_question(meta: dict) -> str | None:
    """The question, when this turn arrived as text rather than as audio.

    A suggestion tapped on the kiosk while it is in voice mode. It rides the
    same turn as speech on purpose: the same conversation, so a follow-up asked
    out loud has it as context; and the kiosk's floor control, so the reply is
    governed by barge-in like any other.
    """
    text = meta.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


class TurnSocket:
    """One kiosk's connection: the turn being recorded, the reply being made."""

    def __init__(self, websocket: WebSocket, container, *, kiosk_id, character, conversation_id, sample_rate):
        self.websocket = websocket
        self.container = container
        self.kiosk_id = kiosk_id
        self.character = character
        self.conversation_id = conversation_id
        self.sample_rate = sample_rate
        # Audio for the turn currently being spoken. Capped so a kiosk that
        # never sends `turn_end` (crashed, or a bug in its endpointer) cannot
        # grow this process's memory without bound.
        self.max_bytes = int(settings.voice_max_utterance_seconds * sample_rate) * 2
        self.audio = bytearray()
        self.turn_id: str | None = None
        self.turn_meta: dict = {}
        self.truncated = False
        self.pipeline: asyncio.Task | None = None
        self.pipeline_turn: str | None = None
        self._warming: asyncio.Task | None = None
        self._stop_warming = asyncio.Event()

    # --- sending ------------------------------------------------------------

    async def send(self, event: str, payload: dict | None = None, turn_id: str | None = None) -> None:
        body = dict(payload or {})
        if turn_id is not None:
            body["turn_id"] = turn_id
        try:
            await self.websocket.send_text(json.dumps({"event": event, "payload": body}))
        except (WebSocketDisconnect, RuntimeError, ConnectionError) as e:
            logger.debug("Turn WS send failed (client disconnected): %s", e)
            raise WebSocketDisconnect() from e

    # --- keeping the STT connection warm -----------------------------------

    def _start_warming(self) -> None:
        stt = self.container.stt_service
        if stt is None or self._warming is not None:
            return
        self._stop_warming = asyncio.Event()
        self._warming = asyncio.create_task(self._keep_warm(stt), name="stt-keep-warm")

    async def _keep_warm(self, stt) -> None:
        """While the visitor speaks, keep the STT connection open for turn_end.

        The first touch is immediate: by the time a turn opens, the connection
        from the last one has long been closed by the provider.
        """
        interval = settings.stt_keep_warm_interval_seconds
        while not self._stop_warming.is_set():
            await stt.keep_warm()
            try:
                await asyncio.wait_for(self._stop_warming.wait(), timeout=interval)
            except TimeoutError:
                pass

    async def _end_warming(self) -> None:
        """Stop, and wait for a touch still in flight: otherwise the
        transcription finds that connection busy and opens a cold one."""
        task, self._warming = self._warming, None
        if task is None:
            return
        self._stop_warming.set()
        try:
            await asyncio.wait_for(task, timeout=5)
        except (asyncio.CancelledError, Exception):
            task.cancel()

    # --- the reply ------------------------------------------------------------

    async def _cancel_pipeline(self, reason: str) -> None:
        task, turn = self.pipeline, self.pipeline_turn
        self.pipeline = self.pipeline_turn = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        # Sent from here, not from inside the cancelled coroutine: a task being
        # cancelled is a poor place to do I/O, and if this send is skipped the
        # kiosk never stops playing the reply it just asked us to abandon.
        await self.send(
            "interrupted", {"content": "Reply interrupted", "reason": reason}, turn_id=turn
        )

    async def _run_pipeline_bounded(self, pcm: bytes, meta: dict, this_turn: str) -> None:
        """The pipeline under a ceiling that TELLS the kiosk when it is hit.

        A bare `wait_for` cancelled the pipeline and said nothing, while the
        receive loop went on sending `ping`: the kiosk read that as the reply
        still coming and held the floor for as long as the socket stayed up.
        """
        try:
            async with asyncio.timeout(settings.agent_idle_timeout_seconds * 2):
                await self._run_pipeline(pcm, meta, this_turn)
        except TimeoutError:
            logger.warning("turn %s: reply exceeded its time limit — abandoned", this_turn[:8])
            try:
                await self.send("error", {"content": "Reply timed out"}, turn_id=this_turn)
            except WebSocketDisconnect:
                pass
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Turn pipeline crashed")

    async def _run_pipeline(self, pcm: bytes, meta: dict, this_turn: str) -> None:
        container = self.container
        if container.db_connection is None:
            await self.send("error", {"content": "Database not initialized"}, turn_id=this_turn)
            return

        typed = _typed_question(meta)
        try:
            # Nothing to transcribe for a typed question — and no STT round
            # trip to wait on, so a tapped suggestion starts answering at once.
            transcript = typed or await container.stt_service.transcribe_pcm(pcm, self.sample_rate)
        except ValueError as e:
            logger.debug("STT validation failed: %s", e)
            await self.send("no_speech", turn_id=this_turn)
            return
        except Exception:
            logger.exception("STT failed")
            await self.send("error", {"content": "Transcription failed"}, turn_id=this_turn)
            return

        if not transcript or not transcript.strip():
            # Silent mouthing, a cough, or room noise the kiosk let through.
            # Deepgram returns an empty transcript for audio with no words, so
            # this is the expected quiet path.
            await self.send("no_speech", turn_id=this_turn)
            return

        await self.send("transcript", stream_event_to_dict(TranscriptEvent(content=transcript)), turn_id=this_turn)

        user_context = _context_from(meta)
        # DEBUG, never INFO: name / NIM / emotion are PII.
        logger.debug(
            "turn %s context: recognised=%s emotion=%s",
            this_turn[:8], user_context.name is not None, user_context.emotion,
        )

        done: dict | None = None
        try:
            async with dependencies.db_session(container) as session:
                use_case = dependencies.build_response_text_use_case(container, session)
                async with aclosing(
                    use_case.execute(
                        conversation_id=self.conversation_id,
                        user_input=transcript,
                        character=self.character,
                        context=user_context,
                        enable_tts=settings.enable_tts and container.tts_service is not None,
                    )
                ) as stream:
                    async for event in stream:
                        payload = stream_event_to_dict(event)
                        if isinstance(event, DoneEvent):
                            # Held until the session has committed, below.
                            done = payload
                            continue
                        await self.send(payload["type"], payload, turn_id=this_turn)
        except WebSocketDisconnect:
            raise
        except Exception:
            logger.exception("Turn pipeline failed")
            await self.send("error", {"content": "Internal server error"}, turn_id=this_turn)
            return

        if done is not None:
            # After the commit: a kiosk that continues the conversation the
            # moment it hears `done` must find it there.
            if done.get("conversation_id"):
                self.conversation_id = done["conversation_id"]
            await self.send("done", done, turn_id=this_turn)

    # --- control ----------------------------------------------------------------

    async def handle_control(self, text_data: str) -> None:
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            await self.send("error", {"content": "Invalid JSON"})
            return

        event = data.get("event")
        payload = data.get("payload") or {}

        if event == "turn_begin":
            await self._begin(payload)
        elif event == "turn_end":
            await self._end(payload)
        elif event == "cancel":
            self.audio = bytearray()
            await self._end_warming()
            await self._cancel_pipeline(payload.get("reason") or "cancelled")
        elif event == "reset":
            await self._end_warming()
            await self._cancel_pipeline("reset")
            self.audio = bytearray()
            self.turn_id = None
            self.turn_meta = {}
            self.conversation_id = None
            await self.send("reset_ack")
        elif event == "set_conversation":
            requested = payload.get("conversation_id")
            if requested is not None and _conversation_id(requested) is None:
                await self.send("error", {"content": "Invalid conversation_id"})
                return
            self.conversation_id = requested
            await self.send("ack", {"conversation_id": self.conversation_id})
        else:
            await self.send("error", {"content": f"Unknown event: {event}"})

    async def _begin(self, payload: dict) -> None:
        # A new turn supersedes anything still running: the kiosk granted the
        # floor, and it is the only thing that knows the room.
        await self._end_warming()
        await self._cancel_pipeline("superseded")
        self.turn_id = payload.get("turn_id")
        self.turn_meta = payload
        self.audio = bytearray()
        self.truncated = False
        logger.info(
            "turn %s begin (kiosk=%s trigger=%s)",
            (self.turn_id or "?")[:8], self.kiosk_id, payload.get("trigger"),
        )
        await self.send("ack", turn_id=self.turn_id)
        if not _typed_question(payload):
            self._start_warming()

    async def _end(self, payload: dict) -> None:
        # Named by the kiosk's own id when this socket never saw the turn begin
        # — a reconnect between the two lands turn_end here — so the kiosk can
        # still tell the answer belongs to the turn it is holding.
        this_turn = self.turn_id or payload.get("turn_id")
        typed = _typed_question(self.turn_meta)
        await self._end_warming()
        if self.turn_id is None or not (self.audio or typed):
            await self.send("no_speech", turn_id=this_turn)
            return
        if typed and len(typed) > settings.chat_max_prompt_length:
            await self.send("error", {"content": "Question too long"}, turn_id=this_turn)
            return
        pcm, meta = bytes(self.audio), self.turn_meta
        self.audio = bytearray()
        if typed:
            logger.info("turn %s end (typed, %d chars)", this_turn[:8], len(typed))
        else:
            logger.info(
                "turn %s end (%.2fs audio%s)",
                this_turn[:8], len(pcm) / 2 / self.sample_rate, ", truncated" if self.truncated else "",
            )
        self.pipeline_turn = this_turn
        self.pipeline = asyncio.create_task(
            self._run_pipeline_bounded(pcm, meta, this_turn), name="turn-pipeline"
        )

    def handle_audio(self, bytes_data: bytes) -> None:
        if not bytes_data or len(bytes_data) < _HEADER.size:
            return
        msg_type, _ts = _HEADER.unpack_from(bytes_data, 0)
        if msg_type != _TYPE_AUDIO or self.turn_id is None:
            return
        chunk = bytes_data[_HEADER.size:]
        if len(self.audio) + len(chunk) > self.max_bytes:
            if not self.truncated:
                logger.warning(
                    "turn %s exceeded %.0fs — truncating",
                    self.turn_id[:8], settings.voice_max_utterance_seconds,
                )
                self.truncated = True
            return
        self.audio.extend(chunk)

    async def run(self) -> None:
        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.receive(), timeout=settings.frame_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    await self.send("ping")
                    continue

                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect()

                text_data = message.get("text")
                if text_data is not None:
                    await self.handle_control(text_data)
                    continue
                self.handle_audio(message.get("bytes"))

        except WebSocketDisconnect:
            logger.info("Kiosk %s disconnected from turn stream", self.kiosk_id)
        except Exception:
            logger.exception("Unexpected error in turn stream")
        finally:
            if self._warming is not None:
                self._warming.cancel()
            if self.pipeline is not None and not self.pipeline.done():
                self.pipeline.cancel()
                try:
                    await self.pipeline
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await self.websocket.close()
            except Exception as e:
                logger.debug("Error closing turn WS: %s", e)


@router.websocket("/stream")
async def turn_stream(websocket: WebSocket):
    """Per-kiosk turn socket. Query params: `kiosk_id`, `character`,
    `conversation_id`, `sample_rate`. Needs `Authorization: Bearer <API token>`
    when API_ACCESS_TOKEN is set."""
    if not token_accepted(websocket):
        logger.warning("Refused a turn socket without a valid API token")
        await websocket.close(code=1008, reason="Not authorised")
        return
    if not settings.enable_voice_ws:
        await websocket.close(code=1011, reason="Turn WS disabled")
        return
    if not settings.enable_stt:
        await websocket.close(code=1011, reason="STT disabled")
        return

    container = dependencies.container_from(websocket)
    if container.stt_service is None or container.agent_service is None:
        await websocket.close(code=1011, reason="Voice services not initialized")
        return

    await websocket.accept()

    kiosk_id = websocket.query_params.get("kiosk_id") or "unknown"
    try:
        sample_rate = int(websocket.query_params.get("sample_rate") or settings.voice_sample_rate)
    except (TypeError, ValueError):
        sample_rate = settings.voice_sample_rate

    logger.info("Kiosk %s connected to turn stream (sample_rate=%d)", kiosk_id, sample_rate)
    await TurnSocket(
        websocket,
        container,
        kiosk_id=kiosk_id,
        character=_parse_character(websocket.query_params.get("character")),
        conversation_id=_conversation_id(websocket.query_params.get("conversation_id")),
        sample_rate=sample_rate,
    ).run()
