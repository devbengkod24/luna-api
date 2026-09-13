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
"""

import asyncio
import json
import logging
import struct
from contextlib import aclosing

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.config import settings
from src.domain.entities import UserContext
from src.domain.events import TranscriptEvent
from src.drivers import dependencies
from src.drivers.rest.presenters import stream_event_to_dict
from src.shared.constants import Character

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/turn", tags=["turn"])

_TYPE_AUDIO = 0x01
_HEADER = struct.Struct(">BQ")


def _parse_character(value: str | None) -> Character:
    if value is None:
        return Character.LUNA
    try:
        return Character(int(value))
    except (ValueError, TypeError):
        return Character.LUNA


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


async def _safe_send(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_text(json.dumps(payload))
    except (WebSocketDisconnect, RuntimeError, ConnectionError) as e:
        logger.debug("Turn WS send failed (client disconnected): %s", e)
        raise WebSocketDisconnect() from e


@router.websocket("/stream")
async def turn_stream(websocket: WebSocket):
    """Per-kiosk turn socket. Query params: `kiosk_id`, `character`,
    `conversation_id`, `sample_rate`."""
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
    character = _parse_character(websocket.query_params.get("character"))
    conversation_id = websocket.query_params.get("conversation_id")
    try:
        sample_rate = int(
            websocket.query_params.get("sample_rate") or settings.voice_sample_rate
        )
    except (TypeError, ValueError):
        sample_rate = settings.voice_sample_rate

    logger.info("Kiosk %s connected to turn stream (sample_rate=%d)", kiosk_id, sample_rate)

    # Audio for the turn currently being spoken. Capped so a kiosk that never
    # sends `turn_end` (crashed, or a bug in its endpointer) cannot grow this
    # process's memory without bound.
    max_bytes = int(settings.voice_max_utterance_seconds * sample_rate) * 2
    audio = bytearray()
    turn_id: str | None = None
    turn_meta: dict = {}
    truncated = False
    pipeline_task: asyncio.Task | None = None

    async def _cancel_pipeline(reason: str) -> None:
        nonlocal pipeline_task
        if pipeline_task is None or pipeline_task.done():
            pipeline_task = None
            return
        pipeline_task.cancel()
        try:
            await pipeline_task
        except (asyncio.CancelledError, Exception):
            pass
        pipeline_task = None
        # Sent from here, not from inside the cancelled coroutine: a task being
        # cancelled is a poor place to do I/O, and if this send is skipped the
        # kiosk never stops playing the reply it just asked us to abandon.
        await _safe_send(
            websocket,
            {"event": "interrupted", "payload": {"content": "Reply interrupted", "reason": reason}},
        )

    async def _run_pipeline(pcm: bytes, meta: dict, this_turn: str) -> None:
        nonlocal conversation_id

        if container.db_connection is None:
            await _safe_send(
                websocket, {"event": "error", "payload": {"content": "Database not initialized"}}
            )
            return

        from src.adapters.persistence import ConversationRepository, MessageRepository
        from src.use_cases import ResponseTextUseCase

        typed = _typed_question(meta)
        try:
            # Nothing to transcribe for a typed question — and no Whisper to
            # wait on, so a tapped suggestion starts answering at once.
            transcript = typed or await container.stt_service.transcribe_pcm(pcm, sample_rate)
        except ValueError as e:
            logger.debug("STT validation failed: %s", e)
            await _safe_send(websocket, {"event": "no_speech", "payload": {"turn_id": this_turn}})
            return
        except Exception:
            logger.exception("STT failed")
            await _safe_send(
                websocket, {"event": "error", "payload": {"content": "Transcription failed"}}
            )
            return

        if not transcript or not transcript.strip():
            # Silent mouthing, a cough, or room noise the kiosk let through.
            # Whisper's own VAD filter returns empty rather than inventing
            # subtitle boilerplate, so this is the expected quiet path.
            await _safe_send(websocket, {"event": "no_speech", "payload": {"turn_id": this_turn}})
            return

        payload = stream_event_to_dict(TranscriptEvent(content=transcript))
        payload["turn_id"] = this_turn
        await _safe_send(websocket, {"event": payload["type"], "payload": payload})

        user_context = _context_from(meta)
        # DEBUG, never INFO: name / NIM / emotion are PII.
        logger.debug(
            "turn %s context: recognised=%s emotion=%s",
            this_turn[:8], user_context.name is not None, user_context.emotion,
        )

        try:
            async for db_session in container.db_connection.get_session():
                use_case = ResponseTextUseCase(
                    conversation_repo=ConversationRepository(db_session),
                    message_repo=MessageRepository(db_session),
                    agent_service=container.agent_service,
                    tts_service=container.tts_service,
                    prompt_repo=container.prompt_repository,
                )
                async with aclosing(
                    use_case.execute(
                        conversation_id=conversation_id,
                        user_input=transcript,
                        character=character,
                        context=user_context,
                        enable_tts=settings.enable_tts and container.tts_service is not None,
                    )
                ) as stream:
                    async for event in stream:
                        event_payload = stream_event_to_dict(event)
                        event_type = event_payload.get("type")
                        if event_type == "done" and event_payload.get("conversation_id"):
                            conversation_id = event_payload["conversation_id"]
                        event_payload["turn_id"] = this_turn
                        await _safe_send(
                            websocket, {"event": event_type, "payload": event_payload}
                        )
        except WebSocketDisconnect:
            raise
        except Exception:
            logger.exception("Turn pipeline failed")
            await _safe_send(
                websocket, {"event": "error", "payload": {"content": "Internal server error"}}
            )

    async def _handle_control(text_data: str) -> None:
        nonlocal turn_id, turn_meta, audio, truncated, pipeline_task, conversation_id

        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            await _safe_send(websocket, {"event": "error", "payload": {"content": "Invalid JSON"}})
            return

        event = data.get("event")
        payload = data.get("payload") or {}

        if event == "turn_begin":
            # A new turn supersedes anything still running: the kiosk granted the
            # floor, and it is the only thing that knows the room.
            await _cancel_pipeline("superseded")
            turn_id = payload.get("turn_id")
            turn_meta = payload
            audio = bytearray()
            truncated = False
            logger.info(
                "turn %s begin (kiosk=%s trigger=%s)",
                (turn_id or "?")[:8], kiosk_id, payload.get("trigger"),
            )
            await _safe_send(websocket, {"event": "ack", "payload": {"turn_id": turn_id}})

        elif event == "turn_end":
            typed = _typed_question(turn_meta)
            if turn_id is None or not (audio or typed):
                await _safe_send(websocket, {"event": "no_speech", "payload": {"turn_id": turn_id}})
                return
            if typed and len(typed) > settings.chat_max_prompt_length:
                await _safe_send(
                    websocket,
                    {"event": "error", "payload": {"content": "Question too long", "turn_id": turn_id}},
                )
                return
            pcm, meta, this_turn = bytes(audio), turn_meta, turn_id
            audio = bytearray()
            if typed:
                logger.info("turn %s end (typed, %d chars)", this_turn[:8], len(typed))
            else:
                logger.info(
                    "turn %s end (%.2fs audio%s)",
                    this_turn[:8], len(pcm) / 2 / sample_rate, ", truncated" if truncated else "",
                )
            pipeline_task = asyncio.create_task(
                asyncio.wait_for(
                    _run_pipeline(pcm, meta, this_turn),
                    timeout=settings.agent_idle_timeout_seconds * 2,
                ),
                name="turn-pipeline",
            )

        elif event == "cancel":
            audio = bytearray()
            await _cancel_pipeline(payload.get("reason") or "cancelled")

        elif event == "reset":
            await _cancel_pipeline("reset")
            audio = bytearray()
            turn_id = None
            turn_meta = {}
            conversation_id = None
            await _safe_send(websocket, {"event": "reset_ack", "payload": {}})

        elif event == "set_conversation":
            conversation_id = payload.get("conversation_id")
            await _safe_send(
                websocket, {"event": "ack", "payload": {"conversation_id": conversation_id}}
            )

        else:
            await _safe_send(
                websocket, {"event": "error", "payload": {"content": f"Unknown event: {event}"}}
            )

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=settings.frame_timeout_seconds
                )
            except asyncio.TimeoutError:
                await _safe_send(websocket, {"event": "ping", "payload": {}})
                continue

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            text_data = message.get("text")
            if text_data is not None:
                await _handle_control(text_data)
                continue

            bytes_data = message.get("bytes")
            if not bytes_data or len(bytes_data) < _HEADER.size:
                continue
            msg_type, _ts = _HEADER.unpack_from(bytes_data, 0)
            if msg_type != _TYPE_AUDIO or turn_id is None:
                continue
            chunk = bytes_data[_HEADER.size:]
            if len(audio) + len(chunk) > max_bytes:
                if not truncated:
                    logger.warning(
                        "turn %s exceeded %.0fs — truncating",
                        turn_id[:8], settings.voice_max_utterance_seconds,
                    )
                    truncated = True
                continue
            audio.extend(chunk)

    except WebSocketDisconnect:
        logger.info("Kiosk %s disconnected from turn stream", kiosk_id)
    except Exception:
        logger.exception("Unexpected error in turn stream")
    finally:
        if pipeline_task is not None and not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await websocket.close()
        except Exception as e:
            logger.debug("Error closing turn WS: %s", e)
