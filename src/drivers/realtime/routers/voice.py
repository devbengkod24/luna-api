"""Voice WebSocket router.

The client streams raw 16-bit PCM mono audio at the configured sample rate.
Server-side VAD segments utterances, runs them through the existing speech
response pipeline (STT → agent → TTS), and streams events back. Supports
barge-in: a new speech segment cancels the in-flight reply.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.config import settings
from src.domain.entities import UserContext
from src.domain.events import TranscriptEvent
from src.drivers import dependencies
from src.drivers.realtime.voice import SegmentEvent, VadSegmenter
from src.drivers.rest.presenters import stream_event_to_dict
from src.shared.constants import Character

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])


def _parse_character(value: Optional[str]) -> Character:
    if value is None:
        return Character.LUNA
    try:
        return Character(int(value))
    except (ValueError, TypeError):
        return Character.LUNA


async def _safe_send(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_text(json.dumps(payload))
    except (WebSocketDisconnect, RuntimeError, ConnectionError) as e:
        logger.debug("Voice WS send failed (client disconnected): %s", e)
        raise WebSocketDisconnect() from e


@router.websocket("/stream")
async def voice_stream(websocket: WebSocket):
    """Continuous-listening voice endpoint with VAD segmentation and barge-in.

    Query params:
      - `character`: 0 (LUNA, active). 1 (VENDRA) is reserved for future use.
        Defaults to LUNA.
      - `conversation_id`: optional existing conversation to continue.
      - `sample_rate`: 8000/16000/32000/48000. Defaults to 16000.
    """
    if not settings.enable_voice_ws:
        await websocket.close(code=1011, reason="Voice WS disabled")
        return
    if not settings.enable_stt:
        await websocket.close(code=1011, reason="STT disabled")
        return

    container = dependencies.container_from(websocket)
    if container.stt_service is None or container.agent_service is None:
        await websocket.close(code=1011, reason="Voice services not initialized")
        return

    await websocket.accept()
    logger.info("Client connected to voice stream")

    character = _parse_character(websocket.query_params.get("character"))
    conversation_id = websocket.query_params.get("conversation_id")
    try:
        sample_rate = int(
            websocket.query_params.get("sample_rate") or settings.voice_vad_sample_rate
        )
    except (TypeError, ValueError):
        sample_rate = settings.voice_vad_sample_rate

    try:
        if container.vad_service is not None and container.vad_service.sample_rate == sample_rate:
            vad = container.vad_service
        else:
            # Client requested a non-default sample rate. Build a per-connection
            # VAD honouring the configured provider (silero/webrtc) with the
            # same fallback semantics as the global factory.
            from src.adapters.vad import SileroVadService, WebRtcVadService

            provider = (settings.vad_provider or "silero").lower().strip()
            vad = None
            if provider == "silero":
                try:
                    vad = SileroVadService(
                        sample_rate=sample_rate,
                        frame_duration_ms=settings.voice_vad_frame_ms,
                        threshold=settings.silero_vad_threshold,
                    )
                except Exception as e:
                    logger.warning(
                        "Silero VAD load failed (%s) — falling back to WebRTC", e,
                    )
            if vad is None:
                vad = WebRtcVadService(
                    aggressiveness=settings.voice_vad_aggressiveness,
                    sample_rate=sample_rate,
                    frame_duration_ms=settings.voice_vad_frame_ms,
                )
    except Exception as e:
        logger.warning("Failed to construct VAD: %s", e)
        await _safe_send(
            websocket,
            {"event": "error", "payload": {"content": "VAD initialization failed"}},
        )
        await websocket.close(code=1011)
        return

    max_utterance_frames = max(
        1,
        int(settings.voice_max_utterance_seconds * 1000 / vad.frame_duration_ms),
    )
    segmenter = VadSegmenter(
        vad=vad,
        speech_start_frames=settings.voice_speech_start_frames,
        speech_end_frames=settings.voice_speech_end_frames,
        prebuffer_frames=settings.voice_prebuffer_frames,
        max_utterance_frames=max_utterance_frames,
    )

    pipeline_task: Optional[asyncio.Task] = None
    audio_buffer = bytearray()

    async def _run_pipeline(pcm: bytes) -> None:
        nonlocal conversation_id

        if container.db_connection is None:
            await _safe_send(
                websocket,
                {"event": "error", "payload": {"content": "Database not initialized"}},
            )
            return

        from src.adapters.persistence import ConversationRepository, MessageRepository
        from src.use_cases import ResponseTextUseCase

        # Gate VAD for the duration of this reply. Barge-in is exclusively
        # client-driven via hotword ("stop" / "berhenti") sent as an
        # {"event": "interrupt"} text frame. The finally clause guarantees the
        # gate is released no matter how the pipeline exits.
        segmenter.set_responding(True)
        try:
            try:
                transcript = await container.stt_service.transcribe_pcm(pcm, sample_rate)
            except ValueError as e:
                logger.debug("STT validation failed: %s", e)
                return
            except Exception:
                logger.exception("STT failed")
                await _safe_send(
                    websocket,
                    {"event": "error", "payload": {"content": "Transcription failed"}},
                )
                return

            # VAD picked up noise / a too-short utterance — skip silently.
            # Client is already in "listening" state; no need to surface this.
            if not transcript or not transcript.strip():
                logger.debug("Empty transcript — skipping pipeline")
                await _safe_send(websocket, {"event": "no_speech", "payload": {}})
                return

            transcript_payload = stream_event_to_dict(TranscriptEvent(content=transcript))
            await _safe_send(
                websocket,
                {"event": transcript_payload["type"], "payload": transcript_payload},
            )

            try:
                async for session in container.db_connection.get_session():
                    use_case = ResponseTextUseCase(
                        conversation_repo=ConversationRepository(session),
                        message_repo=MessageRepository(session),
                        agent_service=container.agent_service,
                        tts_service=container.tts_service,
                        prompt_repo=container.prompt_repository,
                        title_runner=dependencies.make_title_runner(container),
                    )
                    async for event in use_case.execute(
                        conversation_id=conversation_id,
                        user_input=transcript,
                        character=character,
                        context=UserContext(),
                        enable_tts=settings.enable_tts and container.tts_service is not None,
                    ):
                        payload = stream_event_to_dict(event)
                        if payload.get("type") == "done" and payload.get("conversation_id"):
                            conversation_id = payload["conversation_id"]
                        await _safe_send(
                            websocket,
                            {"event": payload.get("type"), "payload": payload},
                        )
            except asyncio.CancelledError:
                await _safe_send(
                    websocket,
                    {"event": "interrupted", "payload": {"content": "Reply interrupted"}},
                )
                raise
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.exception("Voice pipeline failed")
                await _safe_send(
                    websocket,
                    {"event": "error", "payload": {"content": "Internal server error"}},
                )
        finally:
            segmenter.set_responding(False)

    async def _process_segment(tick) -> None:
        nonlocal pipeline_task

        if tick.event == SegmentEvent.NONE:
            return

        if tick.event == SegmentEvent.SPEECH_START:
            if pipeline_task is not None and not pipeline_task.done():
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except (asyncio.CancelledError, Exception):
                    pass
                pipeline_task = None
            await _safe_send(websocket, {"event": "speech_start", "payload": {}})
            return

        if tick.event == SegmentEvent.SPEECH_END and tick.utterance:
            await _safe_send(websocket, {"event": "speech_end", "payload": {}})
            pipeline_task = asyncio.create_task(
                _run_pipeline(tick.utterance), name="voice-pipeline"
            )

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=settings.frame_timeout_seconds,
                )
            except asyncio.TimeoutError:
                await _safe_send(websocket, {"event": "ping", "payload": {}})
                continue

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            text_data = message.get("text")
            bytes_data = message.get("bytes")

            if text_data is not None:
                try:
                    data = json.loads(text_data)
                except json.JSONDecodeError:
                    await _safe_send(
                        websocket,
                        {"event": "error", "payload": {"content": "Invalid JSON"}},
                    )
                    continue

                event = data.get("event")
                if event == "reset":
                    if pipeline_task is not None and not pipeline_task.done():
                        pipeline_task.cancel()
                    pipeline_task = None
                    segmenter.reset()
                    audio_buffer.clear()
                    conversation_id = None
                    await _safe_send(websocket, {"event": "reset_ack", "payload": {}})
                elif event == "interrupt":
                    # Client-driven barge-in (hotword "stop" / "berhenti" or
                    # manual stop button). Cancel the in-flight reply; the
                    # pipeline's `finally` releases the segmenter gate so the
                    # next utterance is captured cleanly.
                    if pipeline_task is not None and not pipeline_task.done():
                        pipeline_task.cancel()
                    await _safe_send(websocket, {"event": "interrupt_ack", "payload": {}})
                elif event == "set_conversation":
                    conversation_id = (data.get("payload") or {}).get("conversation_id")
                    await _safe_send(
                        websocket,
                        {"event": "ack", "payload": {"conversation_id": conversation_id}},
                    )
                else:
                    await _safe_send(
                        websocket,
                        {"event": "error", "payload": {"content": f"Unknown event: {event}"}},
                    )
                continue

            if bytes_data is None:
                continue

            audio_buffer.extend(bytes_data)
            frame_size = segmenter.frame_bytes
            while len(audio_buffer) >= frame_size:
                frame = bytes(audio_buffer[:frame_size])
                del audio_buffer[:frame_size]
                tick = segmenter.push(frame)
                await _process_segment(tick)

    except WebSocketDisconnect:
        logger.info("Client disconnected from voice stream")
    except Exception:
        logger.exception("Unexpected error in voice stream")
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
            logger.debug("Error closing voice WS: %s", e)
