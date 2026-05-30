"""Unified session WebSocket: audio + video on one timestamped stream.

This endpoint merges the legacy `/voice/stream` (PCM → VAD → STT → LLM → TTS)
and `/vision/stream` (JPEG → face recognition + emotion) onto a single
connection so that device-directed speech detection can fuse the two on a
common timeline.

Wire format (client → server):
  - Binary frame:  byte[0] = type (0x01 = audio PCM-16 mono, 0x02 = video JPEG)
                   bytes[1:9] = uint64 big-endian capture timestamp (ms)
                   bytes[9:]  = payload
  - Text frame (JSON):
      {"event": "facing",  "payload": {"score": 0..1, "ts": <ms>}}  facing metadata
      {"event": "reset"}                                            reset session
      {"event": "interrupt"}                                        barge-in
      {"event": "set_conversation", "payload": {"conversation_id": ...}}

Server → client events mirror the legacy routers: `speech_start`, `speech_end`,
`transcript`, `token`, `assistant`, `audio`, `done`, `interrupted`, `no_speech`,
`not_directed`, `vision_update`, `reset_ack`, `interrupt_ack`, `ack`, `ping`,
`error`.

Runs alongside the legacy routers; the frontend migrates to it, then the old
ones are removed.
"""

import asyncio
import json
import logging
import struct
import time
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.config import settings
from src.domain.entities import UserContext
from src.domain.entities.vision import VisionResult
from src.domain.events import TranscriptEvent
from src.domain.image import Image
from src.drivers import dependencies
from src.drivers.realtime.presenters import VisionDeltaCalculator, full_snapshot
from src.drivers.realtime.voice import SegmentEvent, VadSegmenter
from src.drivers.rest.presenters import stream_event_to_dict
from src.shared.constants import Character

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/session", tags=["session"])

_TYPE_AUDIO = 0x01
_TYPE_VIDEO = 0x02
_HEADER = struct.Struct(">BQ")  # type byte + uint64 ms timestamp


def _parse_character(value: str | None) -> Character:
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
        logger.debug("Session WS send failed (client disconnected): %s", e)
        raise WebSocketDisconnect() from e


def _crop_area(image: Image) -> int:
    raw = image.raw
    if raw is None:
        return 0
    return int(raw.shape[0]) * int(raw.shape[1])


@router.websocket("/stream")
async def session_stream(websocket: WebSocket):
    """Continuous audio+video session with VAD segmentation, device-directed
    gating (facing + active-speaker fusion), and barge-in.

    Query params: `character`, `conversation_id`, `sample_rate` (default 16000).
    """
    if not settings.enable_voice_ws:
        await websocket.close(code=1011, reason="Session WS disabled")
        return
    if not settings.enable_stt:
        await websocket.close(code=1011, reason="STT disabled")
        return

    container = dependencies.container_from(websocket)
    if container.stt_service is None or container.agent_service is None:
        await websocket.close(code=1011, reason="Voice services not initialized")
        return

    await websocket.accept()
    logger.info("Client connected to session stream")

    character = _parse_character(websocket.query_params.get("character"))
    conversation_id = websocket.query_params.get("conversation_id")
    try:
        sample_rate = int(
            websocket.query_params.get("sample_rate") or settings.voice_vad_sample_rate
        )
    except (TypeError, ValueError):
        sample_rate = settings.voice_vad_sample_rate

    # --- VAD / segmenter (same construction semantics as the voice router) ---
    try:
        if container.vad_service is not None and container.vad_service.sample_rate == sample_rate:
            vad = container.vad_service
        else:
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
                    logger.warning("Silero VAD load failed (%s) — falling back to WebRTC", e)
            if vad is None:
                vad = WebRtcVadService(
                    aggressiveness=settings.voice_vad_aggressiveness,
                    sample_rate=sample_rate,
                    frame_duration_ms=settings.voice_vad_frame_ms,
                )
    except Exception as e:
        logger.warning("Failed to construct VAD: %s", e)
        await _safe_send(websocket, {"event": "error", "payload": {"content": "VAD initialization failed"}})
        await websocket.close(code=1011)
        return

    max_utterance_frames = max(
        1, int(settings.voice_max_utterance_seconds * 1000 / vad.frame_duration_ms)
    )
    segmenter = VadSegmenter(
        vad=vad,
        speech_start_frames=settings.voice_speech_start_frames,
        speech_end_frames=settings.voice_speech_end_frames,
        prebuffer_frames=settings.voice_prebuffer_frames,
        max_utterance_frames=max_utterance_frames,
    )
    prebuffer_ms = settings.voice_prebuffer_frames * vad.frame_duration_ms

    # --- Vision pipeline (optional; degrades to audio-only if unavailable) ---
    vision_use_case = None
    if container.face_recognition is not None:
        try:
            vision_use_case = dependencies.get_vision_use_case(websocket)
        except Exception as e:
            logger.warning("Vision pipeline unavailable for session: %s", e)
    directedness = dependencies.make_directedness_use_case(container)
    delta = VisionDeltaCalculator()

    # --- Per-track crop ring buffers + global facing buffer, keyed by capture ts ---
    # Cap each track's buffer to the longest utterance so memory stays bounded.
    crop_buffers: dict[int, deque[tuple[int, Image]]] = {}
    facing_buffer: deque[tuple[int, float]] = deque(maxlen=2000)
    last_audio_ts = 0
    speech_start_ts = 0

    audio_buffer = bytearray()
    pipeline_task: asyncio.Task | None = None

    # latest-wins video slot
    pending_video: tuple[bytes, int] | None = None
    new_video_event = asyncio.Event()
    stop_event = asyncio.Event()
    video_frame_id = 0

    def _record_crops(crops: dict[int, Image], ts: int) -> None:
        for tid, crop in crops.items():
            dq = crop_buffers.get(tid)
            if dq is None:
                dq = deque(maxlen=max_utterance_frames)
                crop_buffers[tid] = dq
            dq.append((ts, crop))
        # Drop stale track buffers — runs on every crop arrival and also
        # unconditionally so video-stop doesn't leave stale entries indefinitely.
        cutoff = ts - int(settings.voice_max_utterance_seconds * 1000) - 2000
        for tid in [t for t, dq in list(crop_buffers.items()) if dq and dq[-1][0] < cutoff]:
            crop_buffers.pop(tid, None)

    def _select_candidate(start_ts: int, end_ts: int) -> tuple[int | None, list[Image], list[float]]:
        """Pick the closest (largest-crop) track present during the utterance."""
        best_tid: int | None = None
        best_area = -1
        best_crops: list[Image] = []
        for tid, dq in crop_buffers.items():
            crops = [c for (ts, c) in dq if start_ts <= ts <= end_ts]
            if not crops:
                continue
            area = max(_crop_area(c) for c in crops)
            if area > best_area:
                best_area, best_tid, best_crops = area, tid, crops
        facing = [s for (ts, s) in facing_buffer if start_ts <= ts <= end_ts]
        return best_tid, best_crops, facing

    # --- Reply pipeline (directedness gate → STT → LLM → TTS) ---

    async def _run_pipeline(pcm: bytes, start_ts: int, end_ts: int) -> None:
        nonlocal conversation_id

        if container.db_connection is None:
            await _safe_send(websocket, {"event": "error", "payload": {"content": "Database not initialized"}})
            return

        from src.adapters.persistence import ConversationRepository, MessageRepository
        from src.use_cases import ResponseTextUseCase

        segmenter.set_responding(True)
        try:
            # --- Device-directed gate ---
            candidate_tid, crops, facing = _select_candidate(start_ts - prebuffer_ms, end_ts)
            verdict = await directedness.evaluate(
                pcm16=pcm,
                sample_rate=sample_rate,
                candidate_crops=crops,
                facing_scores=facing,
                candidate_track_id=candidate_tid,
            )
            logger.info(
                "directedness: directed=%s reason=%s facing=%.2f asd=%.2f ratio=%.2f track=%s",
                verdict.directed, verdict.reason, verdict.facing_score,
                verdict.asd_score, verdict.asd_active_ratio, verdict.candidate_track_id,
            )
            if not verdict.directed:
                await _safe_send(websocket, {"event": "not_directed", "payload": {"reason": verdict.reason}})
                return

            try:
                transcript = await container.stt_service.transcribe_pcm(pcm, sample_rate)
            except ValueError as e:
                # Bad/unsupported audio (e.g. sample-rate mismatch). Tell the
                # client so it leaves "processing" and returns to listening,
                # mirroring the empty-transcript path below — otherwise the FE
                # hangs until the next utterance.
                logger.debug("STT validation failed: %s", e)
                await _safe_send(websocket, {"event": "no_speech", "payload": {}})
                return
            except Exception:
                logger.exception("STT failed")
                await _safe_send(websocket, {"event": "error", "payload": {"content": "Transcription failed"}})
                return

            if not transcript or not transcript.strip():
                logger.debug("Empty transcript — skipping pipeline")
                await _safe_send(websocket, {"event": "no_speech", "payload": {}})
                return

            transcript_payload = stream_event_to_dict(TranscriptEvent(content=transcript))
            await _safe_send(websocket, {"event": transcript_payload["type"], "payload": transcript_payload})

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
                        await _safe_send(websocket, {"event": payload.get("type"), "payload": payload})
            except asyncio.CancelledError:
                await _safe_send(websocket, {"event": "interrupted", "payload": {"content": "Reply interrupted"}})
                raise
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.exception("Session pipeline failed")
                await _safe_send(websocket, {"event": "error", "payload": {"content": "Internal server error"}})
        finally:
            segmenter.set_responding(False)

    async def _process_segment(tick) -> None:
        nonlocal pipeline_task, speech_start_ts

        if tick.event == SegmentEvent.NONE:
            return

        if tick.event == SegmentEvent.SPEECH_START:
            speech_start_ts = last_audio_ts
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
            start_ts, end_ts = speech_start_ts, last_audio_ts
            pipeline_task = asyncio.create_task(
                asyncio.wait_for(
                    _run_pipeline(tick.utterance, start_ts, end_ts),
                    timeout=settings.agent_idle_timeout_seconds * 2,
                ),
                name="session-pipeline",
            )

    # --- Audio demux: VAD inference offloaded to thread pool ---
    async def _handle_audio(payload: bytes, ts: int) -> None:
        nonlocal last_audio_ts
        last_audio_ts = ts
        audio_buffer.extend(payload)
        frame_size = segmenter.frame_bytes
        while len(audio_buffer) >= frame_size:
            frame = bytes(audio_buffer[:frame_size])
            del audio_buffer[:frame_size]
            tick = await asyncio.to_thread(segmenter.push, frame)
            await _process_segment(tick)

    # --- Video processor: latest-wins, never blocks the receive loop ---
    async def _video_processor() -> None:
        nonlocal pending_video, video_frame_id
        if vision_use_case is None:
            return
        try:
            while not stop_event.is_set():
                await new_video_event.wait()
                if stop_event.is_set():
                    break
                slot, pending_video = pending_video, None
                new_video_event.clear()
                if slot is None:
                    continue
                frame_bytes, ts = slot
                fid = video_frame_id
                video_frame_id += 1
                try:
                    results, crops = await vision_use_case.execute_with_crops(frame_bytes, fid)
                except Exception:
                    logger.exception("Vision frame processing failed")
                    continue
                _record_crops(crops, ts)
                if results:
                    await _safe_send(
                        websocket,
                        {
                            "event": "vision_update",
                            "payload": {
                                "frame_id": fid,
                                "delta": delta.diff(results),
                                "snapshot": full_snapshot(results),
                            },
                        },
                    )
        except WebSocketDisconnect:
            stop_event.set()
        except Exception:
            logger.exception("Video processor crashed")
            stop_event.set()

    async def _handle_control(text_data: str) -> None:
        nonlocal pipeline_task, conversation_id, video_frame_id
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            await _safe_send(websocket, {"event": "error", "payload": {"content": "Invalid JSON"}})
            return

        event = data.get("event")
        payload = data.get("payload") or {}
        if event == "facing":
            try:
                score = float(payload.get("score", 0.0))
                ts = int(payload.get("ts") or (time.time() * 1000))
            except (TypeError, ValueError):
                return
            facing_buffer.append((ts, max(0.0, min(1.0, score))))
        elif event == "reset":
            if pipeline_task is not None and not pipeline_task.done():
                pipeline_task.cancel()
            pipeline_task = None
            segmenter.reset()
            audio_buffer.clear()
            crop_buffers.clear()
            facing_buffer.clear()
            delta.reset()
            video_frame_id = 0
            conversation_id = None
            if vision_use_case is not None:
                await vision_use_case.reset()
            await _safe_send(websocket, {"event": "reset_ack", "payload": {}})
        elif event == "interrupt":
            if pipeline_task is not None and not pipeline_task.done():
                pipeline_task.cancel()
            await _safe_send(websocket, {"event": "interrupt_ack", "payload": {}})
        elif event == "set_conversation":
            conversation_id = payload.get("conversation_id")
            await _safe_send(websocket, {"event": "ack", "payload": {"conversation_id": conversation_id}})
        else:
            await _safe_send(websocket, {"event": "error", "payload": {"content": f"Unknown event: {event}"}})

    # --- Receive loop ---
    video_task = asyncio.create_task(_video_processor(), name="session-video")
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
            bytes_data = message.get("bytes")

            if text_data is not None:
                await _handle_control(text_data)
                continue

            if not bytes_data or len(bytes_data) < _HEADER.size:
                continue
            msg_type, ts = _HEADER.unpack_from(bytes_data, 0)
            payload = bytes_data[_HEADER.size:]
            if msg_type == _TYPE_AUDIO:
                await _handle_audio(payload, ts)
            elif msg_type == _TYPE_VIDEO:
                if payload:
                    pending_video = (payload, ts)
                    new_video_event.set()
    except WebSocketDisconnect:
        logger.info("Client disconnected from session stream")
    except Exception:
        logger.exception("Unexpected error in session stream")
    finally:
        stop_event.set()
        new_video_event.set()
        if pipeline_task is not None and not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except (asyncio.CancelledError, Exception):
                pass
        if not video_task.done():
            video_task.cancel()
            try:
                await video_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await websocket.close()
        except Exception as e:
            logger.debug("Error closing session WS: %s", e)
