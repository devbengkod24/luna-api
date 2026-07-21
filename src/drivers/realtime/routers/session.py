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
from contextlib import AsyncExitStack

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
from src.ports.streaming_stt import StreamingSTTEventType
from src.shared.constants import Character

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/session", tags=["session"])

_TYPE_AUDIO = 0x01
_TYPE_VIDEO = 0x02
_HEADER = struct.Struct(">BQ")  # type byte + uint64 ms timestamp

# How long the audio feed may stall mid-turn before the streaming path gives up
# waiting for Deepgram to end the turn for us. Both of Deepgram's turn-end
# signals need audio to keep arriving; when the client stops sending, neither
# will ever fire. Comfortably longer than a within-sentence pause, which does
# not stall the feed at all — audio flows continuously while the mic is open.
_STREAM_STALE_SECONDS = 1.5

# Client audio arrives in ~32 ms chunks. Anything beyond this between two
# consecutive capture timestamps means frames were dropped before they reached
# us, not that they arrived late.
_AUDIO_GAP_MS = 100


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


def _fps_from_timestamps(ts_list: list[int]) -> float | None:
    """Effective video fps over a window, from crop capture timestamps (ms).

    LR-ASD needs the real delivered fps to scale its MFCC window — the video
    stream is latest-wins, so the rate is ≤ the FE target and jittery. Returns
    None with <2 samples; the adapter then falls back to its nominal fps.
    """
    if len(ts_list) < 2:
        return None
    span_ms = max(ts_list) - min(ts_list)
    if span_ms <= 0:
        return None
    return (len(ts_list) - 1) * 1000.0 / span_ms


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

    # --- Turn detection: either local VAD, or Deepgram's when streaming ---
    # With STT_PROVIDER=deepgram-stream the provider owns endpointing, so no
    # local VAD model is constructed or run at all on this connection.
    streaming_stt = container.streaming_stt_service
    segmenter: VadSegmenter | None = None

    if streaming_stt is None:
        try:
            if (
                container.vad_service is not None
                and container.vad_service.sample_rate == sample_rate
            ):
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

        segmenter = VadSegmenter(
            vad=vad,
            speech_start_frames=settings.voice_speech_start_frames,
            speech_end_frames=settings.voice_speech_end_frames,
            prebuffer_frames=settings.voice_prebuffer_frames,
            max_utterance_frames=max(
                1,
                int(settings.voice_max_utterance_seconds * 1000 / vad.frame_duration_ms),
            ),
            barge_in_start_frames=settings.voice_barge_in_start_frames,
        )
    else:
        logger.info("Session WS using Deepgram streaming — local VAD bypassed")

    # Sized from config rather than the VAD instance so the video-side buffers
    # work identically on the streaming path, where no VAD object exists.
    max_utterance_frames = max(
        1, int(settings.voice_max_utterance_seconds * 1000 / settings.voice_vad_frame_ms)
    )
    prebuffer_ms = settings.voice_prebuffer_frames * settings.voice_vad_frame_ms

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
    # Mirrors segmenter.set_responding() for the streaming path, which has no
    # segmenter to hold that state.
    is_responding = False
    # Confirmed (is_final) segments of the turn in progress, awaiting a
    # speech_final or UtteranceEnd to commit. Streaming path only.
    stream_segments: list[str] = []
    # Deepgram announced speech but the turn has not been committed yet. Tracked
    # separately from `stream_segments` because a turn can open on VAD noise and
    # never produce a single word — the client has already latched on
    # `speech_start` by then, so it still needs to be told the turn is over.
    stream_turn_open = False
    # Seeded at connect so the watchdog measures the gap since the socket opened,
    # not since the epoch — otherwise it fires a KeepAlive before the first frame.
    last_audio_sent = time.monotonic()
    # Health of the audio feed for the turn in progress: how much of what the
    # person said never reached Deepgram at all.
    turn_gap_ms = 0
    turn_gap_count = 0
    turn_gap_max = 0
    audio_withheld = False
    # Why the in-flight reply was cancelled. The client mutes its mic for ~1.5 s
    # after a *hotword* interrupt to swallow the trailing "stop luna"; doing that
    # after a VAD barge-in would punch a hole in the question the person is in
    # the middle of asking, so the two must be distinguishable.
    interrupt_reason = "client"

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

    def _select_candidate(
        start_ts: int, end_ts: int
    ) -> tuple[int | None, list[Image], list[float], float | None]:
        """Pick the closest (largest-crop) track present during the utterance.

        Also returns that track's effective fps (from its crop timestamps) so the
        ASD adapter can scale its MFCC window to the real delivered video rate.
        """
        best_tid: int | None = None
        best_area = -1
        best_items: list[tuple[int, Image]] = []
        for tid, dq in crop_buffers.items():
            items = [(ts, c) for (ts, c) in dq if start_ts <= ts <= end_ts]
            if not items:
                continue
            area = max(_crop_area(c) for (_, c) in items)
            if area > best_area:
                best_area, best_tid, best_items = area, tid, items
        crops = [c for (_, c) in best_items]
        crop_fps = _fps_from_timestamps([ts for (ts, _) in best_items])
        facing = [s for (ts, s) in facing_buffer if start_ts <= ts <= end_ts]
        return best_tid, crops, facing, crop_fps

    # --- Reply pipeline (directedness gate → STT → LLM → TTS) ---

    def _set_responding(value: bool) -> None:
        """Gate new turns while Luna replies, on whichever path is active."""
        nonlocal is_responding, stream_turn_open
        is_responding = value
        if segmenter is not None:
            segmenter.set_responding(value)
        if value:
            # Anything Deepgram heard mid-reply (typically TTS leaking back
            # through the mic) must not be glued onto the next real turn.
            stream_segments.clear()
            stream_turn_open = False

    async def _reply(transcript: str) -> None:
        """transcript → LLM → TTS. Shared by the VAD path and the streaming path."""
        nonlocal conversation_id

        from src.adapters.persistence import ConversationRepository, MessageRepository
        from src.use_cases import ResponseTextUseCase

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
            await _safe_send(websocket, {
                "event": "interrupted",
                "payload": {"content": "Reply interrupted", "reason": interrupt_reason},
            })
            raise
        except WebSocketDisconnect:
            raise
        except Exception:
            logger.exception("Session pipeline failed")
            await _safe_send(websocket, {"event": "error", "payload": {"content": "Internal server error"}})

    async def _notify_pipeline_error(message: str) -> None:
        try:
            await _safe_send(websocket, {"event": "error", "payload": {"content": message}})
        except Exception:
            pass  # client already gone — nothing left to tell

    def _on_pipeline_done(task: asyncio.Task) -> None:
        """Surface failures the reply task would otherwise swallow.

        Nothing awaits `pipeline_task` on the happy path, so without this an
        exception — a `wait_for` timeout above all — would vanish into an
        unretrieved-exception warning and leave the client sitting on
        "processing" until it gave up on the whole session.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        if isinstance(exc, asyncio.TimeoutError):
            # The cancellation `wait_for` raised into `_reply` already emitted
            # `interrupted`, so the client is unstuck; this is just the log.
            logger.warning("Session pipeline timed out")
            return
        logger.error("Session pipeline task failed", exc_info=exc)
        asyncio.create_task(_notify_pipeline_error("Internal server error"))

    def _spawn_pipeline(coro) -> asyncio.Task:
        """Run a reply under a hard ceiling, with failures always reported."""
        task = asyncio.create_task(
            asyncio.wait_for(coro, timeout=settings.agent_idle_timeout_seconds * 2),
            name="session-pipeline",
        )
        task.add_done_callback(_on_pipeline_done)
        return task

    async def _run_stream_pipeline(transcript: str) -> None:
        """Streaming path: Deepgram already transcribed, so go straight to reply.

        No directedness gate here — it needs the utterance PCM for ASD, which
        this path never buffers.

        The floor is claimed by the caller, synchronously at commit time, rather
        than here — see `_commit_stream_turn`. Releasing it is still this task's
        job, so that it happens on cancellation too.
        """
        try:
            if container.db_connection is None:
                await _safe_send(websocket, {"event": "error", "payload": {"content": "Database not initialized"}})
                return
            await _reply(transcript)
        finally:
            _set_responding(False)

    async def _commit_stream_turn() -> None:
        """End the turn Deepgram has been transcribing and start the reply.

        Called from the event consumer (`speech_final` / `UtteranceEnd`) and from
        the watchdog when the audio feed stalls. Turn state is cleared before the
        first await, so a second caller racing in is a no-op rather than a
        duplicate reply.
        """
        nonlocal pipeline_task, stream_turn_open
        nonlocal turn_gap_ms, turn_gap_count, turn_gap_max

        if not stream_turn_open and not stream_segments:
            return
        text = " ".join(s for s in stream_segments if s).strip()
        stream_segments.clear()
        stream_turn_open = False

        # One line per turn: how intact the audio behind this transcript was. A
        # healthy feed reports gaps=0. Anything else means the transcript is of
        # audio with pieces missing, which reads as mis-recognition or a
        # half-swallowed sentence rather than as an obvious dropout.
        logger.info(
            "turn committed: chars=%d gaps=%d missing=%dms longest=%dms",
            len(text), turn_gap_count, turn_gap_ms, turn_gap_max,
        )
        turn_gap_ms = turn_gap_count = turn_gap_max = 0

        if not text:
            # Deepgram's VAD opened a turn on something that produced no words —
            # noise, a cough, TTS leaking back through the mic. The client
            # latched "listening" on `speech_start`, so it needs an explicit way
            # back out, exactly as the VAD path gets from `_run_pipeline`.
            await _safe_send(websocket, {"event": "no_speech", "payload": {}})
            return

        # Claim the floor before awaiting anything: Deepgram keeps emitting
        # events for this same utterance while the task is still being scheduled,
        # and those must not open a second turn on top of this one.
        _set_responding(True)
        await _safe_send(websocket, {"event": "speech_end", "payload": {}})
        pipeline_task = _spawn_pipeline(_run_stream_pipeline(text))

    async def _run_pipeline(pcm: bytes, start_ts: int, end_ts: int) -> None:
        if container.db_connection is None:
            await _safe_send(websocket, {"event": "error", "payload": {"content": "Database not initialized"}})
            return

        _set_responding(True)
        try:
            # --- Device-directed gate ---
            candidate_tid, crops, facing, crop_fps = _select_candidate(
                start_ts - prebuffer_ms, end_ts
            )
            verdict = await directedness.evaluate(
                pcm16=pcm,
                sample_rate=sample_rate,
                candidate_crops=crops,
                facing_scores=facing,
                candidate_track_id=candidate_tid,
                crop_fps=crop_fps,
            )
            logger.info(
                "directedness: directed=%s reason=%s facing=%.2f asd=%.2f ratio=%.2f track=%s fps=%s",
                verdict.directed, verdict.reason, verdict.facing_score,
                verdict.asd_score, verdict.asd_active_ratio, verdict.candidate_track_id,
                f"{crop_fps:.1f}" if crop_fps else "n/a",
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

            await _reply(transcript)
        finally:
            _set_responding(False)

    async def _process_segment(tick) -> None:
        nonlocal pipeline_task, speech_start_ts, interrupt_reason

        if tick.event == SegmentEvent.NONE:
            return

        if tick.event == SegmentEvent.SPEECH_START:
            speech_start_ts = last_audio_ts
            if pipeline_task is not None and not pipeline_task.done():
                interrupt_reason = "barge_in"
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
            pipeline_task = _spawn_pipeline(
                _run_pipeline(tick.utterance, start_ts, end_ts)
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

    # --- Streaming path: forward audio verbatim, no local framing or VAD ---
    async def _handle_audio_streaming(payload: bytes, ts: int, session) -> None:
        """Forward mic audio to Deepgram, and watch the feed for holes.

        A streaming recogniser has no way to know that audio it never received
        existed: drop 300 ms out of the middle of a sentence and it splices the
        two halves into one continuous-sounding utterance and transcribes the
        result. That is silent corruption, so the size of every hole is measured
        here from the client's own capture timestamps and reported per turn.

        Holes come from the client's mic gates (engagement, interrupt cooldown),
        from this router withholding audio mid-reply, or from the network.
        """
        nonlocal last_audio_ts, last_audio_sent
        nonlocal turn_gap_ms, turn_gap_count, turn_gap_max, audio_withheld

        gap = ts - last_audio_ts if last_audio_ts else 0
        last_audio_ts = ts

        # With barge-in off, the consumer discards everything heard mid-reply, so
        # forwarding it only pays Deepgram to transcribe Luna's own voice. Skip
        # the send instead; the watchdog holds the socket open meanwhile.
        if is_responding and not settings.deepgram_stream_barge_in:
            audio_withheld = True
            return

        # A hole that this router opened on purpose is not a client-side gap, and
        # the first frame after one would otherwise be blamed for it.
        if audio_withheld:
            audio_withheld = False
        elif gap > _AUDIO_GAP_MS:
            turn_gap_ms += gap
            turn_gap_count += 1
            turn_gap_max = max(turn_gap_max, gap)

        try:
            await session.send_audio(payload)
            last_audio_sent = time.monotonic()
        except Exception:
            logger.exception("Failed to forward audio to Deepgram")

    async def _stream_watchdog(session) -> None:
        """Hold the socket open through silence, and rescue stalled turns.

        Two jobs on one timer:

        1. **KeepAlive.** Deepgram drops an idle connection after ~10 s.
           KeepAlive is a control message, not audio, so it costs nothing —
           sending silence instead would be billed.
        2. **Flush a turn whose audio stopped arriving.** Deepgram's endpointing
           only fires when it *hears* silence, and `utterance_end_ms` only
           advances while audio keeps coming. So when the client stops feeding
           mid-turn — engagement gate closing, interrupt cooldown, mic off, or
           this router itself withholding audio — neither turn-end signal ever
           arrives and the turn would hang open forever.
        """
        last_keepalive = time.monotonic()
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.25)
                now = time.monotonic()
                idle = now - last_audio_sent
                if idle >= _STREAM_STALE_SECONDS and (stream_turn_open or stream_segments):
                    logger.debug("Audio feed stalled %.1fs mid-turn — flushing", idle)
                    await _commit_stream_turn()
                if idle >= 5.0 and now - last_keepalive >= 3.0:
                    await session.keep_alive()
                    last_keepalive = now
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Deepgram watchdog stopped: %s", e)

    async def _stream_consumer(session) -> None:
        """Drive the reply pipeline from Deepgram events instead of local VAD."""
        nonlocal pipeline_task, speech_start_ts, interrupt_reason, stream_turn_open

        try:
            async for ev in session.events():
                # Without local VAD, the mic keeps feeding Deepgram while Luna
                # speaks, so her own TTS comes back as "speech". When barge-in is
                # off, drop everything heard mid-reply rather than answering it.
                if is_responding and not settings.deepgram_stream_barge_in:
                    continue

                if ev.type is StreamingSTTEventType.SPEECH_STARTED:
                    speech_start_ts = last_audio_ts
                    stream_turn_open = True
                    if pipeline_task is not None and not pipeline_task.done():
                        interrupt_reason = "barge_in"
                        pipeline_task.cancel()
                        try:
                            await pipeline_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        pipeline_task = None
                    await _safe_send(websocket, {"event": "speech_start", "payload": {}})

                elif ev.type is StreamingSTTEventType.INTERIM:
                    await _safe_send(
                        websocket,
                        {"event": "partial_transcript", "payload": {"content": ev.transcript}},
                    )

                elif ev.type is StreamingSTTEventType.FINAL:
                    stream_segments.append(ev.transcript)
                    if ev.speech_final:
                        await _commit_stream_turn()

                elif ev.type is StreamingSTTEventType.UTTERANCE_END:
                    # Backstop for when room noise kept Deepgram's audio VAD from
                    # ever seeing silence, so speech_final never arrived. A no-op
                    # if speech_final already committed this turn.
                    await _commit_stream_turn()
        except WebSocketDisconnect:
            stop_event.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Deepgram stream consumer crashed")
            await _safe_send(websocket, {"event": "error", "payload": {"content": "Transcription stream failed"}})
            stop_event.set()

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
        nonlocal pipeline_task, conversation_id, video_frame_id, interrupt_reason
        nonlocal stream_turn_open
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
            if segmenter is not None:
                segmenter.reset()
            stream_segments.clear()
            stream_turn_open = False
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
                interrupt_reason = "client"
                pipeline_task.cancel()
            await _safe_send(websocket, {"event": "interrupt_ack", "payload": {}})
        elif event == "set_conversation":
            conversation_id = payload.get("conversation_id")
            await _safe_send(websocket, {"event": "ack", "payload": {"conversation_id": conversation_id}})
        else:
            await _safe_send(websocket, {"event": "error", "payload": {"content": f"Unknown event: {event}"}})

    # --- Receive loop ---
    video_task = asyncio.create_task(_video_processor(), name="session-video")
    stream_tasks: list[asyncio.Task] = []
    stack = AsyncExitStack()
    stt_session = None
    try:
        if streaming_stt is not None:
            try:
                stt_session = await stack.enter_async_context(streaming_stt.open(sample_rate))
            except Exception:
                logger.exception("Failed to open Deepgram stream")
                await _safe_send(
                    websocket,
                    {"event": "error", "payload": {"content": "Transcription stream unavailable"}},
                )
                await websocket.close(code=1011)
                return
            stream_tasks = [
                asyncio.create_task(_stream_consumer(stt_session), name="session-dg-consumer"),
                asyncio.create_task(_stream_watchdog(stt_session), name="session-dg-watchdog"),
            ]

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
                if stt_session is not None:
                    await _handle_audio_streaming(payload, ts, stt_session)
                else:
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
        for task in stream_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        try:
            await stack.aclose()
        except Exception as e:
            logger.debug("Error closing Deepgram stream: %s", e)
        try:
            await websocket.close()
        except Exception as e:
            logger.debug("Error closing session WS: %s", e)
