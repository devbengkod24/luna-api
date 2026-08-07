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
from collections import Counter, deque
from contextlib import aclosing

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
        barge_in_window_frames=(
            settings.voice_barge_in_window_frames
            if settings.voice_barge_in_mode.strip().lower() == "vad"
            else 0
        ),
        barge_in_ratio=settings.voice_barge_in_ratio,
        barge_in_settle_frames=settings.voice_barge_in_settle_frames,
    )
    prebuffer_ms = settings.voice_prebuffer_frames * vad.frame_duration_ms
    _, _barge_needed, _ = segmenter.barge_in_stats
    logger.info(
        "Voice barge-in: mode=%s window=%s frames (~%d ms), need %d speech frames; "
        "utterance ends after %d ms of silence",
        settings.voice_barge_in_mode,
        segmenter.barge_in_window_frames or "off",
        segmenter.barge_in_window_frames * vad.frame_duration_ms,
        _barge_needed,
        settings.voice_speech_end_frames * vad.frame_duration_ms,
    )

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
    # Per-track (ts, name, nim, emotion) so the reply can be personalised.
    vision_buffers: dict[int, deque[tuple[int, str, str | None, str]]] = {}
    facing_buffer: deque[tuple[int, float]] = deque(maxlen=2000)
    last_audio_ts = 0
    speech_start_ts = 0

    audio_buffer = bytearray()
    pipeline_task: asyncio.Task | None = None
    # Bumped per reply. The playback-release watchdog carries the generation it
    # was armed for, so a stale timer can never re-open the mic on a newer reply.
    respond_gen = 0
    playback_timer: asyncio.Task | None = None
    # Mic frames actually received since barge-in was armed. Separates "the user
    # said nothing" from "the client stopped sending" — the client's engagement
    # pre-gate is pose-based and used to mute the mic mid-sentence, which reads
    # as silence here and is indistinguishable from it without this counter.
    frames_since_arm = 0
    # Why the in-flight reply was cancelled, forwarded to the client on the
    # `interrupted` event. "hotword" needs a mic cooldown there (the trailing
    # "stop luna" would otherwise be transcribed as the next question);
    # "barge_in" must NOT cool down — the user is mid-sentence and we want to
    # keep hearing them.
    cancel_reason = "hotword"

    # latest-wins video slot
    pending_video: tuple[bytes, int] | None = None
    new_video_event = asyncio.Event()
    stop_event = asyncio.Event()
    video_frame_id = 0

    def _record_vision(results: list[VisionResult], ts: int) -> None:
        """Buffer per-track identity + emotion alongside the crops.

        The vision pipeline already computes these for `vision_update`; without
        this they were sent to the client and then dropped, so the voice path
        always handed the LLM an empty UserContext ("Unknown" / "neutral") even
        though the camera had recognised the person.
        """
        for r in results:
            if r.track_id is None:
                continue
            dq = vision_buffers.get(r.track_id)
            if dq is None:
                dq = deque(maxlen=max_utterance_frames)
                vision_buffers[r.track_id] = dq
            dq.append((ts, r.name, r.nim, r.emotion))
        cutoff = ts - int(settings.voice_max_utterance_seconds * 1000) - 2000
        for tid in [t for t, dq in list(vision_buffers.items()) if dq and dq[-1][0] < cutoff]:
            vision_buffers.pop(tid, None)

    def _context_for(track_id: int | None, start_ts: int, end_ts: int) -> UserContext:
        """Identity + emotion for the track that spoke, over the utterance window.

        Uses the MODE rather than the last sample: per-frame emotion is jittery
        and a single mis-recognition shouldn't decide who Luna thinks she is
        talking to. Falls back to an empty context (prompt renders
        "Unknown"/"neutral") whenever there is nothing usable — the reply is
        still generated, just without personalisation.
        """
        if track_id is None:
            return UserContext()
        samples = [
            (name, nim, emotion)
            for (ts, name, nim, emotion) in vision_buffers.get(track_id, ())
            if start_ts <= ts <= end_ts
        ]
        if not samples:
            return UserContext()
        # Identity is voted as a (name, nim) pair so the two can never disagree.
        identities = Counter(
            (n, nim) for (n, nim, _) in samples if n and n != "Unknown"
        )
        emotions = Counter(e for (_, _, e) in samples if e)
        name, nim = identities.most_common(1)[0][0] if identities else (None, None)
        emotion = emotions.most_common(1)[0][0] if emotions else None
        return UserContext(name=name, nim=nim, emotion=emotion)

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

    # --- Listening gate: closed for as long as the speaker is actually busy ---

    def _cancel_playback_timer() -> None:
        nonlocal playback_timer
        if playback_timer is not None and not playback_timer.done():
            playback_timer.cancel()
        playback_timer = None

    def _open_mic() -> None:
        """Re-open the listening gate now (playback finished, or nothing to play)."""
        _cancel_playback_timer()
        segmenter.set_responding(False)

    async def _release_after_grace(gen: int) -> None:
        try:
            await asyncio.sleep(settings.voice_playback_grace_seconds)
        except asyncio.CancelledError:
            return
        if gen == respond_gen:
            logger.warning(
                "No playback_done from client after %.0fs — re-opening the mic anyway.",
                settings.voice_playback_grace_seconds,
            )
            segmenter.set_responding(False)

    # --- Reply pipeline (directedness gate → STT → LLM → TTS) ---

    async def _run_pipeline(pcm: bytes, start_ts: int, end_ts: int) -> None:
        nonlocal conversation_id, respond_gen, playback_timer, frames_since_arm

        if container.db_connection is None:
            await _safe_send(websocket, {"event": "error", "payload": {"content": "Database not initialized"}})
            return

        from src.adapters.persistence import ConversationRepository, MessageRepository
        from src.use_cases import ResponseTextUseCase

        respond_gen += 1
        gen = respond_gen
        _cancel_playback_timer()
        segmenter.set_responding(True)
        sent_audio = False
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

            user_context = _context_for(candidate_tid, start_ts - prebuffer_ms, end_ts)
            # DEBUG, never INFO: name / NIM / emotion are PII (see CLAUDE.md).
            logger.debug(
                "user context: track=%s recognised=%s emotion=%s",
                candidate_tid, user_context.name is not None, user_context.emotion,
            )

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
                    )
                    # execute() spawns the LLM producer and the TTS worker as
                    # tasks and cancels them in its own `finally`. Cancelling
                    # this task unwinds the consumer, leaving the generator
                    # suspended at its `yield`; its cleanup then rides on
                    # garbage collection. That happened to be prompt in testing,
                    # so this is belt-and-braces rather than a proven fix — but
                    # relying on GC timing to stop a TTS stream is not something
                    # to leave to chance.
                    async with aclosing(use_case.execute(
                        conversation_id=conversation_id,
                        user_input=transcript,
                        character=character,
                        context=user_context,
                        enable_tts=settings.enable_tts and container.tts_service is not None,
                    )) as stream:
                        async for event in stream:
                            payload = stream_event_to_dict(event)
                            ptype = payload.get("type")
                            if ptype == "done" and payload.get("conversation_id"):
                                conversation_id = payload["conversation_id"]
                            if ptype in ("audio", "audio_chunk"):
                                if not sent_audio:
                                    # First audio out: Luna is now audibly
                                    # speaking, so the user has something to talk
                                    # over. Before this the pipeline was silently
                                    # thinking and any trigger would just restart it.
                                    segmenter.arm_barge_in()
                                    frames_since_arm = 0
                                sent_audio = True
                            await _safe_send(websocket, {"event": ptype, "payload": payload})
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.exception("Session pipeline failed")
                await _safe_send(websocket, {"event": "error", "payload": {"content": "Internal server error"}})
        finally:
            # How close the user got to interrupting. If barge-in "never works"
            # on the real kiosk, this says whether the mic heard them at all
            # (peak near 0 → echo canceller is eating the near-end talker) or
            # whether the ratio is simply set too high (peak just under need).
            peak, need, armed = segmenter.barge_in_stats
            if armed and need:
                logger.info(
                    "barge-in watch: peak %d/%d speech frames, %d mic frames received (%s)",
                    peak, need, frames_since_arm,
                    "triggered" if peak >= need else "never triggered",
                )
            # The reply's audio is generated far faster than it plays, so the
            # stream ending says nothing about the speaker being quiet. Keep the
            # gate closed until the client reports playback_done — otherwise the
            # kiosk spends the gap transcribing its own voice. An accepted
            # barge-in has already re-opened the gate; this only re-arms the
            # watchdog, which is generation-guarded and harmless.
            if sent_audio and gen == respond_gen and segmenter.is_responding:
                playback_timer = asyncio.create_task(
                    _release_after_grace(gen), name="playback-grace"
                )
            elif gen == respond_gen:
                _open_mic()

    def _facing_allows_barge_in() -> bool:
        """Fuse the audio trigger with the facing signal.

        Someone chatting to a friend beside the kiosk should not cancel the
        reply. Fails OPEN: with no facing samples in the window (camera off, or
        voice-only mode) barge-in proceeds on audio alone, so the feature never
        silently stops working.
        """
        if not facing_buffer:
            return True
        cutoff = last_audio_ts - settings.voice_barge_in_facing_window_ms
        recent = [s for (ts, s) in facing_buffer if ts >= cutoff]
        if not recent:
            return True
        mean_facing = sum(recent) / len(recent)
        allowed = mean_facing >= settings.facing_threshold
        if not allowed:
            logger.info(
                "barge-in rejected: facing=%.2f < %.2f (n=%d)",
                mean_facing, settings.facing_threshold, len(recent),
            )
        return allowed

    async def _cancel_pipeline(reason: str) -> None:
        """Cancel the in-flight reply and tell the client to stop playing.

        The `interrupted` event is sent from here, not from inside the cancelled
        coroutine: a task that is being cancelled is a poor place to do I/O, and
        if that send is skipped the client never stops its audio and plays the
        abandoned reply to the end. Sent only when there was something to cancel.
        """
        nonlocal pipeline_task, cancel_reason
        cancel_reason = reason
        if pipeline_task is None or pipeline_task.done():
            pipeline_task = None
            return
        pipeline_task.cancel()
        try:
            await pipeline_task
        except (asyncio.CancelledError, Exception):
            pass
        pipeline_task = None
        await _safe_send(websocket, {
            "event": "interrupted",
            "payload": {"content": "Reply interrupted", "reason": reason},
        })

    async def _process_segment(tick) -> None:
        # `pipeline_task` MUST stay in this nonlocal list: the SPEECH_END branch
        # assigns it, and without the declaration Python makes it a local, the
        # enclosing one stays None, and _cancel_pipeline finds nothing to cancel
        # — barge-in then detects fine but never stops the reply.
        nonlocal speech_start_ts, pipeline_task

        if tick.event == SegmentEvent.NONE:
            return

        if tick.event == SegmentEvent.BARGE_IN:
            if not _facing_allows_barge_in():
                return
            logger.info("barge-in accepted — cancelling in-flight reply")
            # Re-open the mic and throw away everything captured while the
            # speaker was playing; the settle window inside the segmenter drops
            # the next few frames too. The user's sentence is picked up fresh by
            # normal VAD, so what reaches Whisper is clean and contiguous.
            segmenter.accept_barge_in()
            _cancel_playback_timer()
            await _cancel_pipeline("barge_in")
            # No speech_start here — no utterance has begun yet. The client
            # moves to "listening" off the interrupted(reason=barge_in) event,
            # and the real speech_start follows when the user's continuing
            # speech clears the normal VAD threshold.
            return

        if tick.event == SegmentEvent.SPEECH_START:
            speech_start_ts = last_audio_ts
            # Same semantics as a barge-in for the client: the user is talking,
            # so the mic must keep flowing rather than enter a cooldown.
            await _cancel_pipeline("barge_in")
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
        nonlocal last_audio_ts, frames_since_arm
        last_audio_ts = ts
        audio_buffer.extend(payload)
        frame_size = segmenter.frame_bytes
        while len(audio_buffer) >= frame_size:
            frame = bytes(audio_buffer[:frame_size])
            del audio_buffer[:frame_size]
            frames_since_arm += 1
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
                _record_vision(results, ts)
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
        nonlocal conversation_id, video_frame_id
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
            await _cancel_pipeline("reset")
            segmenter.reset()
            # segmenter.reset() leaves the responding gate alone; without this a
            # reset issued mid-reply would leave the mic closed for good.
            _open_mic()
            audio_buffer.clear()
            crop_buffers.clear()
            vision_buffers.clear()
            facing_buffer.clear()
            delta.reset()
            video_frame_id = 0
            conversation_id = None
            if vision_use_case is not None:
                await vision_use_case.reset()
            await _safe_send(websocket, {"event": "reset_ack", "payload": {}})
        elif event == "playback_done":
            # The client's audio queue has drained — the speaker is finally
            # quiet, so it is safe to listen again without hearing ourselves.
            _open_mic()
        elif event == "interrupt":
            # Client-side barge-in (hotword / stop button). It has already
            # stopped playback locally, so nothing is coming out of the speaker
            # — but the trailing "stop luna" is still in the air, hence the
            # client-side cooldown that "hotword" triggers.
            await _cancel_pipeline("hotword")
            _open_mic()
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
