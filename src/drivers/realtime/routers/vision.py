"""Vision WebSocket router with frame skip, delta updates, msgpack, and adaptive FPS."""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Callable

import msgpack
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from src.config import settings
from src.drivers import dependencies
from src.drivers.realtime.presenters import VisionDeltaCalculator, full_snapshot
from src.use_cases import VisionUseCase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["vision"])


_VALID_FORMATS = {"json", "msgpack"}


def _make_serializer(fmt: str) -> tuple[Callable[[dict], bytes], bool]:
    """Return (encode_fn, is_binary) for the selected wire format."""
    if fmt == "msgpack":
        return (lambda payload: msgpack.packb(payload, use_bin_type=True), True)
    return (lambda payload: json.dumps(payload).encode("utf-8"), False)


async def _send(websocket: WebSocket, payload: dict, encode, binary: bool) -> None:
    data = encode(payload)
    try:
        if binary:
            await websocket.send_bytes(data)
        else:
            await websocket.send_text(data.decode("utf-8"))
    except (WebSocketDisconnect, RuntimeError, ConnectionError) as e:
        logger.debug("Cannot send message, client disconnected: %s", e)
        raise WebSocketDisconnect() from e


class _ThrottleAdvisor:
    """Rolling-window latency tracker that emits adaptive-fps suggestions."""

    def __init__(
        self,
        window: int,
        target_latency_ms: float,
        min_fps: int,
        max_fps: int,
        emit_interval: float,
    ) -> None:
        self.samples: deque[float] = deque(maxlen=window)
        self.target_latency_ms = target_latency_ms
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.emit_interval = emit_interval
        self._last_suggested: int | None = None
        self._last_emit: float = 0.0

    def record(self, latency_ms: float) -> None:
        self.samples.append(latency_ms)

    def suggestion(self) -> int | None:
        if len(self.samples) < self.samples.maxlen:
            return None
        avg = sum(self.samples) / len(self.samples)
        suggested = self.max_fps
        if avg > self.target_latency_ms:
            suggested = max(self.min_fps, int(1000.0 / avg))
        suggested = min(self.max_fps, max(self.min_fps, suggested))

        now = time.monotonic()
        if suggested == self._last_suggested and (now - self._last_emit) < self.emit_interval:
            return None
        self._last_suggested = suggested
        self._last_emit = now
        return suggested


@router.websocket("/stream")
async def vision_stream(
    websocket: WebSocket,
    use_case: VisionUseCase = Depends(dependencies.get_vision_use_case),
):
    """WebSocket endpoint for real-time vision analysis.

    Query params:
      - `fmt`: `json` (default) or `msgpack` for response encoding.

    Client → server: binary JPEG/PNG frames, or JSON control messages.
    Server → client: `vision_update` (delta), `throttle`, `reset_ack`, `ping`, `error`.
    """
    await websocket.accept()
    logger.info("Client connected to vision stream")

    fmt = (websocket.query_params.get("fmt") or "json").lower()
    if fmt not in _VALID_FORMATS:
        fmt = "json"
    encode, binary = _make_serializer(fmt)
    logger.debug("vision stream: format=%s", fmt)

    delta = VisionDeltaCalculator()
    advisor = _ThrottleAdvisor(
        window=settings.vision_latency_window,
        target_latency_ms=settings.vision_target_latency_ms,
        min_fps=settings.vision_throttle_min_fps,
        max_fps=settings.vision_throttle_max_fps,
        emit_interval=settings.vision_throttle_emit_interval,
    )

    # Latest-wins frame slot. Receiver overwrites pending; processor consumes.
    pending_frame: bytes | None = None
    new_frame_event = asyncio.Event()
    stop_event = asyncio.Event()
    frames_dropped = 0
    frame_id = 0

    async def _process_frame(frame_bytes: bytes, fid: int) -> None:
        nonlocal frames_dropped
        started = time.perf_counter()
        try:
            results = await use_case.execute(frame_bytes, fid)
        except ValueError as e:
            logger.warning("Invalid frame: %s", e)
            await _send(
                websocket,
                {"event": "error", "payload": {"content": "Invalid frame data"}},
                encode,
                binary,
            )
            return
        except Exception:
            logger.exception("Error processing frame")
            await _send(
                websocket,
                {"event": "error", "payload": {"content": "Internal server error processing frame"}},
                encode,
                binary,
            )
            return
        latency_ms = (time.perf_counter() - started) * 1000.0
        advisor.record(latency_ms)

        payload = {
            "event": "vision_update",
            "payload": {
                "frame_id": fid,
                "delta": delta.diff(results),
                "snapshot": full_snapshot(results),
                "latency_ms": round(latency_ms, 2),
                "dropped": frames_dropped,
            },
        }
        frames_dropped = 0
        await _send(websocket, payload, encode, binary)

        suggested = advisor.suggestion()
        if suggested is not None:
            await _send(
                websocket,
                {"event": "throttle", "payload": {"target_fps": suggested}},
                encode,
                binary,
            )

    async def _receiver() -> None:
        nonlocal pending_frame, frames_dropped, frame_id
        try:
            while not stop_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=settings.frame_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    await _send(
                        websocket,
                        {"event": "ping", "payload": {}},
                        encode,
                        binary,
                    )
                    continue

                msg_type = message.get("type")
                if msg_type == "websocket.disconnect":
                    raise WebSocketDisconnect()

                text_data = message.get("text")
                bytes_data = message.get("bytes")

                if text_data is not None:
                    await _handle_control(text_data)
                elif bytes_data is not None:
                    if not bytes_data:
                        await _send(
                            websocket,
                            {"event": "error", "payload": {"content": "Missing binary frame data"}},
                            encode,
                            binary,
                        )
                        continue
                    if pending_frame is not None:
                        frames_dropped += 1
                    pending_frame = bytes_data
                    new_frame_event.set()
                else:
                    await _send(
                        websocket,
                        {"event": "error", "payload": {"content": "Unsupported WebSocket message type"}},
                        encode,
                        binary,
                    )
        except WebSocketDisconnect:
            stop_event.set()
            new_frame_event.set()
        except Exception:
            logger.exception("Receiver loop crashed")
            stop_event.set()
            new_frame_event.set()

    async def _handle_control(text_data: str) -> None:
        nonlocal frame_id
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON received: %s", e)
            await _send(
                websocket,
                {"event": "error", "payload": {"content": "Invalid JSON format"}},
                encode,
                binary,
            )
            return

        event = data.get("event")
        if event == "reset":
            frame_id = 0
            delta.reset()
            await use_case.reset()
            await _send(
                websocket,
                {"event": "reset_ack", "payload": {}},
                encode,
                binary,
            )
        else:
            await _send(
                websocket,
                {"event": "error", "payload": {"content": f"Unknown event: {event}"}},
                encode,
                binary,
            )

    async def _processor() -> None:
        nonlocal pending_frame, frame_id
        try:
            while not stop_event.is_set():
                await new_frame_event.wait()
                if stop_event.is_set():
                    break
                frame, pending_frame = pending_frame, None
                new_frame_event.clear()
                if frame is None:
                    continue
                fid = frame_id
                frame_id += 1
                await _process_frame(frame, fid)
        except WebSocketDisconnect:
            stop_event.set()
        except Exception:
            logger.exception("Processor loop crashed")
            stop_event.set()

    receiver_task = asyncio.create_task(_receiver(), name="vision-receiver")
    processor_task = asyncio.create_task(_processor(), name="vision-processor")

    try:
        await asyncio.gather(receiver_task, processor_task)
    except WebSocketDisconnect:
        logger.info("Client disconnected from vision stream")
    except Exception:
        logger.exception("Unexpected error in vision stream")
    finally:
        stop_event.set()
        new_frame_event.set()
        for task in (receiver_task, processor_task):
            if not task.done():
                task.cancel()
        for task in (receiver_task, processor_task):
            try:
                await task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            except Exception:
                logger.debug("Task ended with error", exc_info=True)
        try:
            await websocket.close()
        except Exception as e:
            logger.debug("Error closing WebSocket: %s", e)
