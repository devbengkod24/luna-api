"""Vision WebSocket router."""
import json
import logging
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from src.use_cases import VisionUseCase
from src.drivers import dependencies
from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["vision"])


async def safe_send_json(websocket: WebSocket, data: dict) -> None:
    """Safely send JSON to WebSocket, handling disconnects gracefully."""
    try:
        await websocket.send_json(data)
    except (WebSocketDisconnect, RuntimeError, ConnectionError) as e:
        # Client already disconnected, log and re-raise
        logger.debug(f"Cannot send message, client disconnected: {e}")
        raise WebSocketDisconnect() from e
    except Exception as e:
        # Unexpected error, log it
        logger.error(f"Unexpected error sending WebSocket message: {e}", exc_info=True)
        raise

@router.websocket("/stream")
async def vision_stream(
    websocket: WebSocket,
    use_case: VisionUseCase = Depends(dependencies.get_vision_use_case),
):
    """WebSocket endpoint for real-time vision analysis."""
    await websocket.accept()
    logger.info("Client connected to vision stream")

    frame_id = 0

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=settings.frame_timeout_seconds,
                )
            except asyncio.TimeoutError:
                # Send ping to check if client is still alive
                try:
                    await safe_send_json(
                        websocket,
                        {
                            "event": "ping",
                            "message": "Connection check",
                        },
                    )
                except WebSocketDisconnect:
                    raise
                continue

            try:
                msg_type = message.get("type")
                if msg_type == "websocket.disconnect":
                    raise WebSocketDisconnect()

                text_data = message.get("text")
                bytes_data = message.get("bytes")

                # --- Text JSON path (control messages only) ---
                if text_data is not None:
                    try:
                        data = json.loads(text_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON received: {e}")
                        await safe_send_json(
                            websocket,
                            {
                                "event": "error",
                                "message": "Invalid JSON format",
                            },
                        )
                        continue

                    event = data.get("event")

                    if event == "reset":
                        frame_id = 0
                        use_case.reset()
                        await safe_send_json(
                            websocket,
                            {
                                "event": "reset_ack",
                                "message": "Frame counter reset",
                            },
                        )

                    else:
                        await safe_send_json(
                            websocket,
                            {
                                "event": "error",
                                "message": f"Unknown event: {event}",
                            },
                        )

                # --- Binary path (raw JPEG/PNG frame) ---
                elif bytes_data is not None:
                    if not bytes_data:
                        await safe_send_json(
                            websocket,
                            {
                                "event": "error",
                                "message": "Missing binary frame data",
                            },
                        )
                        continue

                    try:
                        results = await use_case.execute(bytes_data, frame_id)
                        frame_id += 1

                        await safe_send_json(
                            websocket,
                            {
                                "event": "vision_update",
                                "payload": {
                                    "results": [
                                        result.to_dict() for result in results
                                    ],
                                    "frame_id": frame_id - 1,
                                },
                            },
                        )
                    except ValueError as e:
                        logger.warning(f"Invalid binary frame data: {e}")
                        await safe_send_json(
                            websocket,
                            {
                                "event": "error",
                                "message": "Invalid frame data",
                            },
                        )
                    except Exception as e:
                        logger.error(
                            f"Error processing binary frame: {e}", exc_info=True
                        )
                        await safe_send_json(
                            websocket,
                            {
                                "event": "error",
                                "message": "Internal server error processing frame",
                            },
                        )

                else:
                    await safe_send_json(
                        websocket,
                        {
                            "event": "error",
                            "message": "Unsupported WebSocket message type",
                        },
                    )

            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error(
                    f"Unexpected error in message processing: {e}", exc_info=True
                )
                try:
                    await safe_send_json(
                        websocket,
                        {
                            "event": "error",
                            "message": "Internal server error",
                        },
                    )
                except WebSocketDisconnect:
                    raise

    except WebSocketDisconnect:
        logger.info("Client disconnected from vision stream")
    except Exception as e:
        logger.error(
            f"Unexpected error in WebSocket connection: {e}", exc_info=True
        )
    finally:
        # Cleanup resources if needed
        try:
            await websocket.close()
        except Exception:
            pass