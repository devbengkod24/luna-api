"""Vision WebSocket router."""
import json
import logging
import base64
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from src.use_cases.vision import VisionUseCase
from src.drivers import dependencies
from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/vision", tags=["vision"])

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

def validate_base64(data: str) -> bool:
    """Validate if string is valid base64."""
    try:
        if not data or not isinstance(data, str):
            return False
        # Try to decode
        base64.b64decode(data, validate=True)
        return True
    except Exception:
        return False

@router.websocket("/stream")
async def vision_stream(websocket: WebSocket, use_case: VisionUseCase = Depends(dependencies.get_vision_use_case)):
    """WebSocket endpoint for real-time vision analysis.
    
    Protocol:
    - Client sends: {"event": "webcam_frame", "data": "base64-encoded-jpeg"}
    - Server responds: {
        "event": "vision_update",
        "payload": {
          "results": [
            {
              "name": "User_ABC",
              "nim": "123456",
              "emotion": "happy",
              "box": [x, y, w, h],
              "confidence": {"face": 0.95, "emotion": 0.8},
              "face_id": "uuid",
              "track_id": 1
            }
          ]
        }
      }
    """
    await websocket.accept()
    logger.info("Client connected to vision stream")
    
    frame_id = 0
    
    try:
        while True:
            # Receive message with timeout
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=settings.frame_timeout_seconds
                )
            except asyncio.TimeoutError:
                # Send ping to check if client is still alive
                try:
                    await safe_send_json(websocket, {
                        "event": "ping",
                        "message": "Connection check"
                    })
                except WebSocketDisconnect:
                    raise
                continue
            
            try:
                data = json.loads(message)
                event = data.get("event")
                
                if event == "webcam_frame":
                    frame_base64 = data.get("data")
                    
                    if not frame_base64:
                        await safe_send_json(websocket, {
                            "event": "error",
                            "message": "Missing frame data",
                        })
                        continue
                    
                    # Validate base64
                    if not validate_base64(frame_base64):
                        await safe_send_json(websocket, {
                            "event": "error",
                            "message": "Invalid base64 data",
                        })
                        continue
                    
                    # Process frame
                    try:
                        results = await use_case.execute(frame_base64, frame_id)
                        frame_id += 1
                        
                        await safe_send_json(websocket, {
                            "event": "vision_update",
                            "payload": {
                                "results": [result.to_dict() for result in results],
                                "frame_id": frame_id - 1
                            }
                        })
                    except ValueError as e:
                        logger.warning(f"Invalid frame data: {e}")
                        await safe_send_json(websocket, {
                            "event": "error",
                            "message": "Invalid frame data",
                        })
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}", exc_info=True)
                        await safe_send_json(websocket, {
                            "event": "error",
                            "message": "Internal server error processing frame",
                        })
                
                elif event == "reset":
                    frame_id = 0
                    await safe_send_json(websocket, {
                        "event": "reset_ack",
                        "message": "Frame counter reset"
                    })
                
                else:
                    await safe_send_json(websocket, {
                        "event": "error",
                        "message": f"Unknown event: {event}",
                    })
            
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received: {e}")
                await safe_send_json(websocket, {
                    "event": "error",
                    "message": "Invalid JSON format",
                })
            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error(f"Unexpected error in message processing: {e}", exc_info=True)
                try:
                    await safe_send_json(websocket, {
                        "event": "error",
                        "message": "Internal server error",
                    })
                except WebSocketDisconnect:
                    raise
    
    except WebSocketDisconnect:
        logger.info("Client disconnected from vision stream")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection: {e}", exc_info=True)
    finally:
        # Cleanup resources if needed
        try:
            await websocket.close()
        except Exception:
            pass