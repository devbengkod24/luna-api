"""Vision WebSocket router."""
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any

# from src.use_cases.vision_stream import VisionStreamUseCase

from src.drivers import dependencies
router = APIRouter(prefix="/v1/vision", tags=["vision"])


# This will be injected in main.py
# vision_use_case: VisionStreamUseCase = None


# def set_vision_use_case(use_case: VisionStreamUseCase):
#     """Set the vision use case (called from main.py)."""
#     global vision_use_case
#     vision_use_case = use_case


async def safe_send_json(websocket: WebSocket, data: dict):
    """Safely send JSON to WebSocket, handling disconnects gracefully."""
    try:
        await websocket.send_json(data)
    except (WebSocketDisconnect, RuntimeError) as e:
        # Client already disconnected, silently ignore
        print(f"Cannot send message, client disconnected: {e}")
        raise WebSocketDisconnect()


@router.websocket("/stream")
async def vision_stream(websocket: WebSocket):
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

    try:
        vision_use_case = dependencies.get_vision_stream_use_case()
    except RuntimeError:
        await safe_send_json(websocket, {
            "event": "error",
            "message": "Vision service not initialized",
        })
        await websocket.close()
        return
    
    frame_id = 0
    
    try:
        while True:
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                
                if data.get("event") == "webcam_frame":
                    frame_base64 = data.get("data")
                    
                    if not frame_base64:
                        await safe_send_json(websocket, {
                            "event": "error",
                            "message": "Missing frame data",
                        })
                        continue
        
                    results = await vision_use_case.execute(frame_base64, frame_id)
                    frame_id += 1
                    
                    await safe_send_json(websocket, {
                        "event": "vision_update",
                        "payload": {
                            "results": [result.to_dict() for result in results],
                            "frame_id": frame_id - 1
                        }
                    })
                
                elif data.get("event") == "reset":
                    frame_id = 0
                    await safe_send_json(websocket, {
                        "event": "reset_ack",
                        "message": "Frame counter reset"
                    })
                  
                else:
                    await safe_send_json(websocket, {
                        "event": "error",
                        "message": f"Unknown event: {data.get('event')}",
                    })
            
            except json.JSONDecodeError:
                await safe_send_json(websocket, {
                    "event": "error",
                    "message": "Invalid JSON",
                })
            except WebSocketDisconnect:
                raise
            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                await safe_send_json(websocket, {
                    "event": "error",
                    "message": str(e),
                })
    
    except WebSocketDisconnect:
        print("Client disconnected from vision stream")
