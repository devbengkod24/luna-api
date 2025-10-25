"""Vision WebSocket router."""
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any

from src.use_cases.vision_stream import VisionStreamUseCase

router = APIRouter(prefix="/v1/vision", tags=["vision"])


# This will be injected in main.py
vision_use_case: VisionStreamUseCase = None


def set_vision_use_case(use_case: VisionStreamUseCase):
    """Set the vision use case (called from main.py)."""
    global vision_use_case
    vision_use_case = use_case


@router.websocket("/stream")
async def vision_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time vision analysis.
    
    Protocol:
    - Client sends: {"event": "webcam_frame", "data": "base64-encoded-jpeg"}
    - Server responds: {
        "event": "vision_update",
        "payload": {
          "faces": [{"name": "User_ABC", "box": [x, y, w, h], "confidence": 0.95}],
          "emotions": [{"emotion": "happy", "box": [x, y, w, h], "confidence": 0.8}]
        }
      }
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                
                if data.get("event") == "webcam_frame":
                    frame_base64 = data.get("data")
                    
                    if not frame_base64:
                        await websocket.send_json({
                            "event": "error",
                            "message": "Missing frame data",
                        })
                        continue
                    
                    # Process frame
                    if vision_use_case:
                        result = await vision_use_case.execute(frame_base64)
                        
                        if result:
                            # Send result back to client with new format
                            await websocket.send_json({
                                "event": "vision_update",
                                "payload": result.to_dict(),
                            })
                        else:
                            # No face detected
                            await websocket.send_json({
                                "event": "vision_update",
                                "payload": {
                                    "faces": [],
                                    "emotions": []
                                },
                            })
                    else:
                        await websocket.send_json({
                            "event": "error",
                            "message": "Vision service not initialized",
                        })
                else:
                    await websocket.send_json({
                        "event": "error",
                        "message": f"Unknown event: {data.get('event')}",
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "event": "error",
                    "message": "Invalid JSON",
                })
            except Exception as e:
                await websocket.send_json({
                    "event": "error",
                    "message": str(e),
                })
    
    except WebSocketDisconnect:
        print("Client disconnected from vision stream")

