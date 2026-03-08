"""Response REST routers — text and speech AI response endpoints."""
import json
from typing import AsyncIterator
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.drivers import dependencies
from src.drivers.rest.schemas import (
    ResponseTextRequest,
    ResponseAudioRequest,
)
from src.use_cases import (
    ResponseSpeechUseCase,
    ResponseTextUseCase,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/response", tags=["response"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


@router.post("/text")
async def response_text(
    request: ResponseTextRequest,
    use_case: ResponseTextUseCase = Depends(dependencies.get_response_text_use_case),
):
    """Handle text-based request with streaming AI response."""
    logger.info(
        f"Response text request - Conversation ID: {request.conversation_id}, "
        f"User input length: {len(request.user_input)}, "
        f"Character: {request.character}, "
        f"Emotion: {request.context.emotion if request.context else 'None'}, "
        f"Name: {request.context.name if request.context else 'None'}, "
        f"NIM: {request.context.nim if request.context else 'None'}"
    )

    async def event_stream() -> AsyncIterator[str]:
        async for event in use_case.execute(
            request.conversation_id, request.user_input, request.character, request.context
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/audio")
async def response_audio(
    request: ResponseAudioRequest,
    use_case: ResponseSpeechUseCase = Depends(dependencies.get_response_speech_use_case),
):
    """Handle audio-based request with streaming AI response."""
    logger.info(
        f"Response audio request - Conversation ID: {request.conversation_id}, "
        f"Audio size: {len(request.audio_input)} bytes, "
        f"Character: {request.character}, "
        f"TTS enabled: {request.enable_tts}, "
        f"Emotion: {request.context.emotion if request.context else 'None'}, "
        f"Name: {request.context.name if request.context else 'None'}, "
        f"NIM: {request.context.nim if request.context else 'None'}"
    )

    async def event_stream() -> AsyncIterator[str]:
        async for event in use_case.execute(
            request.conversation_id, request.audio_input, request.character, request.context, request.enable_tts
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)
