"""Response REST routers — text and speech AI response endpoints."""
import json
from typing import AsyncIterator
import logging
import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from src.drivers import dependencies
from src.domain.entities import UserContext
from src.config import settings
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

async def _sse_event_stream_with_idle_timeout(source: AsyncIterator[dict]) -> AsyncIterator[str]:
    """Wrap event generator with idle timeout to avoid hanging streams."""
    try:
        while True:
            try:
                async with asyncio.timeout(settings.sse_idle_timeout_seconds):
                    event = await anext(source)
            except StopAsyncIteration:
                return
            except TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Stream timeout'})}\n\n"
                return
            yield f"data: {json.dumps(event)}\n\n"
    except asyncio.CancelledError:
        # client disconnected
        return


@router.post("/text")
async def response_text(
    http_request: Request,
    request: ResponseTextRequest,
    use_case: ResponseTextUseCase = Depends(dependencies.get_response_text_use_case),
):
    """Handle text-based request with streaming AI response."""
    logger.info(
        "Response text request - request_id=%s, conversation_id=%s, input_len=%d, character=%s, tts=%s",
        getattr(http_request.state, "request_id", None),
        request.conversation_id,
        len(request.user_input),
        request.character,
        request.enable_tts,
    )

    async def event_stream() -> AsyncIterator[str]:
        context = (
            UserContext(**request.context.model_dump()) if request.context is not None else None
        )
        source = use_case.execute(
            request.conversation_id,
            request.user_input,
            request.character,
            context,
            request.enable_tts,
        )
        async for payload in _sse_event_stream_with_idle_timeout(source):
            yield payload

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/audio")
async def response_audio(
    http_request: Request,
    request: ResponseAudioRequest,
    use_case: ResponseSpeechUseCase = Depends(dependencies.get_response_speech_use_case),
):
    """Handle audio-based request with streaming AI response."""
    logger.info(
        "Response audio request - request_id=%s, conversation_id=%s, audio_bytes=%d, character=%s, tts=%s",
        getattr(http_request.state, "request_id", None),
        request.conversation_id,
        len(request.audio_input),
        request.character,
        request.enable_tts,
    )

    async def event_stream() -> AsyncIterator[str]:
        context = (
            UserContext(**request.context.model_dump()) if request.context is not None else None
        )
        source = use_case.execute(
            request.conversation_id,
            request.audio_input,
            request.character,
            context,
            request.enable_tts,
        )
        async for payload in _sse_event_stream_with_idle_timeout(source):
            yield payload

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)
