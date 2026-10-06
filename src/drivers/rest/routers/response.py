"""Response REST routers — text and speech AI response endpoints."""
import json
import logging
import asyncio
from contextlib import aclosing
from typing import AsyncIterator, Callable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from src.config import settings
from src.domain.entities import UserContext
from src.domain.events import DoneEvent, ErrorEvent, StreamEvent
from src.drivers import dependencies
from src.drivers.dependencies import AppContainer
from src.drivers.rest.presenters import stream_event_to_dict
from src.drivers.rest.schemas import (
    ResponseTextRequest,
    ResponseAudioRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/response", tags=["response"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _sse(event: StreamEvent) -> str:
    return f"data: {json.dumps(stream_event_to_dict(event))}\n\n"


async def _reply_stream(
    container: AppContainer,
    start: Callable[[AppContainer, object], AsyncIterator[StreamEvent]],
) -> AsyncIterator[str]:
    """Run one reply inside its own database session and stream it as SSE.

    The session is opened here, inside the stream, rather than by a request
    dependency: the reply outlives the handler. `done` is held back until the
    session has committed. It used to be sent first, so a page that reacted to
    it by reading or continuing the conversation could find it not there yet.

    `aclosing` closes the use case deterministically on client disconnect. It
    spawns the LLM producer and the TTS worker as tasks and cancels them only in
    its own `finally`; without an explicit aclose() that cleanup rides on
    garbage collection, and a reply nobody can hear keeps generating.
    """
    done: DoneEvent | None = None
    try:
        async with dependencies.db_session(container) as session:
            async with aclosing(start(container, session)) as source:
                while True:
                    try:
                        async with asyncio.timeout(settings.sse_idle_timeout_seconds):
                            event = await anext(source)
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        yield _sse(ErrorEvent(content="Stream timeout"))
                        return
                    if isinstance(event, DoneEvent):
                        done = event
                        continue
                    yield _sse(event)
    except asyncio.CancelledError:
        # client disconnected
        return
    except Exception:
        # Raised out of a StreamingResponse, an error cuts the stream off with
        # no event at all, and the page waits for a reply that already failed.
        logger.exception("Reply stream failed")
        yield _sse(ErrorEvent(content="Internal server error"))
        return
    if done is not None:
        yield _sse(done)


@router.post("/text")
async def response_text(
    http_request: Request,
    request: ResponseTextRequest,
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
    # Resolved here, after the body has validated: as a dependency its failure
    # would pre-empt a 422 with a 503 or a 500.
    container = dependencies.require_response_services(http_request)
    context = UserContext(**request.context.model_dump()) if request.context is not None else None

    def start(container, session):
        return dependencies.build_response_text_use_case(container, session).execute(
            request.conversation_id,
            request.user_input,
            request.character,
            context,
            request.enable_tts,
        )

    return StreamingResponse(
        _reply_stream(container, start), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.post("/audio")
async def response_audio(
    http_request: Request,
    request: ResponseAudioRequest,
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
    # Resolved here, after the body has validated: as a dependency its failure
    # would pre-empt a 422 with a 503 or a 500.
    container = dependencies.require_response_services(http_request)
    context = UserContext(**request.context.model_dump()) if request.context is not None else None

    def start(container, session):
        return dependencies.build_response_speech_use_case(container, session).execute(
            request.conversation_id,
            request.audio_input,
            request.character,
            context,
            request.enable_tts,
        )

    return StreamingResponse(
        _reply_stream(container, start), media_type="text/event-stream", headers=SSE_HEADERS
    )
