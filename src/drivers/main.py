"""FastAPI application entry point."""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response

from src.config import settings
from src.drivers import dependencies
from src.drivers.middleware import BodySizeLimitMiddleware
from src.drivers.realtime.routers import turn
from src.drivers.rest.routers import conversation
from src.drivers.rest.routers import face
from src.drivers.rest.routers import file as file_router
from src.drivers.rest.routers import photo
from src.drivers.rest.routers import response
from src.drivers.security import require_api_token, warn_if_open
from src.shared.exceptions import LunaException

logger = logging.getLogger(__name__)

# Libraries that are chatty at INFO/DEBUG and drown out application logs.
# httpx is deliberately NOT here — one line per Ollama/Exa call is the cheapest
# latency trace we have.
_NOISY_LOGGERS = (
    "httpcore",
    "urllib3",
    "watchfiles",
    "filelock",
    "fsspec",
    "PIL",
    "asyncio",
)


def configure_logging() -> None:
    """Attach a handler to the root logger so `logger.info()` in src/ is emitted.

    Uvicorn configures only its own `uvicorn.*` loggers (with propagate=False),
    leaving the root logger bare — so without this every INFO log in the app is
    dropped by logging's last-resort handler, which passes WARNING and above
    only. Called at import time, after uvicorn has loaded its own config, so the
    two don't clobber each other and uvicorn's access logs are not duplicated.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the application container and run async initializers."""
    logger.info("Starting up...")
    warn_if_open()

    try:
        container = dependencies.initialize_container()
        app.state.container = container

        if container.rag_service is not None:
            await container.rag_service.initialize()
        if container.face_service is not None:
            await container.face_service.initialize()

        logger.info("Startup complete")
    except Exception as e:
        logger.critical("Fatal error during startup: %s", e, exc_info=True)
        raise

    yield

    logger.info("Shutting down...")

    try:
        container = getattr(app.state, "container", None)
        if container is not None:
            if container.db_connection:
                await container.db_connection.close()
            if container.rag_service:
                await container.rag_service.close()
            if container.face_service:
                await container.face_service.close()
            close_stt = getattr(container.stt_service, "close", None)
            if close_stt is not None:
                await close_stt()
    except Exception as e:
        logger.error("Error during shutdown: %s", e, exc_info=True)

    logger.info("Shutdown complete")


def _luna_exception_handler(request, exc: LunaException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


def _unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    """Generic fallback so infra error messages never reach the client."""
    request_id = getattr(getattr(request, "state", None), "request_id", None)
    logger.exception("Unhandled exception (request_id=%s)", request_id)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)
app.add_exception_handler(LunaException, _luna_exception_handler)
app.add_exception_handler(Exception, _unhandled_exception_handler)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    header_name = settings.request_id_header
    request_id = request.headers.get(header_name) or str(uuid.uuid4())
    request.state.request_id = request_id
    response_obj = await call_next(request)
    response_obj.headers[header_name] = request_id
    return response_obj


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=("*" not in settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)
# Outermost, so an oversized body is refused before CORS, routing or parsing.
app.add_middleware(BodySizeLimitMiddleware)

# Every REST route needs the API token (when one is set). The turn socket checks
# it itself, before accepting: a WebSocket refused by a dependency would be
# accepted and closed instead, which a kiosk reads as the cloud being down.
_guarded = [Depends(require_api_token)]
for router in (response.router, conversation.router, file_router.router, face.router, photo.router):
    app.include_router(router, prefix=settings.api_v1_prefix, dependencies=_guarded)
app.include_router(turn.router, prefix=settings.api_v1_prefix)


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
async def ready(request: Request):
    container = getattr(request.app.state, "container", None)
    if container is None:
        return {"status": "starting", "db_ready": False, "rag_ready": False}
    rag_ready = container.rag_service is not None and container.rag_service.is_ready
    db_ready = container.db_connection is not None
    return {"status": "ok", "db_ready": db_ready, "rag_ready": rag_ready}


@app.get("/", include_in_schema=False)
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api": settings.api_v1_prefix,
    }
