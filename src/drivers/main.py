"""FastAPI application entry point."""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response

from src.config import settings
from src.drivers import dependencies
from src.drivers.realtime.routers import session
from src.drivers.rest.routers import conversation
from src.drivers.rest.routers import face
from src.drivers.rest.routers import file as file_router
from src.drivers.rest.routers import response
from src.shared.exceptions import VSAException

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the application container and run async initializers."""
    logger.info("Starting up...")

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
    except Exception as e:
        logger.error("Error during shutdown: %s", e, exc_info=True)

    logger.info("Shutdown complete")


def _vsa_exception_handler(request, exc: VSAException) -> JSONResponse:
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
app.add_exception_handler(VSAException, _vsa_exception_handler)
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
app.include_router(response.router, prefix=settings.api_v1_prefix)
app.include_router(conversation.router, prefix=settings.api_v1_prefix)
app.include_router(file_router.router, prefix=settings.api_v1_prefix)
app.include_router(face.router, prefix=settings.api_v1_prefix)
app.include_router(session.router, prefix=settings.api_v1_prefix)


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
