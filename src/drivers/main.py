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
from src.shared.exceptions import VSAException
from src.drivers import dependencies
from src.drivers.rest.routers import response, face, file as file_router, conversation
from src.drivers.realtime.routers import vision

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting up...")

    try:
        dependencies.initialize_dependencies()
        await dependencies.initialize_rag_async()
        await dependencies.initialize_face_async()

        logger.info("Startup complete")

    except Exception as e:
        logger.critical("Fatal error during startup: %s", e, exc_info=True)
        raise

    yield

    logger.info("Shutting down...")

    try:
        if dependencies.db_connection:
            await dependencies.db_connection.close()
        if dependencies.rag_service:
            await dependencies.rag_service.close()
    except Exception as e:
        logger.error("Error during shutdown: %s", e, exc_info=True)

    logger.info("Shutdown complete")


def _vsa_exception_handler(request, exc: VSAException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)
app.add_exception_handler(VSAException, _vsa_exception_handler)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    header_name = settings.request_id_header
    request_id = request.headers.get(header_name) or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers[header_name] = request_id
    return response

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
app.include_router(vision.router, prefix=settings.api_v1_prefix)

@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
async def ready():
    rag_ready = dependencies.rag_service is not None and dependencies.rag_service.is_ready
    db_ready = dependencies.db_connection is not None
    return {"status": "ok", "db_ready": db_ready, "rag_ready": rag_ready}

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api": settings.api_v1_prefix,
    }