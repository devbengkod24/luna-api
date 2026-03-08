"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.shared.exceptions import VSAException
from src.drivers import dependencies
from src.drivers.rest.routers import response, face, file as file_router, conversation
from src.drivers.realtime.routers import vision


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("Starting up...")

    try:
        # Initialize dependencies
        dependencies.initialize_dependencies()
        await dependencies.initialize_rag_async()
        await dependencies.initialize_face_async()

        print("Startup complete")

    except Exception as e:
        print(f"Fatal error during startup: {e}")
        raise

    yield

    print("Shutting down...")

    try:
        if dependencies.db_connection:
            await dependencies.db_connection.close()
    except Exception as e:
        print(f"Error during shutdown: {e}")

    print("Shutdown complete")


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(response.router, prefix=settings.api_v1_prefix)
app.include_router(conversation.router, prefix=settings.api_v1_prefix)
app.include_router(file_router.router, prefix=settings.api_v1_prefix)
app.include_router(face.router, prefix=settings.api_v1_prefix)
app.include_router(vision.router, prefix=settings.api_v1_prefix)

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api": settings.api_v1_prefix,
    }