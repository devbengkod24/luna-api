"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.drivers.rest import dependencies
from src.drivers.rest.routers import chat, face
from src.drivers.realtime.routers import vision


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    print("Starting up...")
    
    # Initialize dependencies
    dependencies.initialize_dependencies(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
        enable_face_recognition=settings.enable_face_recognition,
    )
    
    # Initialize database tables (optional - use alembic in production)
    # await dependencies.db_connection.create_tables()
    
    # Initialize Qdrant collections
    await dependencies.rag_adapter.initialize()
    
    # Initialize face store collection
    await dependencies.initialize_face_store()
    
    # Initialize vision use case for WebSocket
    vision_use_case = dependencies.get_vision_stream_use_case()
    vision.set_vision_use_case(vision_use_case)
    
    print("Startup complete")
    
    yield
    
    # Shutdown
    print("Shutting down...")
    
    # Close connections
    await dependencies.db_connection.close()
    await dependencies.cache_adapter.close()
    
    print("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(face.router)
app.include_router(vision.router)

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "VSA Backend API",
        "version": settings.app_version,
        "docs": "/docs",
    }