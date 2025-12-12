"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.drivers import dependencies
from src.drivers.rest.routers import chat, face, document
from src.drivers.realtime.routers import vision


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    print("Starting up...")

    try:
        # Initialize dependencies
        dependencies.initialize_dependencies(
            database_url=settings.database_url,
            redis_url=settings.redis_url,
            face_recognition_model=settings.face_recognition_model,
            gpu_device=settings.gpu_device,
            enable_face_recognition=settings.enable_face_recognition,
            llm_model=settings.llm_model,
        )
        
        # Initialize database tables (optional - use alembic in production)
        # await dependencies.db_connection.create_tables()
        
        # Initialize Qdrant collections
        await dependencies.rag_adapter.initialize()
        
        # Initialize face store collection
        if settings.init_face_embedding:
            await dependencies.initialize_face_embedding(run_seed=True)
        else:
            await dependencies.initialize_face_embedding(run_seed=False)
        
        # Initialize vision use case for WebSocket
        # vision_use_case = dependencies.get_vision_stream_use_case()
        # vision.set_vision_use_case(vision_use_case)
        try:
            if settings.enable_face_recognition:
                # Test that vision use case can be created
                _ = dependencies.get_vision_stream_use_case()
                print("Vision streaming ready")
        except RuntimeError as e:
            print(f"Warning: Vision streaming not available: {e}")

        print("Startup complete")

    except Exception as e:
        print(f"Fatal error during startup: {e}")
        raise
        
    yield
    
    # Shutdown
    print("Shutting down...")
    
    try:
        # Close connections
        if dependencies.db_connection:
            await dependencies.db_connection.close()
        if dependencies.cache_adapter:
            await dependencies.cache_adapter.close()
    except Exception as e:
        print(f"Error during shutdown: {e}")
    
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
app.include_router(document.router)
app.include_router(vision.router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
    }