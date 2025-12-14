"""Application configuration."""

import os
from typing import ClassVar
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    app_name: str = "VSA-BE-FastAPI-v2"
    app_version: str = "1.0.0"

    # Database
    postgres_host: str
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: str

    @property
    def database_url(self) -> str:
        """Build async database URL for application."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """Build sync database URL for migrations."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Qdrant
    qdrant_host: str
    qdrant_grpc_port: int = 6334
    qdrant_api_key: str
    qdrant_collection_name: str = "vsa_collection"

    # LLM
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "embeddinggemma:latest"
    reranker_model: str = "jina-reranker-v3-Q4_K_M.gguf"
    llm_temperature: float = 0.7
    
    # Ollama
    ollama_host: str = "ollama"
    ollama_port: int = 11434

    # Face Recognition (InsightFace)
    enable_face_recognition: bool = True
    face_recognition_model: str = (
        "buffalo_sc"  # SCRFD detector + ArcFace recognition
    )
    gpu_device: int = -1  # -1 for CPU, 0+ for GPU
    face_detection_size: int = 320  # Detection size (320 faster, 640 more accurate)
    face_similarity_threshold: float = (
        0.5  # Cosine similarity threshold (0.4-0.6 recommended)
    )
    
    # ByteTrack Tracker
    tracker_max_age: int = 60  # Max frames to keep track alive without detection
    tracker_min_hits: int = 3  # Min detections before track is confirmed
    tracker_min_score: float = 0.4  # Min confidence to initialize track
    
    # Emotion Detection
    enable_emotion_detection: bool = True
    seed_faces: bool = False

    # Rate limiting constants
    frame_timeout_seconds: float = 5.0

    # Vision
    emotion_detection_interval: int = 5  # Frames between emotion detection
    recognition_retry_interval: float = 2.0  # Seconds between recognition retries
    max_recognition_attempts: int = 5  # Max attempts per track


    # Storage paths
    base_dir: ClassVar[str] = os.path.dirname(os.path.dirname(__file__))
    media_base_dir: str = os.path.join(base_dir, "storage")
    faces_dir: str = os.path.join(media_base_dir, "faces")
    rag_storage_path: str = os.path.join(media_base_dir, "rag", "documents")
    rag_metadata_path: str = os.path.join(media_base_dir, "rag", "metadata")
    rag_docs_dir: str = os.path.join(media_base_dir, "rag", "documents")  # Deprecated, use rag_storage_path

    # CORS
    cors_origins: str = "*"


# Global settings instance
settings = Settings()
