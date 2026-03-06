"""Application configuration."""

from functools import lru_cache
from typing import Set
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="VSA-BE-FastAPI-v2", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")

    # Database
    postgres_host: str
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: str
    database_pool_size: int = Field(default=20, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, alias="DATABASE_MAX_OVERFLOW")


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
    knowledge_collection_name: str = "knowledge_collection"
    faces_collection_name: str = "faces_collection"

    # Qdrant
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")
    knowledge_collection_name: str = Field(
        default="knowledge_collection", alias="QDRANT_COLLECTION_NAME"
    )
    faces_collection_name: str = Field(default="faces_collection", alias="FACES_COLLECTION_NAME")

    # MinIO
    minio_endpoint: str = Field(..., alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(..., alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(..., alias="MINIO_SECRET_KEY")
    minio_bucket_name: str = Field(default="luna-files", alias="MINIO_BUCKET_NAME")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_llm_model: str = Field(default="llama3.1:8b", alias="OLLAMA_LLM_MODEL")
    ollama_embedding_model: str = Field(
        default="embeddinggemma:300m", alias="OLLAMA_EMBEDDING_MODEL"
    )
    ollama_temperature: float = Field(default=0.4, alias="OLLAMA_TEMPERATURE")
    ollama_top_p: float = Field(default=0.9, alias="OLLAMA_TOP_P")
    ollama_max_tokens: int = Field(default=1024, alias="OLLAMA_MAX_TOKENS")

    gpu_device: int = Field(default=-1, alias="GPU_DEVICE") # -1 for CPU, 0+ for GPU
    
    # Face Recognition (InsightFace)
    enable_face_recognition: bool = Field(default=True, alias="ENABLE_FACE_RECOGNITION")
    face_recognition_model: str = Field(default="buffalo_sc", alias="FACE_RECOGNITION_MODEL")
    face_detection_size: int = Field(default=320, alias="FACE_DETECTION_SIZE") # Detection size (320 faster, 640 more accurate)
    face_similarity_threshold: float = Field(default=0.5, alias="FACE_SIMILARITY_THRESHOLD") # Cosine similarity threshold (0.4-0.6 recommended)

    faces_dir: str = "storage/faces"
    face_max_batch_size: int = 100  # Max faces per request
    face_max_image_size: int = 10 * 1024 * 1024  # 10MB
    faces_allowed_extensions: Set[str] = {'.jpg', '.jpeg', '.png'}

    rag_dir: str = "storage/rag"
    rag_max_file_size: int = 50 * 1024 * 1024  # 50MB
    rag_allowed_extensions: Set[str] = {'.pdf', '.txt', '.docx', '.doc'}

    # Chat
    chat_max_prompt_length: int = 10000
    chat_max_audio_base64_size: int = 10 * 1024 * 1024  # 10MB
    
    # ByteTrack Tracker
    tracker_max_age: int = 60  # Max frames to keep track alive without detection
    tracker_min_hits: int = 3  # Min detections before track is confirmed
    tracker_min_score: float = 0.4  # Min confidence to initialize track
    
    # Emotion Detection
    enable_emotion_detection: bool = True
    seed_faces: bool = False
    seed_knowledge: bool = False
    
    # Speech-to-Text (Faster Whisper)
    whisper_model: str = "large"  # tiny, base, xsmall, medium, large-v1, large-v2, large-v3
    
    # Text-to-Speech (Facebook MMS-TTS Indonesian)
    voice_clone_dir: str = "storage/voice"  # Directory for future voice cloning (currently unused)

    # Rate limiting constants
    frame_timeout_seconds: float = 5.0

    # Vision
    emotion_detection_interval: int = 5  # Frames between emotion detection
    recognition_retry_interval: float = 2.0  # Seconds between recognition retries
    max_recognition_attempts: int = 5  # Max attempts per track

    # CORS
    cors_origins: str = "*"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

settings = get_settings()
