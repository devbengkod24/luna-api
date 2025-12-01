"""Application configuration."""
import os
from pydantic import Field
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

    # Redis
    redis_host: str
    redis_port: int = 6379
    redis_password: str

    @property
    def redis_url(self) -> str:
        """Build Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    qdrant_host: str
    qdrant_grpc_port: int = 6334
    qdrant_api_key: str
    qdrant_collection_name: str = "vsa_collection"
    
    # LLM
    llm_model: str = "Supa-AI/llama3-8b-cpt-sahabatai-v1-instruct:q3_k_s"
    embedding_model: str = "bge-m3:latest"
    
    # CORS
    cors_origins: str = "*"

# Global settings instance
settings = Settings()

