"""Application configuration."""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "VSA-BE-FastAPI"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/vsa_db"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_base"
    
    # LLM
    llm_api_key: str = ""
    llm_model: str = "gpt-3.5-turbo"
    
    # STT
    stt_api_key: str = ""
    
    # CORS
    cors_origins: list[str] = ["*"]
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

