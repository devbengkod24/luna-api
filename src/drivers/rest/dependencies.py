"""Dependency injection for FastAPI."""
from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.database.connection import DatabaseConnection
from src.adapters.database.repository import ChatRepository
from src.adapters.llm.llm_adapter import LLMAdapter
from src.adapters.qdrant.rag_adapter import QdrantRAGAdapter
from src.adapters.speech.stt_adapter import STTAdapter
from src.adapters.redis.cache_adapter import RedisCacheAdapter
from src.adapters.vision.vision_adapter import VisionAdapter
from src.use_cases.chat_text import ChatTextUseCase
from src.use_cases.chat_speech import ChatSpeechUseCase
from src.use_cases.chat_history import GetChatHistoryUseCase, GetChatSessionUseCase
from src.use_cases.vision_stream import VisionStreamUseCase


# Global instances (initialized in main.py)
db_connection: DatabaseConnection = None
llm_adapter: LLMAdapter = None
rag_adapter: QdrantRAGAdapter = None
stt_adapter: STTAdapter = None
cache_adapter: RedisCacheAdapter = None
vision_adapter: VisionAdapter = None


def initialize_dependencies(
    database_url: str,
    redis_url: str,
    llm_api_key: str = None,
    stt_api_key: str = None,
):
    """Initialize global dependencies.
    
    This should be called once at application startup.
    """
    global db_connection, llm_adapter, rag_adapter, stt_adapter, cache_adapter, vision_adapter
    
    db_connection = DatabaseConnection(database_url)
    llm_adapter = LLMAdapter(api_key=llm_api_key)
    rag_adapter = QdrantRAGAdapter()
    stt_adapter = STTAdapter(api_key=stt_api_key)
    cache_adapter = RedisCacheAdapter(url=redis_url)
    vision_adapter = VisionAdapter()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async for session in db_connection.get_session():
        yield session


async def get_chat_repository(
    session: AsyncSession = None,
) -> ChatRepository:
    """Get chat repository."""
    if session is None:
        async for session in get_db_session():
            return ChatRepository(session)
    return ChatRepository(session)


def get_llm_adapter() -> LLMAdapter:
    """Get LLM adapter."""
    return llm_adapter


def get_rag_adapter() -> QdrantRAGAdapter:
    """Get RAG adapter."""
    return rag_adapter


def get_stt_adapter() -> STTAdapter:
    """Get STT adapter."""
    return stt_adapter


def get_cache_adapter() -> RedisCacheAdapter:
    """Get cache adapter."""
    return cache_adapter


def get_vision_adapter() -> VisionAdapter:
    """Get vision adapter."""
    return vision_adapter


# Use case factories
async def get_chat_text_use_case(
    session: AsyncSession = Depends(get_db_session)
) -> ChatTextUseCase:
    """Get chat text use case."""
    repository = ChatRepository(session)
    return ChatTextUseCase(repository, llm_adapter, rag_adapter)


async def get_chat_speech_use_case(
    session: AsyncSession = Depends(get_db_session)
) -> ChatSpeechUseCase:
    """Get chat speech use case."""
    repository = ChatRepository(session)
    return ChatSpeechUseCase(repository, llm_adapter, stt_adapter)


async def get_history_use_case(
    session: AsyncSession = Depends(get_db_session)
) -> GetChatHistoryUseCase:
    """Get chat history use case."""
    repository = ChatRepository(session)
    return GetChatHistoryUseCase(repository)


async def get_session_use_case(
    session: AsyncSession = Depends(get_db_session)
) -> GetChatSessionUseCase:
    """Get chat session use case."""
    repository = ChatRepository(session)
    return GetChatSessionUseCase(repository)


def get_vision_stream_use_case() -> VisionStreamUseCase:
    """Get vision stream use case."""
    return VisionStreamUseCase(vision_adapter, cache_adapter)

