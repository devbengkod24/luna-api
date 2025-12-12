"""Dependency injection for FastAPI."""
from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.postgres.connection import DatabaseConnection
from src.adapters.postgres.chat_repository import ChatRepository
from src.adapters.ollama.llm_adapter import LLMAdapter
from src.adapters.qdrant.rag_adapter import RAGAdapter
from src.adapters.qdrant.face_store import QdrantFaceStore
from src.adapters.filesystem.document_storage import FileSystemDocumentStorage
from src.adapters.speech.stt_adapter import STTAdapter
from src.adapters.redis.cache_adapter import RedisCacheAdapter
from src.adapters.insightface.face_recognition_adapter import FaceRecognitionAdapter
from src.adapters.bytetrack.tracker_adapter import TrackerAdapter
from src.adapters.langchain.agent_adapter import AgentAdapter
from src.adapters.huggingface.emotion_adapter import EmotionDetectionAdapter
from src.config import settings
from src.use_cases.chat_text import ChatTextUseCase
from src.use_cases.chat_speech import ChatSpeechUseCase
from src.use_cases.chat_history import GetChatHistoryUseCase, GetChatSessionUseCase
from src.use_cases.vision_stream import VisionStreamUseCase
from src.use_cases.faces import (
    RegisterFacesUseCase,
    GetFacesUseCase,
    DeleteFaceUseCase,
)
from src.use_cases.document_management import (
    UploadDocumentUseCase,
    ListDocumentsUseCase,
    DeleteDocumentUseCase,
)
# Global instances (initialized in main.py)
db_connection: DatabaseConnection | None = None
llm_adapter: LLMAdapter | None = None
rag_adapter: RAGAdapter | None = None
document_storage: FileSystemDocumentStorage | None = None
stt_adapter: STTAdapter | None = None
cache_adapter: RedisCacheAdapter | None = None
face_recognition_adapter: FaceRecognitionAdapter | None = None
face_store: QdrantFaceStore | None = None
tracker_adapter: TrackerAdapter | None = None
emotion_detection_adapter: EmotionDetectionAdapter | None = None
agent_adapter: AgentAdapter | None = None


def initialize_dependencies(
    database_url: str,
    redis_url: str,
    face_recognition_model: str,
    gpu_device: int,
    enable_face_recognition: bool,
    llm_model: str,
) -> None:
    """Initialize global dependencies.
    
    This should be called once at application startup.
    
    Args:
        database_url: PostgreSQL connection URL
        redis_url: Redis connection URL
        face_recognition_model: InsightFace model name
        face_recognition_ctx_id: GPU context ID (-1 for CPU)
        enable_face_recognition: Whether to enable face recognition
        llm_model: Ollama model name

    Returns:
        None
    """
    global db_connection, llm_adapter, rag_adapter, document_storage
    global stt_adapter, cache_adapter, vision_adapter
    global face_recognition_adapter, face_store, tracker_adapter
    global emotion_detection_adapter, agent_adapter

    try:
        db_connection = DatabaseConnection(url=database_url)
        llm_adapter = LLMAdapter(model=llm_model)
        rag_adapter = RAGAdapter()
        # document_storage = FileSystemDocumentStorage()
        stt_adapter = STTAdapter()
        cache_adapter = RedisCacheAdapter(url=redis_url)
        agent_adapter = AgentAdapter(
            llm_adapter=llm_adapter,
            rag_adapter=rag_adapter,
        )
        
        
        if enable_face_recognition:
            try:
                face_recognition_adapter = FaceRecognitionAdapter(
                    model_name=face_recognition_model,
                    ctx_id=gpu_device,
                )
                face_store = QdrantFaceStore()
                
                tracker_adapter = TrackerAdapter(
                    max_age=settings.tracker_max_age,
                    min_hits=settings.tracker_min_hits,
                    init_tracker_min_score=settings.tracker_min_score,
                )
                
                if settings.enable_emotion_detection:
                    emotion_detection_adapter = EmotionDetectionAdapter(
                    )
                else:
                    emotion_detection_adapter = None
                    print("Emotion detection disabled")
              
                print("Face recognition initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize face recognition: {e}")
        else:
            print("Face recognition disabled")

    except Exception as e:
        print(f"Error initializing dependencies: {e}")
        raise

async def initialize_face_embedding(run_seed: bool = False):
    if not face_store:
        raise RuntimeError("Face store not initialized")
    await face_store.initialize()

    if run_seed:
        if not face_recognition_adapter:
            raise RuntimeError("Face recognition not initialized")
        use_case = RegisterFacesUseCase(face_recognition_adapter, face_store)
        await use_case.initialize_from_storage()

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    if not db_connection:
        raise RuntimeError("Database connection not initialized")
    async for session in db_connection.get_session():
        yield session


# Use case factories
async def get_chat_text_use_case(
    session: AsyncSession = Depends(get_db_session)
) -> ChatTextUseCase:
    """Get chat text use case."""
    if not llm_adapter or not rag_adapter:
        raise RuntimeError("LLM adapter or RAG adapter not initialized")
    repository = ChatRepository(session)
    return ChatTextUseCase(repository, agent_adapter)
    # return ChatTextUseCase(repository, llm_adapter, rag_adapter)


async def get_chat_speech_use_case(
    session: AsyncSession = Depends(get_db_session)
) -> ChatSpeechUseCase:
    """Get chat speech use case."""
    if not llm_adapter or not stt_adapter:
        raise RuntimeError("LLM adapter or STT adapter not initialized")
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
    if not face_recognition_adapter:
        raise RuntimeError("Face recognition adapter not initialized")
    if not tracker_adapter:
        raise RuntimeError("Tracker adapter not initialized")
    if cache_adapter is None:
        raise RuntimeError("Cache adapter not initialized")
    if face_store is None:
        raise RuntimeError("Face store not initialized")
    
    return VisionStreamUseCase(
        face_recognition=face_recognition_adapter,
        face_store=face_store,
        tracker=tracker_adapter,
        emotion_detection=emotion_detection_adapter,
        cache_adapter=cache_adapter,
        similarity_threshold=settings.face_similarity_threshold,
    )


async def get_register_faces_use_case() -> RegisterFacesUseCase:
    """Get register faces use case (multiple faces)."""
    if not face_recognition_adapter:
        raise RuntimeError("Face recognition not initialized")
    if not face_store:
        raise RuntimeError("Face store not initialized")
    return RegisterFacesUseCase(face_recognition_adapter, face_store)


async def get_list_faces_use_case() -> GetFacesUseCase:
    """Get list faces use case."""
    if not face_store:
        raise RuntimeError("Face store not initialized")
    return GetFacesUseCase(face_store=face_store)


async def get_delete_face_use_case() -> DeleteFaceUseCase:
    """Get delete face use case."""
    if not face_store:
        raise RuntimeError("Face store not initialized")
    return DeleteFaceUseCase(face_store=face_store)

# Document management dependencies
async def get_upload_document_use_case() -> UploadDocumentUseCase:
    """Get upload document use case."""
    if not rag_adapter or not document_storage:
        raise RuntimeError("RAG or document storage not initialized")
    return UploadDocumentUseCase(rag_adapter, document_storage)


async def get_list_documents_use_case() -> ListDocumentsUseCase:
    """Get list documents use case."""
    if not document_storage:
        raise RuntimeError("Document storage not initialized")
    return ListDocumentsUseCase(document_storage)


async def get_delete_document_use_case() -> DeleteDocumentUseCase:
    """Get delete document use case."""
    if not rag_adapter or not document_storage:
        raise RuntimeError("RAG or document storage not initialized")
    return DeleteDocumentUseCase(rag_adapter, document_storage)