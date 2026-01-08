"""Dependency injection for FastAPI."""
from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.postgres.connection import DatabaseConnection
from src.adapters.postgres.chat_repository import ChatRepository
from src.adapters.llamaindex.rag_adapter import RAGAdapter
from src.adapters.qdrant.rag_store import RAGStore
from src.adapters.qdrant.face_store import FaceStore
from src.adapters.filesystem.storage_adapter import FileStorageAdapter
from src.adapters.whisper.stt_adapter import STTAdapter
from src.adapters.insightface.face_recognition_adapter import FaceRecognitionAdapter
from src.adapters.bytetrack.tracker_adapter import TrackerAdapter
from src.adapters.langchain.agent_adapter import AgentAdapter
from src.adapters.huggingface.emotion_adapter import EmotionDetectionAdapter
from src.adapters.huggingface.tts_adapter import TTSAdapter
from src.domain.entities.storage import StorageType
from src.config import settings
from src.use_cases.chat_text import ChatTextUseCase
from src.use_cases.chat_speech import ChatSpeechUseCase
from src.use_cases.chat_history import GetChatHistoryUseCase, GetChatSessionUseCase
from src.use_cases.vision import VisionUseCase
from src.use_cases.faces import (
    RegisterFacesUseCase,
    GetFacesUseCase,
    DeleteFaceUseCase,
)
from src.use_cases.knowledge import (
    UploadKnowledgeUseCase,
    ListKnowledgeUseCase,
    DownloadKnowledgeUseCase,
    DeleteKnowledgeUseCase,
)
# Global instances (initialized in main.py)
db_connection: DatabaseConnection | None = None
rag_adapter: RAGAdapter | None = None  # retriever (LlamaIndex)
rag_store: RAGStore | None = None      # store (Qdrant)
rag_storage: FileStorageAdapter | None = None  # filesystem storage for RAG files
faces_storage: FileStorageAdapter | None = None
stt_adapter: STTAdapter | None = None
tts_adapter: TTSAdapter | None = None
face_recognition_adapter: FaceRecognitionAdapter | None = None
face_store: FaceStore | None = None
tracker_adapter: TrackerAdapter | None = None
emotion_detection_adapter: EmotionDetectionAdapter | None = None
agent_adapter: AgentAdapter | None = None


def initialize_dependencies() -> None:
    """Initialize global dependencies."""
    global db_connection, rag_adapter, rag_store, rag_storage, faces_storage
    global stt_adapter, tts_adapter
    global face_recognition_adapter, face_store, tracker_adapter
    global emotion_detection_adapter, agent_adapter

    try:
        db_connection = DatabaseConnection(url=settings.database_url)

        # RAG components
        rag_adapter = RAGAdapter()  # retriever
        rag_store = RAGStore()      # vector DB store

        # Initialize file storage and metadata storage for RAG documents
        rag_storage = FileStorageAdapter(
            storage_type=StorageType.RAG,
        )
        faces_storage = FileStorageAdapter(
            storage_type=StorageType.FACES
        )
        
        stt_adapter = STTAdapter(
            model_size=settings.whisper_model, 
            device=settings.gpu_device, 
            compute_type="int8"
        )
        
        # Initialize TTS adapter
        try:
            tts_adapter = TTSAdapter(
                voice_clone_dir=settings.voice_clone_dir
            )
            print("TTS adapter initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize TTS adapter: {e}")
            tts_adapter = None
        
        agent_adapter = AgentAdapter(
            model=settings.llm_model,
            rag_adapter=rag_adapter,
            web_search_enabled=settings.web_search_enabled,
        )
        
        
        if settings.enable_face_recognition:
            try:
                face_recognition_adapter = FaceRecognitionAdapter(
                    model_name=settings.face_recognition_model,
                    ctx_id=settings.gpu_device,
                )
                face_store = FaceStore()
                
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

async def initialize_face_store(run_seed: bool = False):
    if not face_store:
        raise RuntimeError("Face store not initialized")
    await face_store.initialize(collection_name=settings.faces_collection_name)

    if run_seed:
        if not face_recognition_adapter:
            raise RuntimeError("Face recognition not initialized")
        use_case = RegisterFacesUseCase(face_recognition_adapter, face_store, faces_storage)
        await use_case.seed_faces(
            storage_path=settings.faces_dir, 
            supported_extensions=settings.faces_allowed_extensions
        )
        print("Faces initialized successfully")
    else:
        print("Faces disabled")

async def initialize_rag_store(run_seed: bool = False):
    if not rag_adapter or not rag_store:
        raise RuntimeError("RAG adapter/store not initialized")

    # Init retriever & store (share same Qdrant collection)
    await rag_adapter.initialize()
    await rag_store.initialize()

    if run_seed:
        if not rag_storage:
            raise RuntimeError("RAG file storage not initialized")
        use_case = UploadKnowledgeUseCase(rag_storage, rag_store)
        await use_case.seed_knowledge()
        print("Knowledge initialized successfully")
    else:
        print("Knowledge disabled")

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
    if not rag_adapter:
        raise RuntimeError("LLM adapter or RAG adapter not initialized")
    repository = ChatRepository(session)
    return ChatTextUseCase(repository, agent_adapter)


async def get_chat_speech_use_case(
    session: AsyncSession = Depends(get_db_session)
) -> ChatSpeechUseCase:
    """Get chat speech use case."""
    if not stt_adapter:
        raise RuntimeError("STT adapter not initialized")
    repository = ChatRepository(session)
    return ChatSpeechUseCase(repository, agent_adapter, stt_adapter, tts_adapter)


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


def get_vision_use_case() -> VisionUseCase:
    """Get vision use case."""
    if not face_recognition_adapter:
        raise RuntimeError("Face recognition adapter not initialized")
    if not tracker_adapter:
        raise RuntimeError("Tracker adapter not initialized")
    if face_store is None:
        raise RuntimeError("Face store not initialized")
    
    return VisionUseCase(
        face_recognition=face_recognition_adapter,
        face_store=face_store,
        tracker=tracker_adapter,
        emotion_detection=emotion_detection_adapter,
        similarity_threshold=settings.face_similarity_threshold,
        emotion_detection_interval=settings.emotion_detection_interval,
        recognition_retry_interval=settings.recognition_retry_interval,
        max_recognition_attempts=settings.max_recognition_attempts,
    )


async def get_register_faces_use_case() -> RegisterFacesUseCase:
    """Get register faces use case (multiple faces)."""
    if not face_recognition_adapter:
        raise RuntimeError("Face recognition not initialized")
    if not face_store:
        raise RuntimeError("Face store not initialized")
    return RegisterFacesUseCase(face_recognition_adapter, face_store, faces_storage)


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
async def get_upload_knowledge_use_case() -> UploadKnowledgeUseCase:
    """Get upload knowledge use case."""
    if not rag_storage or not rag_store:
        raise RuntimeError("RAG storage or RAG store not initialized")
    return UploadKnowledgeUseCase(
        file_storage=rag_storage,
        rag_store=rag_store,
    )


async def get_list_knowledge_use_case() -> ListKnowledgeUseCase:
    """Get list knowledge use case."""
    if not rag_store:
        raise RuntimeError("RAG store not initialized")
    return ListKnowledgeUseCase(
        rag_store=rag_store,
    )


async def get_download_knowledge_use_case() -> DownloadKnowledgeUseCase:
    """Get download knowledge use case."""
    if not rag_storage or not rag_store:
        raise RuntimeError("RAG storage or RAG store not initialized")
    return DownloadKnowledgeUseCase(
        file_storage=rag_storage,
        rag_store=rag_store,
    )


async def get_delete_knowledge_use_case() -> DeleteKnowledgeUseCase:
    """Get delete knowledge use case."""
    if not rag_storage or not rag_store:
        raise RuntimeError("RAG storage or RAG store not initialized")
    return DeleteKnowledgeUseCase(
        file_storage=rag_storage,
        rag_store=rag_store,
    )