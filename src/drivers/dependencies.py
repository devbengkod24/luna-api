"""Dependency injection for FastAPI."""
from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.exceptions import ServiceUnavailableException
from src.adapters.persistence import (
    DatabaseConnection,
    ConversationRepository, 
    MessageRepository, 
    FileRepository,
)
from src.adapters import (
    LangchainAgentService,
    WhisperSTTService,
    MMSTTSService,
    MinIOStorageService,
    LlamaIndexRAGService,
    FaceStore,
    FaceRecognitionService,
)
from src.config import settings
from src.use_cases import (
    ResponseTextUseCase,
    ResponseSpeechUseCase,
    ListConversationsUseCase,
    UpdateConversationUseCase,
    DeleteConversationUseCase,
    ListConversationMessagesUseCase,
    DeleteMessageUseCase,
    UploadFileUseCase,
    ListFilesUseCase,
    DeleteFileUseCase,
    IngestFileUseCase,
    RegisterFacesUseCase,
    ListFacesUseCase,
    GetFaceUseCase,
    DeleteFaceUseCase,
)

db_connection: DatabaseConnection | None = None
rag_service: LlamaIndexRAGService | None = None
storage_service: MinIOStorageService | None = None
stt_service: WhisperSTTService | None = None
tts_service: MMSTTSService | None = None
agent_service: LangchainAgentService | None = None
face_service: FaceStore | None = None
face_recognition: FaceRecognitionService | None = None


def initialize_dependencies() -> None:
    """Initialize all global adapter instances used by the API."""
    try:
        _init_database()
        _init_rag_service()
        _init_storage()
        _init_speech()
        _init_agent()
        _init_face()
    except Exception as e:
        print(f"Error initializing dependencies: {e}")
        raise


def _init_database() -> None:
    global db_connection
    db_connection = DatabaseConnection(url=settings.database_url)


def _init_rag_service() -> None:
    global rag_service
    rag_service = LlamaIndexRAGService()


async def initialize_rag_async() -> None:
    """Call RAG service's async initialize() (Qdrant collection + vector store). Call from app lifespan."""
    if rag_service is None:
        return
    await rag_service.initialize()


async def initialize_face_async() -> None:
    """Initialize face Qdrant collection. Call from app lifespan."""
    if face_service is None:
        return
    await face_service.initialize()


def _init_storage() -> None:
    global storage_service
    try:
        storage_service = MinIOStorageService()
        print("Storage service initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize storage service: {e}")
        storage_service = None


def _init_speech() -> None:
    global stt_service, tts_service

    stt_service = WhisperSTTService(
        model_size=settings.whisper_model,
        device=settings.gpu_device,
        compute_type="int8",
    )

    try:
        tts_service = MMSTTSService(
            voice_clone_dir=settings.voice_clone_dir,
        )
        print("TTS adapter initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize TTS adapter: {e}")
        tts_service = None


def _init_agent() -> None:
    global agent_service

    agent_service = LangchainAgentService(
        rag_service=rag_service,
    )


def _init_face() -> None:
    global face_service, face_recognition
    if not settings.enable_face_recognition:
        print("Face recognition disabled (ENABLE_FACE_RECOGNITION=False).")
        face_service = None
        face_recognition = None
        return
    try:
        face_service = FaceStore()
        face_recognition = FaceRecognitionService()
        print("Face service initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize face service: {e}")
        face_service = None
        face_recognition = None


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if not db_connection:
        raise RuntimeError("Database connection not initialized")
    async for session in db_connection.get_session():
        yield session


# --- Response use case factories ---

async def get_response_text_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> ResponseTextUseCase:
    conversation_repo = ConversationRepository(session)
    message_repo = MessageRepository(session)
    return ResponseTextUseCase(conversation_repo, message_repo, agent_service, tts_service)


async def get_response_speech_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> ResponseSpeechUseCase:
    conversation_repo = ConversationRepository(session)
    message_repo = MessageRepository(session)
    return ResponseSpeechUseCase(conversation_repo, message_repo, agent_service, tts_service, stt_service)


# --- Conversation use case factories ---

async def get_list_conversations_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> ListConversationsUseCase:
    return ListConversationsUseCase(ConversationRepository(session))


async def get_update_conversation_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> UpdateConversationUseCase:
    return UpdateConversationUseCase(ConversationRepository(session))


async def get_delete_conversation_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> DeleteConversationUseCase:
    return DeleteConversationUseCase(ConversationRepository(session))


async def get_list_conversation_messages_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> ListConversationMessagesUseCase:
    return ListConversationMessagesUseCase(MessageRepository(session))


async def get_delete_message_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> DeleteMessageUseCase:
    return DeleteMessageUseCase(MessageRepository(session))


# --- File use case factories ---

async def get_upload_file_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> UploadFileUseCase:
    if not storage_service:
        raise ServiceUnavailableException(
            "Storage (MinIO) is not available. Check MINIO_* env vars and ensure MinIO is running."
        )
    return UploadFileUseCase(FileRepository(session), storage_service)


async def get_list_files_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> ListFilesUseCase:
    return ListFilesUseCase(FileRepository(session))


async def get_delete_file_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> DeleteFileUseCase:
    if not storage_service:
        raise ServiceUnavailableException(
            "Storage (MinIO) is not available. Check MINIO_* env vars and ensure MinIO is running."
        )
    return DeleteFileUseCase(FileRepository(session), storage_service, rag_service)


async def get_ingest_file_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> IngestFileUseCase:
    if not storage_service:
        raise ServiceUnavailableException(
            "Storage (MinIO) is not available. Check MINIO_* env vars and ensure MinIO is running."
        )
    return IngestFileUseCase(FileRepository(session), storage_service, rag_service)


# --- Face use case factories ---

def get_register_faces_use_case() -> RegisterFacesUseCase:
    if not face_service or not face_recognition:
        raise ServiceUnavailableException("Face service is not available.")
    return RegisterFacesUseCase(face_recognition, face_service)


def get_list_faces_use_case() -> ListFacesUseCase:
    if not face_service:
        raise ServiceUnavailableException("Face service is not available.")
    return ListFacesUseCase(face_service)


def get_face_use_case() -> GetFaceUseCase:
    if not face_service:
        raise ServiceUnavailableException("Face service is not available.")
    return GetFaceUseCase(face_service)


def get_delete_face_use_case() -> DeleteFaceUseCase:
    if not face_service:
        raise ServiceUnavailableException("Face service is not available.")
    return DeleteFaceUseCase(face_service)
