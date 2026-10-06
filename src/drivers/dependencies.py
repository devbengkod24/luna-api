"""Dependency wiring for FastAPI.

All long-lived adapter instances are stored on `app.state` (configured by the
lifespan in `main.py`). Per-request dependency factories below resolve them via
the incoming `Request`.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters import (
    CartesiaTTSService,
    ChatterboxSTTSService,
    DeepgramSTTService,
    F5TTSIndonesianService,
    FaceRecognitionService,
    FaceStore,
    LangchainAgentService,
    LlamaIndexRAGService,
    MMSTTSService,
    MinIOStorageService,
)
from src.adapters.identifiers import StdUuidGenerator
from src.adapters.persistence import (
    ConversationRepository,
    DatabaseConnection,
    FileRepository,
    MessageRepository,
)
from src.adapters.prompts import FilePromptRepository
from src.config import settings
from src.ports.identifiers import IUuidGenerator
from src.ports.prompts import IPromptRepository
from src.ports.services import ISTTService, ITTSService
from src.shared.exceptions import ServiceUnavailableException
from src.use_cases import (
    DeleteConversationUseCase,
    DeleteFaceUseCase,
    DeleteFileUseCase,
    DeleteMessageUseCase,
    DeletePhotoUseCase,
    GetFaceUseCase,
    GetPhotoUseCase,
    IngestFileUseCase,
    ListConversationMessagesUseCase,
    ListConversationsUseCase,
    ListFacesUseCase,
    ListFilesUseCase,
    ListPhotosUseCase,
    MatchFaceUseCase,
    RegisterFacesUseCase,
    ResponseSpeechUseCase,
    ResponseTextUseCase,
    SavePhotoUseCase,
    UpdateConversationUseCase,
    UploadFileUseCase,
)
from src.use_cases.response_use_case import ResponseOptions

logger = logging.getLogger(__name__)


class AppContainer:
    """Holds long-lived adapter instances. Attached to `app.state.container`."""

    def __init__(self) -> None:
        self.db_connection: DatabaseConnection | None = None
        self.rag_service: LlamaIndexRAGService | None = None
        self.storage_service: MinIOStorageService | None = None
        self.stt_service: ISTTService | None = None
        self.tts_service: ITTSService | None = None
        self.agent_service: LangchainAgentService | None = None
        self.face_service: FaceStore | None = None
        self.face_recognition: FaceRecognitionService | None = None
        self.prompt_repository: IPromptRepository | None = None
        self.uuid_generator: IUuidGenerator | None = None


def initialize_container() -> AppContainer:
    """Build the application container with all sync-initialized adapters."""
    container = AppContainer()
    try:
        container.db_connection = DatabaseConnection(url=settings.database_url)
        container.rag_service = LlamaIndexRAGService()
        container.storage_service = _build_storage()
        container.stt_service, container.tts_service = _build_speech()
        container.agent_service = LangchainAgentService(rag_service=container.rag_service)
        container.face_service, container.face_recognition = _build_face()
        container.prompt_repository = FilePromptRepository()
        _warm_prompt_cache(container.prompt_repository)
        container.uuid_generator = StdUuidGenerator()
        return container
    except Exception as e:
        logger.critical("Error initializing dependencies: %s", e, exc_info=True)
        raise


def _build_storage() -> MinIOStorageService | None:
    try:
        service = MinIOStorageService()
        logger.info("Storage service initialized successfully")
        return service
    except Exception as e:
        logger.warning("Could not initialize storage service: %s", e)
        return None


def _build_stt() -> ISTTService | None:
    """Build the Deepgram STT adapter, or None if it cannot start.

    There is no local model to fall back to on this branch — exp/edge is the
    Whisper branch. None makes /turn/stream refuse the kiosk outright, which is
    louder than a kiosk that connects and then never hears anyone.
    """
    try:
        if not settings.deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY is not set")
        return DeepgramSTTService(api_key=settings.deepgram_api_key)
    except Exception as e:
        logger.error("Could not initialize Deepgram STT (%s) — voice turns are disabled.", e)
        return None


def _build_speech() -> tuple[ISTTService | None, ITTSService | None]:
    stt: ISTTService | None = None
    tts: ITTSService | None = None

    if settings.enable_stt:
        stt = _build_stt()
    else:
        logger.info("STT disabled (ENABLE_STT=False).")

    if settings.enable_tts:
        try:
            provider = (settings.tts_provider or "mms").strip().lower()
            if provider in {"chatterbox", "chatterbox-id", "chatterbox_indonesia"}:
                tts = ChatterboxSTTSService(voice_clone_dir=settings.voice_clone_dir)
                logger.info("TTS adapter initialized successfully (provider=chatterbox)")
            elif provider in {"f5", "f5-tts", "f5_tts", "f5-indonesia", "f5_indo"}:
                tts = F5TTSIndonesianService(voice_clone_dir=settings.voice_clone_dir)
                logger.info("TTS adapter initialized successfully (provider=f5)")
            elif provider == "cartesia":
                if not settings.cartesia_api_key or not settings.cartesia_voice_id:
                    logger.warning(
                        "Cartesia provider requires CARTESIA_API_KEY and CARTESIA_VOICE_ID — TTS disabled."
                    )
                else:
                    tts = CartesiaTTSService(
                        api_key=settings.cartesia_api_keys,
                        voice_id=settings.cartesia_voice_id,
                        model_id=settings.cartesia_model_id,
                        sample_rate=settings.cartesia_sample_rate,
                        language=settings.cartesia_language,
                    )
                    logger.info("TTS adapter initialized successfully (provider=cartesia)")
            else:
                tts = MMSTTSService()
                logger.info("TTS adapter initialized successfully (provider=mms)")
        except Exception as e:
            logger.warning("Could not initialize TTS adapter: %s", e)
            tts = None
    else:
        logger.info("TTS disabled (ENABLE_TTS=False).")

    return stt, tts


def _build_face() -> tuple[FaceStore | None, FaceRecognitionService | None]:
    if not settings.enable_face_recognition:
        logger.info("Face recognition disabled (ENABLE_FACE_RECOGNITION=False).")
        return None, None
    try:
        service = FaceStore()
        recognition = FaceRecognitionService()
        logger.info("Face service initialized successfully")
        return service, recognition
    except Exception as e:
        logger.warning("Could not initialize face service: %s", e)
        return None, None


def _warm_prompt_cache(prompts: IPromptRepository) -> None:
    """Pre-render prompts so the first request doesn't pay disk I/O cost."""
    try:
        from src.shared.constants import Character

        for character in Character:
            prompts.get_system_prompt(character)
    except Exception as e:
        logger.warning("Prompt warm-up failed (non-fatal): %s", e)


def container_from(scope) -> AppContainer:
    """Resolve the application container from any object exposing `.app.state`."""
    container: AppContainer | None = getattr(scope.app.state, "container", None)
    if container is None:
        raise RuntimeError("Application container not initialized")
    return container


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    container = container_from(request)
    if container.db_connection is None:
        raise RuntimeError("Database connection not initialized")
    async for session in container.db_connection.get_session():
        yield session


@asynccontextmanager
async def db_session(container: AppContainer) -> AsyncIterator[AsyncSession]:
    """One committed-on-exit session, for streams that outlive a request's
    dependencies. The block exits — and the transaction commits — before the
    caller announces that the reply is done, so a client that reacts to `done`
    by reading the conversation finds it there."""
    if container.db_connection is None:
        raise ServiceUnavailableException("Database is not available.")
    async with container.db_connection.session() as session:
        yield session


# --- Response use case factories ---

def response_options() -> ResponseOptions:
    return ResponseOptions(
        history_limit=settings.chat_history_limit,
        agent_idle_timeout_s=settings.agent_idle_timeout_seconds,
        timezone=settings.app_timezone,
    )


def build_response_text_use_case(container: AppContainer, session: AsyncSession) -> ResponseTextUseCase:
    return ResponseTextUseCase(
        conversation_repo=ConversationRepository(session),
        message_repo=MessageRepository(session),
        agent_service=container.agent_service,
        tts_service=container.tts_service,
        prompt_repo=container.prompt_repository,
        options=response_options(),
    )


def build_response_speech_use_case(container: AppContainer, session: AsyncSession) -> ResponseSpeechUseCase:
    return ResponseSpeechUseCase(
        conversation_repo=ConversationRepository(session),
        message_repo=MessageRepository(session),
        agent_service=container.agent_service,
        tts_service=container.tts_service,
        stt_service=container.stt_service,
        prompt_repo=container.prompt_repository,
        options=response_options(),
    )


def require_response_services(request: Request) -> AppContainer:
    container = container_from(request)
    if container.agent_service is None or container.db_connection is None:
        raise ServiceUnavailableException("Response services are not available.")
    return container


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
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> UploadFileUseCase:
    container = container_from(request)
    if not container.storage_service:
        raise ServiceUnavailableException(
            "Storage (MinIO) is not available. Check MINIO_* env vars and ensure MinIO is running."
        )
    return UploadFileUseCase(
        FileRepository(session), container.storage_service, container.uuid_generator
    )


async def get_list_files_use_case(
    session: AsyncSession = Depends(get_db_session),
) -> ListFilesUseCase:
    return ListFilesUseCase(FileRepository(session))


async def get_delete_file_use_case(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> DeleteFileUseCase:
    container = container_from(request)
    if not container.storage_service:
        raise ServiceUnavailableException(
            "Storage (MinIO) is not available. Check MINIO_* env vars and ensure MinIO is running."
        )
    return DeleteFileUseCase(
        FileRepository(session), container.storage_service, container.rag_service
    )


async def get_ingest_file_use_case(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> IngestFileUseCase:
    container = container_from(request)
    if not container.storage_service:
        raise ServiceUnavailableException(
            "Storage (MinIO) is not available. Check MINIO_* env vars and ensure MinIO is running."
        )
    return IngestFileUseCase(
        FileRepository(session), container.storage_service, container.rag_service
    )


# --- Photo use case factories ---
#
# No database session: a photo is only its object in storage.

def _require_storage(request: Request):
    container = container_from(request)
    if not container.storage_service:
        raise ServiceUnavailableException(
            "Storage (MinIO) is not available. Check MINIO_* env vars and ensure MinIO is running."
        )
    return container


def get_save_photo_use_case(request: Request) -> SavePhotoUseCase:
    return SavePhotoUseCase(_require_storage(request).storage_service)


def get_list_photos_use_case(request: Request) -> ListPhotosUseCase:
    return ListPhotosUseCase(_require_storage(request).storage_service)


def get_get_photo_use_case(request: Request) -> GetPhotoUseCase:
    return GetPhotoUseCase(_require_storage(request).storage_service)


def get_delete_photo_use_case(request: Request) -> DeletePhotoUseCase:
    return DeletePhotoUseCase(_require_storage(request).storage_service)


# --- Face use case factories ---

def get_register_faces_use_case(request: Request) -> RegisterFacesUseCase:
    container = container_from(request)
    if not container.face_service or not container.face_recognition:
        raise ServiceUnavailableException("Face service is not available.")
    return RegisterFacesUseCase(
        face_detector=container.face_recognition,
        face_service=container.face_service,
        uuid_generator=container.uuid_generator,
    )


def get_match_face_use_case(request: Request) -> MatchFaceUseCase:
    container = container_from(request)
    if not container.face_service:
        raise ServiceUnavailableException("Face service is not available")
    return MatchFaceUseCase(container.face_service, settings.face_match_min_threshold)


def get_list_faces_use_case(request: Request) -> ListFacesUseCase:
    container = container_from(request)
    if not container.face_service:
        raise ServiceUnavailableException("Face service is not available.")
    return ListFacesUseCase(container.face_service)


def get_face_use_case(request: Request) -> GetFaceUseCase:
    container = container_from(request)
    if not container.face_service:
        raise ServiceUnavailableException("Face service is not available.")
    return GetFaceUseCase(container.face_service)


def get_delete_face_use_case(request: Request) -> DeleteFaceUseCase:
    container = container_from(request)
    if not container.face_service:
        raise ServiceUnavailableException("Face service is not available.")
    return DeleteFaceUseCase(container.face_service)
