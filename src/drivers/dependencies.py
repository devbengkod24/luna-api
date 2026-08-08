"""Dependency wiring for FastAPI.

All long-lived adapter instances are stored on `app.state` (configured by the
lifespan in `main.py`). Per-request dependency factories below resolve them via
the incoming `Request` (HTTP) or `WebSocket`.
"""

import logging
from typing import AsyncGenerator

from fastapi import Depends, Request, WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters import (
    ActiveSpeakerService,
    CartesiaTTSService,
    ChatterboxSTTSService,
    EmotiEffLibService,
    F5TTSIndonesianService,
    FaceRecognitionService,
    FaceStore,
    LangchainAgentService,
    LlamaIndexRAGService,
    MMSTTSService,
    MinIOStorageService,
    TrackerService,
    WhisperSTTService,
)
from src.adapters.identifiers import StdUuidGenerator
from src.adapters.persistence import (
    ConversationRepository,
    DatabaseConnection,
    FileRepository,
    MessageRepository,
)
from src.adapters.prompts import FilePromptRepository
from src.adapters.vad import SileroVadService, WebRtcVadService
from src.config import settings
from src.ports.identifiers import IUuidGenerator
from src.ports.prompts import IPromptRepository
from src.ports.vad import IVadService
from src.shared.exceptions import ServiceUnavailableException
from src.use_cases import (
    DeleteConversationUseCase,
    DeleteFaceUseCase,
    DeleteFileUseCase,
    DeleteMessageUseCase,
    GetFaceUseCase,
    IngestFileUseCase,
    ListConversationMessagesUseCase,
    ListConversationsUseCase,
    ListFacesUseCase,
    ListFilesUseCase,
    RegisterFacesUseCase,
    ResponseSpeechUseCase,
    ResponseTextUseCase,
    UpdateConversationUseCase,
    UploadFileUseCase,
    VisionUseCase,
)

logger = logging.getLogger(__name__)


class AppContainer:
    """Holds long-lived adapter instances. Attached to `app.state.container`."""

    db_connection: DatabaseConnection | None = None
    rag_service: LlamaIndexRAGService | None = None
    storage_service: MinIOStorageService | None = None
    stt_service: WhisperSTTService | None = None
    tts_service: object | None = None
    agent_service: LangchainAgentService | None = None
    face_service: FaceStore | None = None
    face_recognition: FaceRecognitionService | None = None
    emotion_service: EmotiEffLibService | None = None
    tracker_service: TrackerService | None = None
    asd_service: ActiveSpeakerService | None = None
    prompt_repository: IPromptRepository | None = None
    uuid_generator: IUuidGenerator | None = None
    vad_service: IVadService | None = None


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
        (
            container.emotion_service,
            container.tracker_service,
            container.asd_service,
        ) = _build_vision()
        container.prompt_repository = FilePromptRepository()
        _warm_prompt_cache(container.prompt_repository)
        container.uuid_generator = StdUuidGenerator()
        container.vad_service = _build_vad()
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


def _build_speech() -> tuple[WhisperSTTService | None, object | None]:
    stt: WhisperSTTService | None = None
    tts: object | None = None

    if settings.enable_stt:
        stt = WhisperSTTService(
            model_size=settings.whisper_model,
            device=settings.gpu_device,
        )
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
        prompts.get_tts_system_prompt()
    except Exception as e:
        logger.warning("Prompt warm-up failed (non-fatal): %s", e)


def _build_vad() -> IVadService | None:
    if not settings.enable_voice_ws:
        logger.info("Voice WS disabled (ENABLE_VOICE_WS=False).")
        return None

    provider = (settings.vad_provider or "silero").lower().strip()

    if provider == "silero":
        try:
            service = SileroVadService(
                sample_rate=settings.voice_vad_sample_rate,
                frame_duration_ms=settings.voice_vad_frame_ms,
                threshold=settings.silero_vad_threshold,
            )
            logger.info("VAD service initialized: silero (neural)")
            return service
        except Exception as e:
            logger.warning(
                "Silero VAD load failed (%s) — falling back to WebRTC", e,
            )

    # Either provider == "webrtc" or Silero failed; use the energy-based VAD.
    try:
        service = WebRtcVadService(
            aggressiveness=settings.voice_vad_aggressiveness,
            sample_rate=settings.voice_vad_sample_rate,
            frame_duration_ms=settings.voice_vad_frame_ms,
        )
        logger.info("VAD service initialized: webrtc (energy-based)")
        return service
    except Exception as e:
        logger.warning("Could not initialize any VAD service: %s", e)
        return None


def _build_vision() -> tuple[
    EmotiEffLibService | None,
    TrackerService | None,
    ActiveSpeakerService | None,
]:
    if not settings.enable_face_recognition:
        logger.info("Vision services skipped (ENABLE_FACE_RECOGNITION=False).")
        return None, None, None

    tracker: TrackerService | None = None
    try:
        tracker = TrackerService()
        logger.info("Tracker service initialized successfully")
    except Exception as e:
        logger.warning("Could not initialize tracker service: %s", e)

    emotion: EmotiEffLibService | None = None
    if settings.enable_emotion_detection:
        try:
            emotion = EmotiEffLibService()
            logger.info(
                "Emotion detection service initialized (model=%s)",
                settings.emotiefflib_model,
            )
        except Exception as e:
            logger.warning("Could not initialize emotion detection service: %s", e)
    else:
        logger.info("Emotion detection disabled (ENABLE_EMOTION_DETECTION=False).")

    # Active Speaker Detection (device-directed fusion). Fail-soft: if the model
    # or its optional deps are missing, leave it None and the directedness gate
    # fails open (see DirectednessUseCase).
    asd: ActiveSpeakerService | None = None
    if settings.enable_asd:
        try:
            asd = ActiveSpeakerService()
            logger.info("Active speaker detection service initialized (LR-ASD)")
        except Exception as e:
            logger.warning("Could not initialize active speaker detection service: %s", e)
    else:
        logger.info("Active speaker detection disabled (ENABLE_ASD=False).")

    return emotion, tracker, asd


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


# --- Response use case factories ---

async def get_response_text_use_case(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> ResponseTextUseCase:
    container = container_from(request)
    return ResponseTextUseCase(
        conversation_repo=ConversationRepository(session),
        message_repo=MessageRepository(session),
        agent_service=container.agent_service,
        tts_service=container.tts_service,
        prompt_repo=container.prompt_repository,
    )


async def get_response_speech_use_case(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> ResponseSpeechUseCase:
    container = container_from(request)
    return ResponseSpeechUseCase(
        conversation_repo=ConversationRepository(session),
        message_repo=MessageRepository(session),
        agent_service=container.agent_service,
        tts_service=container.tts_service,
        stt_service=container.stt_service,
        prompt_repo=container.prompt_repository,
    )


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
    return UploadFileUseCase(FileRepository(session), container.storage_service)


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


# --- Vision use case factory ---

def get_vision_use_case(websocket: WebSocket) -> VisionUseCase:
    container = container_from(websocket)
    if not container.face_recognition:
        raise ServiceUnavailableException("Face recognition service is not available.")
    return VisionUseCase(
        image_decoder=container.face_recognition,
        face_detector=container.face_recognition,
        face_aligner=container.face_recognition,
        face_embedder=container.face_recognition,
        face_store=container.face_service,
        tracker=container.tracker_service,
        emotion_detection=container.emotion_service,
        face_cropper=container.face_recognition,
    )


def make_directedness_use_case(container: AppContainer) -> "DirectednessUseCase":
    """Build the device-directed fusion use case from the container.

    `asd_service` may be None (ASD disabled or failed to load) — the use case
    then fails open on the audio-visual dimension. Imported lazily to avoid a
    heavier import at module load.
    """
    from src.use_cases import DirectednessUseCase

    return DirectednessUseCase(active_speaker=container.asd_service)
