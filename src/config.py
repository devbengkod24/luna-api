"""Application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic import field_validator
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
    app_name: str = Field(default="luna-api", alias="APP_NAME")
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
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")
    knowledge_collection_name: str = Field(
        default="knowledge_collection", alias="KNOWLEDGE_COLLECTION_NAME"
    )
    faces_collection_name: str = Field(
        default="faces_collection", alias="FACE_COLLECTION_NAME"
    )

    # MinIO
    minio_endpoint: str = Field(..., alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(..., alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(..., alias="MINIO_SECRET_KEY")
    minio_bucket_name: str = Field(default="luna-files", alias="MINIO_BUCKET_NAME")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # Ollama
    ollama_base_url: str = Field(default="http://ollama:11434", alias="OLLAMA_BASE_URL")
    ollama_llm_model: str = Field(
        default="gpt-oss:120b-cloud", alias="OLLAMA_LLM_MODEL"
    )
    ollama_embedding_model: str = Field(
        default="embeddinggemma:latest", alias="OLLAMA_EMBEDDING_MODEL"
    )
    ollama_temperature: float = Field(default=0.3, alias="OLLAMA_TEMPERATURE")
    ollama_top_p: float = Field(default=0.6, alias="OLLAMA_TOP_P")
    # Defaults chosen to leave adequate input budget within context window.
    ollama_num_ctx: int = Field(default=16384, alias="OLLAMA_NUM_CTX")
    ollama_max_tokens: int = Field(default=1024, alias="OLLAMA_MAX_TOKENS")
    ollama_top_k: int = Field(default=40, alias="OLLAMA_TOP_K")
    ollama_min_p: float = Field(default=0.0, alias="OLLAMA_MIN_P")
    ollama_repeat_penalty: float = Field(default=1.18, alias="OLLAMA_REPEAT_PENALTY")
    ollama_repeat_last_n: int = Field(default=64, alias="OLLAMA_REPEAT_LAST_N")
    ollama_num_batch: int = Field(default=128, alias="OLLAMA_NUM_BATCH")

    gpu_device: int = Field(default=-1, alias="GPU_DEVICE")  # -1 for CPU, 0+ for GPU

    # Face Recognition (InsightFace)
    enable_face_recognition: bool = Field(default=True, alias="ENABLE_FACE_RECOGNITION")
    face_recognition_model: str = Field(
        default="buffalo_sc", alias="FACE_RECOGNITION_MODEL"
    )
    face_recognition_ctx_id: int = Field(
        default=-1, alias="FACE_RECOGNITION_CTX_ID"
    )  # -1 CPU, 0+ GPU
    face_detection_size: int = Field(
        default=320, alias="FACE_DETECTION_SIZE"
    )  # Detection size (320 faster, 640 more accurate)
    face_similarity_threshold: float = Field(
        default=0.5, alias="FACE_SIMILARITY_THRESHOLD"
    )  # Cosine similarity threshold (0.4-0.6 recommended)

    face_max_batch_size: int = 100  # Max faces per request
    face_max_image_size: int = 10 * 1024 * 1024  # 10MB

    persist_dir: str = "storage/index"
    rag_max_file_size: int = 50 * 1024 * 1024  # 50MB
    rag_allowed_extensions: set[str] = {".pdf", ".txt", ".docx", ".doc", ".md"}
    rag_chunk_strategy: str = Field(
        default="semantic", alias="RAG_CHUNK_STRATEGY"
    )  # "token" | "semantic"
    rag_chunk_size: int = Field(default=512, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=50, alias="RAG_CHUNK_OVERLAP")

    # Chat
    chat_max_prompt_length: int = 10000
    chat_max_audio_base64_size: int = 10 * 1024 * 1024  # 10MB
    chat_history_limit: int = Field(default=20, alias="CHAT_HISTORY_LIMIT")
    sse_idle_timeout_seconds: float = Field(
        default=60.0, alias="SSE_IDLE_TIMEOUT_SECONDS"
    )
    agent_idle_timeout_seconds: float = Field(
        default=45.0, alias="AGENT_IDLE_TIMEOUT_SECONDS"
    )
    agent_tool_timeout_seconds: float = Field(
        default=20.0, alias="AGENT_TOOL_TIMEOUT_SECONDS"
    )

    # Emotion Detection
    enable_emotion_detection: bool = Field(
        default=True, alias="ENABLE_EMOTION_DETECTION"
    )
    emotion_provider: str = Field(
        default="emofer", alias="EMOTION_PROVIDER"
    )  # "huggingface" or "emofer"
    emotion_model: str = Field(
        default="dima806/facial_emotions_image_detection", alias="EMOTION_MODEL"
    )
    # Emofer ONNX model paths (used when EMOTION_PROVIDER=emofer)
    emofer_vad_model_path: str = Field(
        default="storage/models/vad_multitask.onnx", alias="EMOFER_VAD_MODEL_PATH"
    )
    emofer_emotic_model_path: str = Field(
        default="storage/models/emotic.onnx", alias="EMOFER_EMOTIC_MODEL_PATH"
    )

    # Vision / Tracker (ByteTrack)
    tracker_max_age: int = Field(default=30, alias="TRACKER_MAX_AGE")
    tracker_min_hits: int = Field(default=3, alias="TRACKER_MIN_HITS")
    tracker_min_score: float = Field(default=0.4, alias="TRACKER_MIN_SCORE")

    # Vision / Recognition pipeline
    vision_emotion_interval: int = Field(default=5, alias="VISION_EMOTION_INTERVAL")
    vision_recognition_retry_interval: float = Field(
        default=2.0, alias="VISION_RECOGNITION_RETRY_INTERVAL"
    )
    vision_max_recognition_attempts: int = Field(
        default=5, alias="VISION_MAX_RECOGNITION_ATTEMPTS"
    )

    # Vision / WebSocket optimizations
    vision_target_latency_ms: float = Field(
        default=150.0, alias="VISION_TARGET_LATENCY_MS"
    )
    vision_latency_window: int = Field(default=10, alias="VISION_LATENCY_WINDOW")
    vision_throttle_min_fps: int = Field(default=5, alias="VISION_THROTTLE_MIN_FPS")
    vision_throttle_max_fps: int = Field(default=30, alias="VISION_THROTTLE_MAX_FPS")
    vision_throttle_emit_interval: float = Field(
        default=2.0, alias="VISION_THROTTLE_EMIT_INTERVAL"
    )

    # Speech-to-Text (Faster Whisper)
    enable_stt: bool = Field(default=True, alias="ENABLE_STT")
    whisper_model: str = Field(default="large-v3", alias="WHISPER_MODEL")
    whisper_language: str = Field(default="id", alias="WHISPER_LANGUAGE")
    whisper_beam_size: int = Field(default=5, alias="WHISPER_BEAM_SIZE")
    # Live voice path is latency-critical and utterances are short commands —
    # greedy decoding (beam=1) shaves a large chunk off per-turn STT time with
    # negligible accuracy loss. Chat/file uploads keep the higher-quality beam
    # via `whisper_beam_size`.
    whisper_voice_beam_size: int = Field(default=5, alias="WHISPER_VOICE_BEAM_SIZE")

    # Voice WebSocket / VAD
    enable_voice_ws: bool = Field(default=True, alias="ENABLE_VOICE_WS")
    # "silero" (neural, robust in noise) or "webrtc" (energy-based fallback).
    # Factory in dependencies.py auto-falls back to webrtc if silero load fails.
    vad_provider: str = Field(default="silero", alias="VAD_PROVIDER")
    silero_vad_threshold: float = Field(default=0.5, alias="SILERO_VAD_THRESHOLD")
    voice_vad_aggressiveness: int = Field(default=2, alias="VOICE_VAD_AGGRESSIVENESS")
    voice_vad_sample_rate: int = Field(default=16000, alias="VOICE_VAD_SAMPLE_RATE")
    voice_vad_frame_ms: int = Field(default=30, alias="VOICE_VAD_FRAME_MS")
    voice_speech_start_frames: int = Field(
        default=3, alias="VOICE_SPEECH_START_FRAMES"
    )  # consecutive speech frames to enter SPEAKING (~90ms)
    voice_speech_end_frames: int = Field(
        default=20, alias="VOICE_SPEECH_END_FRAMES"
    )  # consecutive silence frames to end utterance (~600ms). Lower = snappier
    # turn-taking but risks cutting off hesitant speakers; raise toward 25 (~750ms)
    # if the kiosk audience pauses mid-sentence a lot.
    voice_prebuffer_frames: int = Field(
        default=7, alias="VOICE_PREBUFFER_FRAMES"
    )  # frames retained before speech_start (~210ms)
    voice_max_utterance_seconds: float = Field(
        default=30.0, alias="VOICE_MAX_UTTERANCE_SECONDS"
    )

    # Device-Directed Speech Detection (Active Speaker Detection + facing fusion)
    # When enabled, a finished VAD utterance is only committed to STT/LLM/TTS if
    # the candidate face was both facing the kiosk AND its lip motion correlated
    # with the audio (LR-ASD). Fail-open: if ASD can't load or no face is tracked,
    # the utterance is treated as directed so the system never goes deaf.
    enable_asd: bool = Field(default=False, alias="ENABLE_ASD")
    asd_model_path: str = Field(
        default="storage/models/lr_asd.pth", alias="ASD_MODEL_PATH"
    )
    # Per-frame visual "speaking" score gate (LR-ASD output, sigmoid 0..1).
    asd_score_threshold: float = Field(default=0.4, alias="ASD_SCORE_THRESHOLD")
    # Fraction of the utterance's frames that must clear asd_score_threshold for
    # the candidate to count as the active speaker.
    asd_presence_ratio: float = Field(default=0.35, alias="ASD_PRESENCE_RATIO")
    # Mean facing score (FE MediaPipe metadata, 0..1) required over the utterance.
    facing_threshold: float = Field(default=0.5, alias="FACING_THRESHOLD")

    # Text-to-Speech (Facebook MMS-TTS Indonesian)
    enable_tts: bool = Field(default=True, alias="ENABLE_TTS")
    voice_clone_dir: str = Field(
        default="storage/voice",
        alias="VOICE_CLONE_DIR",
        description="Directory for TTS reference voice audio files.",
    )
    tts_provider: str = Field(default="chatterbox", alias="TTS_PROVIDER")
    chatterbox_audio_prompt_path: str | None = Field(
        default="None",
        alias="CHATTERBOX_AUDIO_PROMPT_PATH",
        description="Optional WAV path for Chatterbox voice cloning.",
    )

    # Rate limiting constants
    frame_timeout_seconds: float = 5.0

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"], alias="CORS_ORIGINS")

    # Request / streaming
    request_id_header: str = Field(default="X-Request-ID", alias="REQUEST_ID_HEADER")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        if v is None:
            return ["*"]
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            raw = v.strip()
            if not raw:
                return ["*"]
            if raw == "*":
                return ["*"]
            # allow comma-separated: "https://a.com,https://b.com"
            if raw.startswith("["):
                # JSON list string; let pydantic parse it in later stage if possible
                return v
            return [part.strip() for part in raw.split(",") if part.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
