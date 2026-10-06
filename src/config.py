"""Application configuration."""

from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# pydantic-settings JSON-decodes env values for complex types (list, dict)
# *before* any `mode="before"` validator runs, so a plain comma-separated
# `FOO=a,b` raises SettingsError instead of reaching our parser. NoDecode hands
# the raw string over untouched and lets the validators below split it.
CsvList = Annotated[list[str], NoDecode]


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
    # Off unless asked for. In debug mode Starlette answers an unhandled
    # exception with its traceback, bypassing the generic 500 handler in
    # drivers/main.py — to anyone who can reach the port.
    debug: bool = Field(default=False, alias="DEBUG")
    # The credential every /api/v1 route and the turn socket require, as
    # `Authorization: Bearer <token>` (or `?token=` for what a browser loads by
    # itself, like an <img>). Empty leaves the API open, which is only right on
    # a machine nobody else can reach: without it anyone on the network can list
    # every enrolled name and NIM. The kiosk presents it as CLOUD_API_TOKEN.
    api_access_token: str = Field(default="", alias="API_ACCESS_TOKEN")
    # Largest request body accepted, enforced before anything parses it. Sized
    # for the biggest legitimate upload (a 50 MB RAG document) and nothing else:
    # a JSON body is parsed whole into memory, and the per-field limits in the
    # schemas only run after that.
    max_request_body_bytes: int = Field(default=51 * 1024 * 1024, alias="MAX_REQUEST_BODY_BYTES")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    # Root log level. Without an explicit logging config the root logger has no
    # handler, so every logger.info() in src/ is silently dropped (uvicorn only
    # configures its own `uvicorn.*` loggers). See configure_logging() in
    # drivers/main.py. "DEBUG" | "INFO" | "WARNING" | "ERROR".
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # IANA zone used for user-facing timestamps (conversation titles). Containers
    # run on UTC, so without this a title would read 7 hours behind the kiosk.
    app_timezone: str = Field(default="Asia/Jakarta", alias="APP_TIMEZONE")

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
    # Reasoning effort for reasoning models (e.g. gpt-oss): "low" | "medium" |
    # "high". Forwarded to Ollama as the `think` option via ChatOllama(reasoning=).
    # None = model default (gpt-oss reasons at medium and leaks <think> tags into
    # content). "low" minimizes thinking latency and routes reasoning to a
    # separate field so the answer stream stays clean. Empty string → treated as
    # None. Latency experiment: set OLLAMA_REASONING_EFFORT=low.
    ollama_reasoning: str | None = Field(default=None, alias="OLLAMA_REASONING_EFFORT")

    gpu_device: int = Field(default=-1, alias="GPU_DEVICE")  # -1 for CPU, 0+ for GPU

    # Face Recognition (OpenCV: YuNet detection + SFace embedding)
    enable_face_recognition: bool = Field(default=True, alias="ENABLE_FACE_RECOGNITION")
    # The longest side a photo is scaled to before detection. YuNet is trained
    # around a fixed input scale, and handing it a 3000 px phone portrait was
    # measured at 23 ms against 6 ms here — and it found fewer faces, not more.
    face_detection_size: int = Field(
        default=320, alias="FACE_DETECTION_SIZE"
    )  # 320 faster, 640 more accurate
    # The lowest threshold `/face/match` will search at, whatever the caller
    # asks for. The kiosk sends SFace's own 0.363; a caller sending 0 would
    # otherwise get the nearest enrolled person back for any vector at all.
    face_match_min_threshold: float = Field(default=0.3, alias="FACE_MATCH_MIN_THRESHOLD")

    # Width of the face vectors in Qdrant. 512 = InsightFace ArcFace, which is
    # what the existing index was seeded with; 128 = SFace, which luna-edge now
    # computes. A collection is created at one width and rejects any other, so
    # changing this requires recreating the collection AND re-seeding every face
    # from the same model that will later query it.
    # SFace, 128. Must equal what the kiosk sends: the edge embeds with the
    # same model, and a vector of the wrong length is not a poor match but an
    # impossible comparison. Changing it means recreating the collection and
    # re-seeding, because vectors of two sizes cannot share an index.
    face_embedding_dim: int = Field(default=128, alias="FACE_EMBEDDING_DIM")
    sface_model_path: str = Field(
        default="storage/models/face_recognition_sface_2021dec.onnx",
        alias="SFACE_MODEL_PATH",
    )
    yunet_model_path: str = Field(
        default="storage/models/face_detection_yunet_2023mar.onnx",
        alias="YUNET_MODEL_PATH",
    )
    # YuNet's own confidence gate, and the NMS overlap above which two boxes
    # are judged to be the same face. Enrolment photos are posed and well lit,
    # so the default is deliberately strict: a second, spurious face in the
    # frame is worse here than a missed one, because it enrols a stranger
    # under somebody else's name.
    face_detection_threshold: float = Field(default=0.6, alias="FACE_DETECTION_THRESHOLD")
    face_detection_nms: float = Field(default=0.3, alias="FACE_DETECTION_NMS")

    face_max_batch_size: int = 100  # Max faces per request
    face_max_image_size: int = 10 * 1024 * 1024  # 10MB
    # A composed photobooth frame is a few hundred KB. The ceiling is for a
    # client that sends something else entirely, not for a large photo.
    photo_max_image_size: int = 10 * 1024 * 1024  # 10MB

    persist_dir: str = "storage/index"
    rag_max_file_size: int = 50 * 1024 * 1024  # 50MB
    rag_allowed_extensions: set[str] = {".pdf", ".txt", ".docx", ".doc", ".md"}
    rag_chunk_strategy: str = Field(
        default="semantic", alias="RAG_CHUNK_STRATEGY"
    )  # "token" | "semantic"
    rag_chunk_size: int = Field(default=512, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=50, alias="RAG_CHUNK_OVERLAP")
    # Number of chunks the rag_search tool retrieves from the Udinus KB.
    rag_top_k: int = Field(default=3, alias="RAG_TOP_K")
    # Register the `rag_search` tool at all. Set false while the Qdrant KB is
    # empty: rag_search then always returns "nothing found", which costs a full
    # extra LLM round-trip before the model falls back to search_web (measured
    # ~1.6s per campus question). With it off, the tool policy switches to a
    # web-first variant — see prompts/tool-policy-*.txt. Flip back to true after
    # seeding (`python -m src.seeder rag`).
    enable_rag_tool: bool = Field(default=True, alias="ENABLE_RAG_TOOL")

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

    # Web search (Exa) — backs the search_web agent tool, used for current/external
    # info or when rag_search returns no relevant Udinus result.
    # Empty/unset key → search_web reports itself unavailable and the model
    # answers from rag_search + its own knowledge.
    exa_api_key: str | None = Field(default=None, alias="EXA_API_KEY")
    web_search_count: int = Field(default=3, alias="WEB_SEARCH_COUNT")  # k results
    # Exa: fetch full page text (slower, richer) vs highlights-only (faster —
    # embedding-selected snippets, good for short spoken answers). Voice default
    # is highlights-only; flip true if web answers come back too thin.
    exa_fetch_text: bool = Field(default=False, alias="EXA_FETCH_TEXT")
    # Exa livecrawl: None/"never" (cached, fast) | "fallback" | "always" (fresh,
    # slower). Set "fallback" if you need the latest campus announcements.
    exa_livecrawl: str | None = Field(default=None, alias="EXA_LIVECRAWL")
    # Optional domain allow-list (comma-separated), e.g. "dinus.ac.id". Empty =
    # general web. Restricting makes campus-current-info come from official
    # sources but BREAKS general queries (news/public figures) — leave empty
    # unless web is used ONLY for campus.
    exa_include_domains: CsvList = Field(
        default_factory=list, alias="EXA_INCLUDE_DOMAINS"
    )
    # Domains the `scope="campus"` branch of search_web is pinned to. Measured
    # motivation: when the model rewrites a campus query into English, an
    # unpinned Exa search returns LinkedIn profiles (including other
    # universities' rectors) instead of dinus.ac.id. Pinning makes the campus
    # branch deterministic; `scope="general"` stays unfiltered so news/public-
    # figure questions still work. Matches subdomains (warta./publikasi./repo.).
    exa_campus_domains: CsvList = Field(
        default_factory=lambda: ["dinus.ac.id"], alias="EXA_CAMPUS_DOMAINS"
    )

    # Speech-to-Text (Deepgram, batch). The kiosk owns endpointing, so the whole
    # turn is transcribed once, at turn_end.
    enable_stt: bool = Field(default=True, alias="ENABLE_STT")
    deepgram_api_key: str | None = Field(default=None, alias="DEEPGRAM_API_KEY")
    # nova-3 has supported Indonesian since Jan 2026. Older docs predate that —
    # check the live language table before assuming a language is nova-2-only.
    deepgram_model: str = Field(default="nova-3", alias="DEEPGRAM_MODEL")
    deepgram_language: str = Field(default="id", alias="DEEPGRAM_LANGUAGE")
    # Punctuation + number/date formatting. Costs nothing extra and makes the
    # transcript read better in the LLM prompt and the chat transcript.
    deepgram_smart_format: bool = Field(default=True, alias="DEEPGRAM_SMART_FORMAT")
    # A hung request blocks the turn, so fail fast rather than let the visitor
    # wait for the kiosk's own reply watchdog.
    deepgram_timeout_seconds: int = Field(default=15, alias="DEEPGRAM_TIMEOUT_SECONDS")
    # Deepgram closes an idle connection after about five seconds, and turns
    # are always further apart than that, so every transcription paid a new
    # TLS handshake: measured 1440 ms cold against ~910 ms kept warm for 2.5 s
    # of audio. While a turn is being spoken the connection is touched this
    # often, so it is still open at turn_end.
    stt_keep_warm_interval_seconds: float = Field(
        default=4.0, alias="STT_KEEP_WARM_INTERVAL_SECONDS"
    )

    # Turn WebSocket (/turn/stream)
    enable_voice_ws: bool = Field(default=True, alias="ENABLE_VOICE_WS")
    # PCM-16 mono sample rate the kiosk streams. A kiosk may override it per
    # connection with `?sample_rate=`; this is what an older client that sends
    # no parameter is assumed to be using.
    voice_sample_rate: int = Field(default=16000, alias="VOICE_SAMPLE_RATE")
    # Ceiling on one turn's audio. The kiosk decides when a turn ends — this is
    # the guard for a kiosk that crashes mid-turn and never sends `turn_end`,
    # so the buffer cannot grow without bound.
    voice_max_utterance_seconds: float = Field(
        default=30.0, alias="VOICE_MAX_UTTERANCE_SECONDS"
    )

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

    # Cartesia TTS (cloud streaming, TTS_PROVIDER=cartesia)
    # Accepts a comma-separated list. Keys are tried in order and a key is only
    # abandoned when Cartesia says it is out of credits (or the key is dead) —
    # see CartesiaTTSService for what counts as exhausted.
    cartesia_api_key: str | None = Field(default=None, alias="CARTESIA_API_KEY")
    cartesia_voice_id: str = Field(default="", alias="CARTESIA_VOICE_ID")
    cartesia_model_id: str = Field(default="sonic-3.5", alias="CARTESIA_MODEL_ID")
    # 24 kHz, not 44.1: the reply is speech, the kiosk resamples to its output
    # anyway, and every chunk is base64 in JSON across two hops and decoded on
    # the Jetson's main thread. 44.1 kHz was 1.8x the bytes for no audible gain.
    cartesia_sample_rate: int = Field(default=24000, alias="CARTESIA_SAMPLE_RATE")
    # How long a TTS stream may go without a chunk before the sentence is given
    # up on. Without it a hung stream held the reply until the whole pipeline
    # timed out, silently.
    tts_stream_idle_timeout_seconds: float = Field(
        default=10.0, alias="TTS_STREAM_IDLE_TIMEOUT_SECONDS"
    )
    cartesia_language: str = Field(default="id", alias="CARTESIA_LANGUAGE")

    @property
    def cartesia_api_keys(self) -> list[str]:
        """Cartesia keys in failover order, parsed from the comma-separated env."""
        return [k.strip() for k in (self.cartesia_api_key or "").split(",") if k.strip()]

    # Rate limiting constants
    frame_timeout_seconds: float = 5.0

    # CORS
    cors_origins: CsvList = Field(default_factory=lambda: ["*"], alias="CORS_ORIGINS")

    # Request / streaming
    request_id_header: str = Field(default="X-Request-ID", alias="REQUEST_ID_HEADER")

    @field_validator("exa_include_domains", "exa_campus_domains", mode="before")
    @classmethod
    def _parse_exa_include_domains(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v

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
