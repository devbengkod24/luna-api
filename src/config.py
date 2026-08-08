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
    debug: bool = Field(default=True, alias="DEBUG")
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

    # Emotion Detection
    enable_emotion_detection: bool = Field(
        default=True, alias="ENABLE_EMOTION_DETECTION"
    )
    # EmotiEffLib model name — must be one of emotiefflib.facial_analysis
    # .get_model_list(). "enet_b2_7" is the 7-class AffectNet EfficientNet-B2
    # (30.8 MB) and its labels map 1:1 onto the vocabulary the rest of Luna
    # speaks. The "_8" variants add a Contempt class; "_va_mtl" ones also emit
    # valence/arousal. Note there is no "enet_b2_8_best" ONNX in the upstream
    # repo despite the .pt of that name — only enet_b2_8 and enet_b2_7.
    #
    # Weights are baked into the image at build time (see Dockerfile). Changing
    # this to another name means rebuilding, or the library downloads it at
    # first use — a network round-trip during kiosk start-up.
    emotiefflib_model: str = Field(
        default="enet_b2_7", alias="EMOTIEFFLIB_MODEL"
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
    whisper_model: str = Field(default="small", alias="WHISPER_MODEL")
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
        default=27, alias="VOICE_SPEECH_END_FRAMES"
    )  # consecutive silence frames to end utterance (~800ms). Lower = snappier
    # turn-taking but risks cutting off hesitant speakers; raise toward 25 (~750ms)
    # if the kiosk audience pauses mid-sentence a lot.
    voice_prebuffer_frames: int = Field(
        default=7, alias="VOICE_PREBUFFER_FRAMES"
    )  # frames retained before speech_start (~210ms)
    voice_max_utterance_seconds: float = Field(
        default=30.0, alias="VOICE_MAX_UTTERANCE_SECONDS"
    )

    # --- Barge-in (talking over the assistant) ---
    # "vad"     — the user can just start talking; sustained speech while the
    #             reply plays cancels it (fused with the facing signal below).
    # "hotword" — only the client-side "stop"/"berhenti" listener and the manual
    #             button can interrupt. Fall back to this if the kiosk's speaker
    #             leaks past echo cancellation badly enough to false-trigger.
    voice_barge_in_mode: str = Field(default="vad", alias="VOICE_BARGE_IN_MODE")
    # Barge-in is decided over a SLIDING WINDOW, not a consecutive run. Requiring
    # an unbroken streak was too brittle: human speech has micro-gaps between
    # words, and during double-talk the browser's echo canceller attenuates the
    # near-end talker in bursts — so a single sub-threshold frame kept resetting
    # the count and barge-in rarely fired. A ratio over a window tolerates those
    # dropouts while still rejecting TTS leak, which is intermittent rather than
    # dense. Window is also how much of the user's opening words is lost, since
    # audio captured during detection is discarded as contaminated.
    voice_barge_in_window_frames: int = Field(
        default=15, alias="VOICE_BARGE_IN_WINDOW_FRAMES"
    )  # 15 x 30 ms = 450 ms of context
    voice_barge_in_ratio: float = Field(default=0.6, alias="VOICE_BARGE_IN_RATIO")
    # Frames dropped after a barge-in is honoured, covering the round-trip until
    # the client actually silences the speaker plus the echo canceller's
    # re-convergence. Everything captured in that gap is part assistant, and
    # splicing it in front of the user's sentence is what ruined the transcript.
    # ~250 ms at 30 ms frames. This is the FIRST knob to raise if barge-in
    # transcripts still come back garbled.
    voice_barge_in_settle_frames: int = Field(
        default=8, alias="VOICE_BARGE_IN_SETTLE_FRAMES"
    )
    # A barge-in is only honoured if the mean facing score over this trailing
    # window clears FACING_THRESHOLD — someone talking to a friend beside the
    # kiosk should not cancel the reply. Fails OPEN: with no facing samples
    # (camera off / voice-only), barge-in works on audio alone.
    voice_barge_in_facing_window_ms: int = Field(
        default=1500, alias="VOICE_BARGE_IN_FACING_WINDOW_MS"
    )
    # Safety net for the "playback finished" signal the client sends once its
    # audio queue drains. The reply's audio is generated much faster than it
    # plays (measured: a 61 s reply finished streaming in 16 s), so the listening
    # gate must stay closed until playback really ends — otherwise the kiosk
    # spends the difference transcribing its own speaker. If the client never
    # reports (disconnect, muted tab), release after this many seconds anyway.
    voice_playback_grace_seconds: float = Field(
        default=120.0, alias="VOICE_PLAYBACK_GRACE_SECONDS"
    )

    # Device-Directed Speech Detection (Active Speaker Detection + facing fusion)
    # When enabled, a finished VAD utterance is only committed to STT/LLM/TTS if
    # the candidate face was both facing the kiosk AND its lip motion correlated
    # with the audio (LR-ASD). Fail-open: if ASD can't load or no face is tracked,
    # the utterance is treated as directed so the system never goes deaf.
    enable_asd: bool = Field(default=False, alias="ENABLE_ASD")
    # ONNX runtime (3 sessions: audio/visual frontends + AV backend). Models are
    # the deeptalk re-export of LR-ASD MIT — see storage/models/asd/README.md.
    asd_audio_model_path: str = Field(
        default="storage/models/asd/audio_frontend.onnx",
        alias="ASD_AUDIO_MODEL_PATH",
    )
    asd_visual_model_path: str = Field(
        default="storage/models/asd/visual_frontend.onnx",
        alias="ASD_VISUAL_MODEL_PATH",
    )
    asd_av_model_path: str = Field(
        default="storage/models/asd/av_backend.onnx",
        alias="ASD_AV_MODEL_PATH",
    )
    # ONNX runtime intra-op threads. Bench: 2 is the sweet spot on typical
    # 2-core kiosks (~2× faster than PyTorch); 1 erodes the win, >2 hurts on
    # small models. ASD runs in short bursts at SPEECH_END so claiming threads
    # is fine — LLM hasn't started yet.
    asd_num_threads: int = Field(default=2, alias="ASD_NUM_THREADS")
    # Per-frame speaking probability gate (sigmoid of av_backend logit, 0..1).
    # The ONNX head emits a single class logit (not the 2-class softmax of the
    # original PyTorch path), so this threshold runs higher than the old 0.4 —
    # 0.5 is a sane default; tune on real speaking input.
    asd_score_threshold: float = Field(default=0.5, alias="ASD_SCORE_THRESHOLD")
    # Fraction of the utterance's frames that must clear asd_score_threshold for
    # the candidate to count as the active speaker.
    asd_presence_ratio: float = Field(default=0.35, alias="ASD_PRESENCE_RATIO")
    # Nominal video fps used to scale the LR-ASD MFCC window when the real crop
    # fps can't be measured (winlen=0.025*25/fps, winstep=0.010*25/fps). The
    # session router measures the actual delivered fps per utterance; this is the
    # fallback. LR-ASD is calibrated to 25 fps — see asd_service.py.
    asd_nominal_fps: float = Field(default=25.0, alias="ASD_NOMINAL_FPS")
    # Mean facing score (FE MediaPipe metadata, 0..1) required over the utterance.
    # The FE score is "fraction of margin left to the frontal-cone boundary along
    # the worst axis" (min of the yaw/pitch/proximity ramps, pitch ramp peaked at
    # the user's observed neutral pose). So 0.35 means "stayed within the inner
    # 65% of the cone, on average". The mean spans the whole utterance window —
    # including the ~600 ms of trailing silence the VAD needs — so heads that
    # relax at the end drag it down; don't set this near 1.
    #
    # PROVISIONAL. Calibrate from the `facing=` field of the `directedness:` INFO
    # log on the real kiosk before trusting it. Was 0.5 against an older score
    # that multiplied the ramps and peaked at the cone's geometric centre, which
    # rejected ~85% of the poses the FE pre-gate accepts.
    facing_threshold: float = Field(default=0.35, alias="FACING_THRESHOLD")

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
    cartesia_sample_rate: int = Field(default=44100, alias="CARTESIA_SAMPLE_RATE")
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
