"""Text-to-Speech adapter implementations."""

import asyncio
import base64
import io
import logging
import os
import threading
import wave
from collections.abc import Sequence

import numpy as np
import torch

from src.adapters.speech.tts_normalizer import normalize_for_tts
from src.config import settings
from src.ports.services import ITTSService
from src.shared.constants import Character

logger = logging.getLogger(__name__)


def _write_wav(buffer, sample_rate: int, audio_int16: np.ndarray) -> None:
    """PCM-16 mono WAV into a buffer, with the standard library.

    Replaces `scipy.io.wavfile.write`. scipy arrived free as an InsightFace
    dependency and cost 40 MB the moment InsightFace left, which is a poor
    trade for a RIFF header the `wave` module writes natively.
    """
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(np.asarray(audio_int16, dtype=np.int16).tobytes())


def _character_slug(character: int) -> str:
    """Map a character ID to its filesystem slug (luna, vendra, ...).

    Falls back to "default" if the ID is outside the Character enum so callers
    can still look up `default.wav` / `default.txt` as a safety net.
    """
    try:
        return Character(int(character)).slug
    except (ValueError, TypeError):
        return "default"


class MMSTTSService(ITTSService):
    """Text-to-Speech service using Facebook MMS-TTS Indonesian model."""

    MODEL_REPO = "facebook/mms-tts-ind"

    def __init__(self):
        """Initialize the TTS adapter."""
        # Deteksi device (CUDA jika tersedia, fallback ke CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Facebook MMS-TTS on device: {self.device}")

        # Inisialisasi model dan tokenizer
        self.model = None
        self.tokenizer = None
        self.sr = None  # Sample rate
        self._load_model()

    def _load_model(self):
        """Load the Facebook MMS-TTS Indonesian model."""
        try:
            logger.info(f"Loading Facebook MMS-TTS model from {self.MODEL_REPO}...")
            from transformers import VitsModel, AutoTokenizer

            # Load model dan tokenizer
            self.model = VitsModel.from_pretrained(self.MODEL_REPO)
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_REPO)

            # Pindahkan model ke device yang sesuai
            self.model = self.model.to(self.device)

            # Ambil sample rate dari konfigurasi model
            self.sr = self.model.config.sampling_rate

            # Set model ke evaluation mode
            self.model.eval()

            # Kosongkan cache GPU jika menggunakan CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info(
                f"Facebook MMS-TTS model loaded successfully (sample rate: {self.sr} Hz)"
            )

        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def _audio_to_base64(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Convert audio numpy array to base64-encoded WAV string.

        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Base64-encoded WAV audio string
        """
        # Konversi ke 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Tulis ke buffer WAV
        buffer = io.BytesIO()
        _write_wav(buffer, sample_rate, audio_int16)
        buffer.seek(0)

        # Encode ke base64
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        return audio_base64

    async def synthesize(self, text: str, character: int) -> str:
        """Synthesize text to speech audio.

        Args:
            text: Text to synthesize in Indonesian
            character: Character ID (currently unused for MMS-TTS)

        Returns:
            Base64-encoded WAV audio data
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("TTS model not initialized")

        text = normalize_for_tts(text)

        try:
            # DEBUG, not INFO: this is assistant reply content (see CLAUDE.md —
            # conversation text must not land in INFO logs).
            logger.debug("Synthesizing text (chars=%d)", len(text or ""))

            # Off the event loop: inference takes as long as the sentence, and
            # run inline it froze the turn socket and every other request.
            audio_base64 = await asyncio.to_thread(self._synthesize_sync, text)

            logger.info("Text synthesized successfully")
            return audio_base64

        except Exception as e:
            logger.error(f"Failed to synthesize text: {e}")
            raise

    def _synthesize_sync(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs).waveform
        # Output shape: [1, sequence_length]
        return self._audio_to_base64(output.squeeze().cpu().numpy(), self.sr)


class ChatterboxSTTSService(ITTSService):
    """Text-to-Speech service using Chatterbox-TTS Indonesian fine-tune.

    Model source:
    - https://huggingface.co/grandhigh/Chatterbox-TTS-Indonesian
    """

    MODEL_REPO = "grandhigh/Chatterbox-TTS-Indonesian"
    CHECKPOINT_FILENAME = "t3_cfg.safetensors"

    def __init__(
        self,
        device: str | None = None,
        voice_clone_dir: str = "storage/voice",
    ) -> None:
        # Chatterbox bisa jalan di CPU, tapi CUDA sangat disarankan.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.voice_clone_dir = voice_clone_dir

        self.model = None
        self.sr: int | None = None
        self._load_model()

    def _character_prompt_path(self, character: int) -> str | None:
        """Resolve audio prompt path for a given character.

        Convention: `storage/voice/{slug}.wav` where slug is the lowercase
        character name (e.g. `luna.wav`, `vendra.wav`).
        """
        slug = _character_slug(character)
        path = os.path.join(self.voice_clone_dir, f"{slug}.wav")
        return path if os.path.isfile(path) else None

    def _load_model(self) -> None:
        """Load base Chatterbox model and apply Indonesian fine-tune checkpoint."""
        try:
            logger.info("Initializing Chatterbox-TTS on device: %s", self.device)

            from chatterbox.tts import ChatterboxTTS  # type: ignore
            import perth  # type: ignore

            perth.PerthImplicitWatermarker = perth.DummyWatermarker

            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.sr = int(getattr(self.model, "sr", 24000))

            # Apply fine-tune weights (if deps are available).
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
                from safetensors.torch import load_file  # type: ignore

                checkpoint_path = hf_hub_download(
                    repo_id=self.MODEL_REPO,
                    filename=self.CHECKPOINT_FILENAME,
                )
                state = load_file(checkpoint_path, device="cpu")

                # The HF card loads into `model.t3`.
                if not hasattr(self.model, "t3"):
                    raise RuntimeError("ChatterboxTTS model missing attribute `t3`.")
                self.model.t3.load_state_dict(state)

                if self.device == "cuda":
                    torch.cuda.empty_cache()

                logger.info(
                    "Chatterbox-TTS Indonesian checkpoint loaded successfully (sr=%s).",
                    self.sr,
                )
            except Exception as e:
                # Keep service usable (base model) even if checkpoint can't be applied.
                logger.warning(
                    "Failed to apply Indonesian checkpoint; using base Chatterbox model. Error: %s",
                    e,
                )

        except Exception as e:
            logger.error("Failed to initialize Chatterbox-TTS: %s", e)
            raise

    def _audio_to_base64(self, audio_array: np.ndarray, sample_rate: int) -> str:
        # Normalize defensively, then encode as 16-bit PCM WAV.
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("Generated empty audio.")

        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak

        audio_int16 = (audio * 32767.0).astype(np.int16)

        buffer = io.BytesIO()
        _write_wav(buffer, sample_rate, audio_int16)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    async def synthesize(self, text: str, character: int) -> str:
        if self.model is None or self.sr is None:
            raise RuntimeError("Chatterbox TTS model not initialized")

        text = normalize_for_tts(text)

        try:
            logger.debug("Chatterbox synthesizing text (chars=%d)", len(text or ""))

            # Off the event loop, for the same reason as MMS above.
            return await asyncio.to_thread(self._synthesize_sync, text, character)

        except Exception as e:
            logger.error("Failed to synthesize with Chatterbox: %s", e)
            raise

    def _synthesize_sync(self, text: str, character: int) -> str:
        # `generate` usually returns a torch Tensor waveform.
        prompt = self._character_prompt_path(character)
        if prompt:
            wav = self.model.generate(text, audio_prompt_path=prompt)
        else:
            wav = self.model.generate(text)
        audio_np = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        return self._audio_to_base64(np.squeeze(audio_np), int(self.sr))


class F5TTSIndonesianService(ITTSService):
    """Text-to-Speech service using F5-TTS Indonesian fine-tune."""

    MODEL_REPO = "Eempostor/F5-TTS-INDO-FINETUNE-V2"
    MODEL_NAME = "F5TTS_v1_Base"
    CHECKPOINT_FILENAME = "f5_tts_indo_v2.pt"
    VOCAB_FILENAME = "vocab.txt"

    def __init__(
        self,
        device: str | None = None,
        voice_clone_dir: str = "storage/voice",
        model_repo: str = MODEL_REPO,
        model_name: str = MODEL_NAME,
        checkpoint_filename: str = CHECKPOINT_FILENAME,
        vocab_filename: str = VOCAB_FILENAME,
        use_ema: bool = True,
        speed: float = 1.0,
        nfe_step: int = 16,
        cfg_strength: float = 2.0,
        remove_silence: bool = False,
    ) -> None:
        self.device = "cpu"
        self.voice_clone_dir = voice_clone_dir
        self.model_repo = model_repo
        self.model_name = model_name
        self.checkpoint_filename = checkpoint_filename
        self.vocab_filename = vocab_filename
        self.use_ema = use_ema
        self.speed = speed
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.remove_silence = remove_silence

        self.model = None
        self._infer_lock = threading.Lock()
        self._load_model()

    def _load_model(self) -> None:
        """Download the Indonesian fine-tune files and initialize F5-TTS."""
        try:
            logger.info("Initializing F5-TTS on device: %s", self.device)

            from f5_tts.api import F5TTS  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore

            checkpoint_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.checkpoint_filename,
            )
            vocab_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.vocab_filename,
            )

            self.model = F5TTS(
                model=self.model_name,
                ckpt_file=checkpoint_path,
                vocab_file=vocab_path,
                use_ema=self.use_ema,
                device=self.device,
            )

            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("F5-TTS Indonesian fine-tune loaded successfully.")
        except Exception as e:
            logger.error("Failed to initialize F5-TTS: %s", e)
            raise

    def _character_prompt_path(self, character: int) -> str:
        """Resolve the reference audio used by F5-TTS for voice conditioning.

        Convention: `storage/voice/{slug}.wav` (e.g. `luna.wav`), with
        `default.wav` as a last-resort fallback.
        """
        slug = _character_slug(character)
        candidates = [
            os.path.join(self.voice_clone_dir, f"{slug}.wav"),
            os.path.join(self.voice_clone_dir, "default.wav"),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            f"F5-TTS requires a reference WAV file in {self.voice_clone_dir} "
            f"(expected `{slug}.wav` or `default.wav`)."
        )

    def _reference_text_for_prompt(self, ref_audio_path: str) -> str:
        """Read optional sidecar transcription for the reference audio.

        If the sidecar is missing, F5-TTS will transcribe the reference audio itself.
        """
        stem, _ = os.path.splitext(ref_audio_path)
        for path in (f"{stem}.txt", os.path.join(self.voice_clone_dir, "default.txt")):
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as file:
                    return file.read().strip()
        return ""

    def _audio_to_base64(self, audio_array: np.ndarray, sample_rate: int) -> str:
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("Generated empty audio.")

        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak

        audio_int16 = (audio * 32767.0).astype(np.int16)

        buffer = io.BytesIO()
        _write_wav(buffer, sample_rate, audio_int16)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _synthesize_sync(self, text: str, character: int) -> str:
        if self.model is None:
            raise RuntimeError("F5-TTS model not initialized")

        ref_audio_path = self._character_prompt_path(character)
        ref_text = self._reference_text_for_prompt(ref_audio_path)

        with self._infer_lock:
            wav, sample_rate, _ = self.model.infer(
                ref_file=ref_audio_path,
                ref_text=ref_text,
                gen_text=text,
                show_info=lambda *_args, **_kwargs: None,
                speed=self.speed,
                nfe_step=self.nfe_step,
                cfg_strength=self.cfg_strength,
                remove_silence=self.remove_silence,
            )

        return self._audio_to_base64(np.squeeze(wav), int(sample_rate))

    async def synthesize(self, text: str, character: int) -> str:
        text = normalize_for_tts(text)
        if not text or not text.strip():
            raise ValueError("Text to synthesize must not be empty.")

        try:
            logger.info("F5-TTS synthesizing text (chars=%s).", len(text))
            return await asyncio.to_thread(self._synthesize_sync, text, character)
        except Exception as e:
            logger.error("Failed to synthesize with F5-TTS: %s", e)
            raise


def _mask_key(key: str) -> str:
    """Last 4 characters only — API keys must never reach the logs in full."""
    return f"...{key[-4:]}" if len(key) > 4 else "..."


# What counts as "this key is finished" rather than "try again in a moment".
#
# Cartesia documents `quota_exceeded` as needing an account top-up, so retrying
# the same key is pointless. 401/403 mean the key is invalid or revoked, which
# is the same situation from our side. 402 covers a payment-required response
# from API versions that predate structured error codes.
#
# Everything else stays out of this set on purpose — above all
# `concurrency_limited` (429), which is explicitly documented as retryable.
# Treating a brief traffic spike as exhaustion would burn the entire key pool in
# seconds and leave the kiosk mute.
_KEY_EXHAUSTED_ERROR_CODES = {"quota_exceeded"}
_KEY_EXHAUSTED_STATUSES = {401, 402, 403}


def _is_key_exhausted(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) in _KEY_EXHAUSTED_STATUSES:
        return True
    # Structured errors (Cartesia-Version 2026-03-01+) carry the code in the body.
    body = getattr(exc, "body", None)
    return (
        isinstance(body, dict)
        and body.get("error_code") in _KEY_EXHAUSTED_ERROR_CODES
    )


class _CartesiaKeyPool:
    """Ordered API keys with failover to the next one on exhaustion.

    Retirement lasts for the process lifetime. A key that is out of credits will
    keep failing until someone tops the account up, so re-testing it on every
    request would add a doomed round-trip to every single synthesis.
    """

    def __init__(self, keys: Sequence[str]) -> None:
        self._keys = list(keys)
        if not self._keys:
            raise ValueError("CartesiaTTSService requires at least one API key.")
        self._retired: set[str] = set()

    def current(self) -> str | None:
        return next((k for k in self._keys if k not in self._retired), None)

    def retire(self, key: str) -> str | None:
        """Mark `key` spent; return the next usable key, or None if that was the last."""
        if key not in self._retired:
            self._retired.add(key)
            logger.warning(
                "Cartesia key %s exhausted — falling back (%d of %d keys left).",
                _mask_key(key),
                len(self._keys) - len(self._retired),
                len(self._keys),
            )
        return self.current()

    def __len__(self) -> int:
        return len(self._keys)


class CartesiaTTSService(ITTSService):
    """Text-to-Speech adapter using Cartesia cloud API with SSE streaming.

    Streams raw PCM16 chunks per sentence instead of waiting for the full WAV,
    which lets the client start playback before synthesis is complete.
    Set TTS_PROVIDER=cartesia, CARTESIA_API_KEY, and CARTESIA_VOICE_ID to activate.

    `api_key` may be a single key or several; with several, synthesis fails over
    to the next key when the current one runs out of credits.
    """

    supports_streaming = True

    def __init__(
        self,
        api_key: str | Sequence[str],
        voice_id: str,
        model_id: str = "sonic-3.5",
        sample_rate: int = 44100,
        language: str = "id",
    ) -> None:
        self._keys = _CartesiaKeyPool([api_key] if isinstance(api_key, str) else api_key)
        self._voice_id = voice_id
        self._model_id = model_id
        self.sample_rate = sample_rate
        self._language = language
        logger.info(
            "CartesiaTTSService initialized (model=%s, voice=%s, sr=%s, lang=%s, keys=%d)",
            model_id, voice_id, sample_rate, language, len(self._keys),
        )

    async def synthesize(self, text: str, character: int) -> str:
        """Collect full SSE stream → WAV → base64 (backward-compat for REST endpoints)."""
        chunks: list[bytes] = []
        async for pcm in self.synthesize_stream(text, character):
            chunks.append(pcm)
        if not chunks:
            raise ValueError("Cartesia returned no audio for the given text.")
        return self._pcm_to_base64_wav(b"".join(chunks), self.sample_rate)

    async def _stream_with_key(self, transcript: str, api_key: str):
        from cartesia import AsyncCartesia

        async with AsyncCartesia(api_key=api_key) as client:
            stream = await client.tts.generate_sse(
                model_id=self._model_id,
                transcript=transcript,
                voice={"mode": "id", "id": self._voice_id},
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
                language=self._language,
            )
            async for event in stream:
                if event.type == "chunk" and event.audio:
                    yield event.audio

    async def synthesize_stream(
        self, text: str, character: int
    ):
        """Yield raw PCM16 byte chunks via Cartesia SSE API, failing over between keys."""
        normalized = normalize_for_tts(text)
        if not normalized.strip():
            return

        logger.info("Cartesia streaming TTS (chars=%s).", len(normalized))

        while True:
            api_key = self._keys.current()
            if api_key is None:
                raise RuntimeError(
                    "All Cartesia API keys are exhausted — top up the account or "
                    "append another key to CARTESIA_API_KEY."
                )

            streamed = False
            carry = b""
            try:
                stream = self._stream_with_key(normalized, api_key).__aiter__()
                while True:
                    try:
                        # Per chunk, not per sentence: a long sentence legitimately
                        # streams for seconds, a stalled one sends nothing.
                        async with asyncio.timeout(settings.tts_stream_idle_timeout_seconds):
                            pcm = await anext(stream)
                    except StopAsyncIteration:
                        break
                    # Whole samples only. An odd byte waits for the next chunk:
                    # the page cannot play half a sample, and splitting one would
                    # shift every later sample into noise.
                    pcm = carry + pcm
                    usable = len(pcm) - (len(pcm) % 2)
                    carry = pcm[usable:]
                    if usable:
                        streamed = True
                        yield pcm[:usable]
                return
            except Exception as e:
                # Once bytes are on the wire the client is already playing them,
                # so retrying on another key would splice two takes of the same
                # sentence together. Only a failure before the first chunk is
                # recoverable.
                if streamed or not _is_key_exhausted(e):
                    logger.error("Cartesia TTS streaming failed: %s: %s", type(e).__name__, e)
                    raise
                if self._keys.retire(api_key) is None:
                    logger.error("Cartesia TTS failed — no keys left: %s", e)
                    raise

    @staticmethod
    def _pcm_to_base64_wav(pcm: bytes, sample_rate: int) -> str:
        """Wrap raw PCM16 bytes in a WAV container and base64-encode."""
        buf = io.BytesIO()
        with __import__("wave").open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return base64.b64encode(buf.getvalue()).decode()
