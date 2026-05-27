"""STT adapter implementation using faster-whisper."""

import asyncio
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from src.config import settings
from src.ports.services import ISTTService

logger = logging.getLogger(__name__)


class WhisperSTTService(ISTTService):
    """STT adapter using faster-whisper."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[int] = None,
        compute_type: Optional[str] = "float16",
    ):
        self.model_size = model_size or settings.whisper_model

        if device is not None and device >= 0:
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"

        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(
                "STT initialized: model=%s device=%s compute_type=%s",
                self.model_size, self.device, self.compute_type,
            )
        except Exception as e:
            logger.error("Failed to initialize STT adapter: %s", e)
            raise RuntimeError(f"Failed to initialize STT adapter: {e}")

    def _run_transcribe_path(self, tmp_path: str) -> str:
        segments, _ = self.model.transcribe(
            tmp_path,
            language=settings.whisper_language,
            beam_size=settings.whisper_beam_size,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _run_transcribe_array(self, audio: np.ndarray) -> str:
        # Input comes from voice WS where segmenter has already applied VAD
        # (silero/webrtc per VAD_PROVIDER). Disabling Whisper's internal VAD
        # avoids re-running it and saves 50-150ms per turn. The live voice path
        # also uses a smaller beam (`whisper_voice_beam_size`, default greedy)
        # since latency matters more than the marginal accuracy of beam search
        # on short utterances.
        segments, _ = self.model.transcribe(
            audio,
            language=settings.whisper_language,
            beam_size=settings.whisper_voice_beam_size,
            vad_filter=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def transcribe(self, speech_base64: str) -> str:
        try:
            audio_data = base64.b64decode(speech_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {e}")

        if not audio_data:
            raise ValueError("Empty audio data")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f:
                f.write(audio_data)
                tmp_path = f.name

            transcript = await asyncio.to_thread(self._run_transcribe_path, tmp_path)
            if not transcript:
                raise RuntimeError("Transcription resulted in empty text")

            logger.debug("Transcription complete: %d chars", len(transcript))
            return transcript

        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    async def transcribe_pcm(self, pcm16: bytes, sample_rate: int) -> str:
        """Transcribe raw 16-bit mono PCM. Returns "" if nothing was heard."""
        if not pcm16:
            return ""

        # faster-whisper expects 16 kHz float32 in [-1, 1].
        if sample_rate != 16000:
            raise ValueError(
                f"transcribe_pcm requires 16 kHz audio, got {sample_rate} Hz. "
                "Resample at the caller."
            )

        try:
            audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
            transcript = await asyncio.to_thread(self._run_transcribe_array, audio)
            logger.debug("PCM transcription: %d chars", len(transcript or ""))
            return transcript or ""
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
