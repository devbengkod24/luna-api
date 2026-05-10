"""STT adapter implementation using faster-whisper."""

import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional
import asyncio

from faster_whisper import WhisperModel

from src.ports.services import ISTTService
from src.config import settings

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
                "STT initialized: model=%s, device=%s, compute_type=%s",
                self.model_size,
                self.device,
                self.compute_type,
            )
        except Exception as e:
            logger.error("Failed to initialize STT adapter: %s", e)
            raise RuntimeError(f"Failed to initialize STT adapter: {e}")

    def _run_transcribe(self, tmp_path: str) -> str:
        segments, _ = self.model.transcribe(
            tmp_path,
            language=settings.whisper_language,
            beam_size=settings.whisper_beam_size,
            vad_filter=True,
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

            transcript = await asyncio.to_thread(self._run_transcribe, tmp_path)

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
