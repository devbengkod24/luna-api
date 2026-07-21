"""STT adapter implementation using the Deepgram cloud API."""

import base64
import io
import logging
import wave

from src.config import settings
from src.ports.services import ISTTService

logger = logging.getLogger(__name__)


class DeepgramSTTService(ISTTService):
    """STT adapter using Deepgram prerecorded transcription (`/v1/listen`).

    Set STT_PROVIDER=deepgram and DEEPGRAM_API_KEY to activate; WhisperSTTService
    stays available under STT_PROVIDER=whisper.

    Uses the REST (prerecorded) endpoint rather than the streaming WebSocket
    because the caller already owns turn detection: the voice WS segmenter runs
    VAD and the device-directedness gate, then hands over one complete utterance.
    A streaming connection would duplicate that endpointing and would have to be
    opened before the gate decides whether the utterance is even for Luna.
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        language: str | None = None,
        smart_format: bool | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("DeepgramSTTService requires an API key.")

        # Imported lazily so the adapter package stays importable without
        # deepgram-sdk installed when STT_PROVIDER=whisper.
        from deepgram import AsyncDeepgramClient

        self.model = model or settings.deepgram_model
        self.language = language or settings.deepgram_language
        self.smart_format = (
            settings.deepgram_smart_format if smart_format is None else smart_format
        )
        self.timeout_seconds = timeout_seconds or settings.deepgram_timeout_seconds

        self._client = AsyncDeepgramClient(api_key=api_key)
        logger.info(
            "STT initialized: provider=deepgram model=%s language=%s smart_format=%s",
            self.model, self.language, self.smart_format,
        )

    @staticmethod
    def _extract_transcript(response) -> str:
        """Pull the first alternative's transcript out of a Deepgram response."""
        results = getattr(response, "results", None)
        channels = getattr(results, "channels", None) or []
        if not channels:
            return ""
        alternatives = getattr(channels[0], "alternatives", None) or []
        if not alternatives:
            return ""
        return (getattr(alternatives[0], "transcript", "") or "").strip()

    @staticmethod
    def _pcm_to_wav(pcm16: bytes, sample_rate: int) -> bytes:
        """Wrap raw mono PCM16 in a 44-byte WAV header.

        The SDK's prerecorded signature exposes `encoding` but not `sample_rate`,
        so raw `linear16` uploads have no way to declare their rate. A WAV
        container carries it in the header and Deepgram sniffs it — cheaper than
        threading the rate through `additional_query_parameters`.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm16)
        return buf.getvalue()

    async def _transcribe_bytes(self, audio: bytes) -> str:
        """Call the prerecorded endpoint with a self-describing container.

        No `encoding` is declared: Deepgram reads the format from the container
        header (WAV/MP3/OGG), and declaring the wrong one yields garbage output.
        """
        response = await self._client.listen.v1.media.transcribe_file(
            request=audio,
            model=self.model,
            language=self.language,
            smart_format=self.smart_format,
            request_options={"timeout_in_seconds": self.timeout_seconds},
        )
        return self._extract_transcript(response)

    async def transcribe(self, speech_base64: str) -> str:
        try:
            audio_data = base64.b64decode(speech_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {e}")

        if not audio_data:
            raise ValueError("Empty audio data")

        try:
            transcript = await self._transcribe_bytes(audio_data)
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

        if not transcript:
            raise RuntimeError("Transcription resulted in empty text")

        logger.debug("Transcription complete: %d chars", len(transcript))
        return transcript

    async def transcribe_pcm(self, pcm16: bytes, sample_rate: int) -> str:
        """Transcribe raw 16-bit mono PCM. Returns "" if nothing was heard."""
        if not pcm16:
            return ""

        try:
            transcript = await self._transcribe_bytes(self._pcm_to_wav(pcm16, sample_rate))
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

        logger.debug("PCM transcription: %d chars", len(transcript))
        return transcript
