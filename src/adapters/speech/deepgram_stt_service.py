"""STT adapter implementation using the Deepgram cloud API."""

import base64
import io
import logging
import wave

import httpx

from src.config import settings
from src.ports.services import ISTTService

logger = logging.getLogger(__name__)


class DeepgramSTTService(ISTTService):
    """STT adapter using Deepgram prerecorded transcription (`/v1/listen`).

    The only STT adapter on this branch; exp/edge carries local Whisper instead.

    Uses the REST (prerecorded) endpoint rather than the streaming WebSocket
    because the kiosk owns turn detection: `/turn/stream` receives one turn the
    kiosk already decided was for Luna and transcribes it once, at `turn_end`.
    A streaming connection would duplicate that endpointing, and bill for audio
    the kiosk never meant to send.
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        language: str | None = None,
        smart_format: bool | None = None,
        timeout_seconds: int | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("DeepgramSTTService requires an API key.")

        # Imported lazily so the adapter package stays importable without
        # deepgram-sdk (unit tests, the seeder); a missing SDK then surfaces in
        # _build_stt's log instead of failing the whole app at import.
        from deepgram import AsyncDeepgramClient

        self.model = model or settings.deepgram_model
        self.language = language or settings.deepgram_language
        self.smart_format = (
            settings.deepgram_smart_format if smart_format is None else smart_format
        )
        self.timeout_seconds = timeout_seconds or settings.deepgram_timeout_seconds

        # Our own client, handed to the SDK, so `keep_warm` touches exactly the
        # connection pool transcription will draw from.
        self._http = http_client or httpx.AsyncClient(timeout=self.timeout_seconds)
        self._client = AsyncDeepgramClient(api_key=api_key, httpx_client=self._http)
        try:
            self._base_url = self._client._client_wrapper.get_environment().base
        except Exception:
            self._base_url = "https://api.deepgram.com"
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

    async def keep_warm(self) -> None:
        """Touch Deepgram so the pooled connection is still open at turn_end.

        Deepgram closes a connection idle for about five seconds, and the gap
        between one turn's transcription and the next is always longer, so each
        turn paid a fresh TLS handshake: 1440 ms cold against ~910 ms warm for
        2.5 s of audio, measured. An unauthenticated GET is refused for free
        and leaves the connection open.
        """
        try:
            await self._http.get(self._base_url, timeout=3.0)
        except Exception as e:
            logger.debug("Deepgram keep-warm failed (%s)", e)

    async def close(self) -> None:
        await self._http.aclose()

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
