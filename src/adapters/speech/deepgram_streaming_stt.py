"""Streaming STT adapter using Deepgram Listen v1 (Streaming: General).

This adapter replaces local VAD as well as the transcriber — Deepgram owns
endpointing here, so `SileroVadService` does not run at all on this path.

Two independent turn-end signals are enabled, and the router needs both:

- `endpointing` is an audio VAD. Fast, but Deepgram's own docs note that
  background noise can stop it from ever seeing silence, which delays or
  suppresses `speech_final` — the exact failure mode of a kiosk in a crowd.
- `utterance_end_ms` is *not* a VAD. It watches word timings in the transcript,
  so it ignores non-speech audio entirely. Slower (≥1000 ms) but noise-proof.

Using only the first goes deaf in a noisy room; only the second is sluggish.
"""

import logging
from collections.abc import AsyncIterator

from src.config import settings
from src.ports.streaming_stt import (
    IStreamingSTTService,
    IStreamingSTTSession,
    StreamingSTTEvent,
    StreamingSTTEventType,
)

logger = logging.getLogger(__name__)


class _DeepgramStreamSession(IStreamingSTTSession):
    def __init__(self, client, params: dict) -> None:
        self._client = client
        self._params = params
        self._cm = None
        self._conn = None

    async def __aenter__(self) -> "_DeepgramStreamSession":
        self._cm = self._client.listen.v1.connect(**self._params)
        self._conn = await self._cm.__aenter__()
        logger.info("Deepgram stream opened (model=%s)", self._params.get("model"))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool | None:
        if self._conn is not None:
            try:
                await self._conn.send_close_stream()
            except Exception as e:
                logger.debug("Deepgram close_stream failed: %s", e)
        cm, self._cm, self._conn = self._cm, None, None
        if cm is None:
            return None
        return await cm.__aexit__(exc_type, exc, tb)

    async def send_audio(self, pcm16: bytes) -> None:
        if self._conn is None or not pcm16:
            return
        await self._conn.send_media(pcm16)

    async def finalize(self) -> None:
        if self._conn is None:
            return
        await self._conn.send_finalize()

    async def keep_alive(self) -> None:
        if self._conn is None:
            return
        await self._conn.send_keep_alive()

    async def events(self) -> AsyncIterator[StreamingSTTEvent]:
        if self._conn is None:
            return

        from deepgram.listen.v1.types import (
            ListenV1Results,
            ListenV1SpeechStarted,
            ListenV1UtteranceEnd,
        )

        async for message in self._conn:
            if isinstance(message, ListenV1SpeechStarted):
                yield StreamingSTTEvent(type=StreamingSTTEventType.SPEECH_STARTED)
                continue

            if isinstance(message, ListenV1UtteranceEnd):
                yield StreamingSTTEvent(type=StreamingSTTEventType.UTTERANCE_END)
                continue

            if isinstance(message, ListenV1Results):
                channel = getattr(message, "channel", None)
                alternatives = getattr(channel, "alternatives", None) or []
                if not alternatives:
                    continue
                transcript = (getattr(alternatives[0], "transcript", "") or "").strip()
                if not transcript:
                    continue
                if getattr(message, "is_final", False):
                    yield StreamingSTTEvent(
                        type=StreamingSTTEventType.FINAL,
                        transcript=transcript,
                        speech_final=bool(getattr(message, "speech_final", False)),
                    )
                else:
                    yield StreamingSTTEvent(
                        type=StreamingSTTEventType.INTERIM, transcript=transcript
                    )


class DeepgramStreamingSTTService(IStreamingSTTService):
    """Opens one Deepgram streaming connection per client WebSocket."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        language: str | None = None,
        endpointing_ms: int | None = None,
        utterance_end_ms: int | None = None,
        keyterms: list[str] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("DeepgramStreamingSTTService requires an API key.")

        from deepgram import AsyncDeepgramClient

        self.model = model or settings.deepgram_model
        self.language = language or settings.deepgram_language
        self.endpointing_ms = (
            settings.deepgram_endpointing_ms if endpointing_ms is None else endpointing_ms
        )
        self.utterance_end_ms = (
            settings.deepgram_utterance_end_ms
            if utterance_end_ms is None
            else utterance_end_ms
        )
        self.keyterms = keyterms if keyterms is not None else settings.deepgram_keyterm_list

        self._client = AsyncDeepgramClient(api_key=api_key)
        logger.info(
            "Streaming STT initialized: provider=deepgram model=%s language=%s "
            "endpointing=%sms utterance_end=%sms keyterms=%d",
            self.model, self.language, self.endpointing_ms,
            self.utterance_end_ms, len(self.keyterms),
        )

    def open(self, sample_rate: int) -> IStreamingSTTSession:
        params: dict = {
            "model": self.model,
            "language": self.language,
            # Raw PCM from the browser has no container, so the format must be
            # declared. Unlike the prerecorded endpoint, connect() does expose
            # sample_rate and channels.
            "encoding": "linear16",
            "sample_rate": sample_rate,
            "channels": 1,
            "interim_results": True,
            "vad_events": True,
            "endpointing": self.endpointing_ms,
            "utterance_end_ms": self.utterance_end_ms,
            "smart_format": settings.deepgram_smart_format,
            "punctuate": True,
            "numerals": True,
        }
        # keyterm is a nova-3 feature; sending it to an older model is rejected.
        if self.keyterms and self.model.startswith("nova-3"):
            params["keyterm"] = self.keyterms
        return _DeepgramStreamSession(self._client, params)
