"""Streaming speech-to-text port.

Distinct from `ISTTService` (in `services.py`) on purpose: that one is
request/response — hand over a finished utterance, get a string back. A
streaming provider inverts the control flow. Audio is pushed continuously and
turn boundaries arrive as *events* from the provider, so the caller no longer
owns endpointing.

Implementations therefore also replace local VAD, not just the transcriber.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum


class StreamingSTTEventType(Enum):
    SPEECH_STARTED = "speech_started"
    # Partial hypothesis; will be revised. Display only, never commit.
    INTERIM = "interim"
    # Confirmed segment. `speech_final=True` additionally means the turn ended.
    FINAL = "final"
    # Turn ended per word-gap analysis rather than audio silence. Backstop for
    # when background noise stops the provider's VAD from ever seeing silence.
    UTTERANCE_END = "utterance_end"


@dataclass
class StreamingSTTEvent:
    type: StreamingSTTEventType
    transcript: str = ""
    speech_final: bool = False


class IStreamingSTTSession(ABC):
    """One provider connection, scoped to a client WebSocket via `async with`."""

    @abstractmethod
    async def __aenter__(self) -> "IStreamingSTTSession": ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb) -> bool | None: ...

    @abstractmethod
    async def send_audio(self, pcm16: bytes) -> None:
        """Push raw PCM16 mono. Chunk size is free — no framing required."""
        ...

    @abstractmethod
    async def finalize(self) -> None:
        """Force the provider to flush its current partial as a final."""
        ...

    @abstractmethod
    async def keep_alive(self) -> None:
        """Hold the connection open without sending audio (and without cost)."""
        ...

    @abstractmethod
    def events(self) -> AsyncIterator[StreamingSTTEvent]:
        """Yield provider events until the connection closes."""
        ...


class IStreamingSTTService(ABC):
    """Factory for per-connection streaming sessions."""

    @abstractmethod
    def open(self, sample_rate: int) -> IStreamingSTTSession:
        """Build an unstarted session. Enter it with `async with` to connect."""
        ...
