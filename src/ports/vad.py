"""Voice activity detection port."""

from abc import ABC, abstractmethod


class IVadService(ABC):
    """Per-frame voice activity detector."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Expected audio sample rate in Hz."""
        ...

    @property
    @abstractmethod
    def frame_duration_ms(self) -> int:
        """Expected frame size in milliseconds."""
        ...

    @property
    def frame_bytes(self) -> int:
        """Bytes per frame for 16-bit PCM mono audio at the configured rate."""
        return int(self.sample_rate * self.frame_duration_ms / 1000) * 2

    @abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        """Return True if the PCM-16 mono frame contains voice."""
        ...
