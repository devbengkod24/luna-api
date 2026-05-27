"""WebRTC-VAD adapter (Google's WebRTC VAD, ported to Python)."""

import logging

import webrtcvad

from src.ports.vad import IVadService

logger = logging.getLogger(__name__)

_SUPPORTED_RATES = {8000, 16000, 32000, 48000}
_SUPPORTED_FRAMES_MS = {10, 20, 30}


class WebRtcVadService(IVadService):
    def __init__(
        self,
        aggressiveness: int = 2,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
    ):
        if sample_rate not in _SUPPORTED_RATES:
            raise ValueError(f"sample_rate must be one of {_SUPPORTED_RATES}")
        if frame_duration_ms not in _SUPPORTED_FRAMES_MS:
            raise ValueError(f"frame_duration_ms must be one of {_SUPPORTED_FRAMES_MS}")

        self._vad = webrtcvad.Vad(aggressiveness)
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        logger.info(
            "WebRtcVadService initialized: aggressiveness=%d sr=%d frame_ms=%d",
            aggressiveness, sample_rate, frame_duration_ms,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_duration_ms(self) -> int:
        return self._frame_duration_ms

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != self.frame_bytes:
            return False
        try:
            return self._vad.is_speech(frame, self._sample_rate)
        except Exception:
            logger.debug("VAD is_speech failed", exc_info=True)
            return False
