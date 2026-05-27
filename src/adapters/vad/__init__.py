"""VAD adapters."""

from .silero_vad_service import SileroVadService
from .webrtc_vad_service import WebRtcVadService

__all__ = ["SileroVadService", "WebRtcVadService"]
