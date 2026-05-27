"""Silero VAD adapter (neural, robust against non-stationary noise).

Silero ships an ONNX model + a thin PyTorch wrapper. Compared to WebRTC VAD
it produces a probability per frame (0.0 – 1.0) instead of a hard 0/1, which
generalises far better in noisy environments (public kiosks, crowds, music,
HVAC) where energy-based VAD over-triggers.

This adapter sticks to the same `IVadService` contract so it's a drop-in
replacement for `WebRtcVadService` — the segmenter is unchanged.
"""

import logging
from typing import Optional

import numpy as np
import torch

from src.ports.vad import IVadService

logger = logging.getLogger(__name__)

# Silero's window size is fixed for accurate inference:
#   - 512 samples at 16 kHz  (~32 ms)
#   - 256 samples at  8 kHz  (~32 ms)
# We accept any caller-configured frame_duration_ms but internally re-window
# the audio so the model gets its native chunk size.
_SILERO_WINDOWS = {16000: 512, 8000: 256}


class SileroVadService(IVadService):
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        threshold: float = 0.5,
    ):
        if sample_rate not in _SILERO_WINDOWS:
            raise ValueError(
                f"Silero supports {list(_SILERO_WINDOWS.keys())} Hz, got {sample_rate}"
            )

        # Lazy import — `silero_vad` package pulls in torchaudio metadata; we
        # delay the import so the rest of the app boots even if the dep is
        # missing (factory will fall back to WebRTC).
        from silero_vad import load_silero_vad

        self._model = load_silero_vad(onnx=False)
        self._model.eval()
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._threshold = float(threshold)
        self._window_samples = _SILERO_WINDOWS[sample_rate]

        # Holds residual samples that didn't fill a full 512-sample window on
        # the previous push. Carrying them forward avoids dropping audio when
        # the caller's frame_duration_ms doesn't divide evenly into 32 ms.
        self._pending: Optional[np.ndarray] = None

        # Track last decision so a *short* frame (smaller than the Silero
        # window) doesn't silently report False. We replay the previous
        # speech/silence verdict for partial frames.
        self._last_speech = False

        logger.info(
            "SileroVadService initialized: sr=%d frame_ms=%d threshold=%.2f window=%d",
            sample_rate, frame_duration_ms, threshold, self._window_samples,
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
            # PCM-16 little-endian -> float32 in [-1, 1]
            pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            if self._pending is not None and self._pending.size > 0:
                pcm = np.concatenate([self._pending, pcm])

            # Run Silero on every 512-sample window we have. The last verdict
            # in the batch is what we return for this frame. The model is
            # stateful (RNN) — sequential calls maintain hidden state across
            # the streaming session.
            window = self._window_samples
            speech = self._last_speech
            i = 0
            while i + window <= pcm.size:
                chunk = torch.from_numpy(pcm[i : i + window].copy())
                with torch.no_grad():
                    out = self._model(chunk, self._sample_rate)
                prob = float(out.flatten()[0].item())
                speech = prob >= self._threshold
                i += window

            # Carry the leftover (< window) into the next call.
            self._pending = pcm[i:].copy() if i < pcm.size else None
            self._last_speech = speech
            return speech
        except Exception:
            logger.debug("Silero is_speech failed", exc_info=True)
            return False
