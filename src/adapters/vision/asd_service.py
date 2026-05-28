"""LR-ASD active-speaker adapter.

Wraps the vendored LR-ASD network (`_lrasd/`) behind `IActiveSpeakerService`.
Given a face-crop sequence for one track plus the corresponding audio segment,
returns a per-frame "speaking" probability (0..1) — high when the visible lip
motion correlates with the audio (the device-directed signal).

Preprocessing mirrors the upstream demo (`Columbia_test.py`):
  - video: BGR crop → grayscale → 112x112, fed as raw 0..255 pixel values.
  - audio: 13-dim MFCC at 100 fps (winlen 25ms / winstep 10ms), 4 audio frames
    per video frame.

Inference is synchronous (PyTorch); callers reach it via `asyncio.to_thread`.
"""

import asyncio
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import settings
from src.domain.image import Image
from src.ports.vision import IActiveSpeakerService

from ._lrasd.model import ASD_Model

logger = logging.getLogger(__name__)

_CROP_SIZE = 112
_MFCC_PER_FRAME = 4  # 100 fps audio MFCC / 25 fps video


class _ScoreHead(nn.Module):
    """Mirrors the checkpoint's `lossAV.FC` / `lossV.FC` linear heads."""

    def __init__(self) -> None:
        super().__init__()
        self.FC = nn.Linear(128, 2)


class _LRASDNet(nn.Module):
    """Container matching the saved ASD state_dict key layout for clean load."""

    def __init__(self) -> None:
        super().__init__()
        self.model = ASD_Model()
        self.lossAV = _ScoreHead()
        self.lossV = _ScoreHead()


class ActiveSpeakerService(IActiveSpeakerService):
    """Active speaker detection using the vendored LR-ASD network."""

    def __init__(self, model_path: str | None = None, device: str | None = None):
        import python_speech_features  # local import: optional dep, fail-soft upstream

        self._mfcc = python_speech_features.mfcc
        self.model_path = model_path or settings.asd_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        net = _LRASDNet()
        state = torch.load(self.model_path, map_location="cpu")
        # The checkpoint stores the full ASD module (model.* + lossAV/lossV.FC).
        # strict=False tolerates the training-only keys we don't reconstruct.
        missing, unexpected = net.load_state_dict(state, strict=False)
        critical = [k for k in missing if k.startswith(("model.", "lossAV."))]
        if critical:
            raise RuntimeError(f"LR-ASD checkpoint missing required keys: {critical[:5]}")

        net.eval()
        net.to(self.device)
        self._net = net
        logger.info(
            "ActiveSpeakerService initialized: model=%s device=%s",
            self.model_path, self.device,
        )

    # --- preprocessing ---

    def _crops_to_tensor(self, face_crops: list[Image]) -> np.ndarray:
        frames = []
        for img in face_crops:
            raw = img.raw
            if raw is None:
                continue
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
            if gray.shape[:2] != (_CROP_SIZE, _CROP_SIZE):
                gray = cv2.resize(gray, (_CROP_SIZE, _CROP_SIZE))
            frames.append(gray)
        if not frames:
            return np.empty((0, _CROP_SIZE, _CROP_SIZE), dtype=np.float32)
        return np.stack(frames).astype(np.float32)

    def _score_sync(self, face_crops: list[Image], pcm16: bytes, sample_rate: int) -> list[float]:
        if sample_rate != 16000:
            raise ValueError(f"ASD requires 16 kHz audio, got {sample_rate} Hz")
        if not pcm16 or not face_crops:
            return []

        video = self._crops_to_tensor(face_crops)
        if video.shape[0] == 0:
            return []

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        mfcc = self._mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

        # Align: 4 audio MFCC frames per video frame; truncate both to a common length.
        n_v = min(video.shape[0], mfcc.shape[0] // _MFCC_PER_FRAME)
        if n_v < 1:
            return []
        audio_feat = mfcc[: n_v * _MFCC_PER_FRAME, :]
        video_feat = video[:n_v, :, :]

        with torch.no_grad():
            input_a = torch.FloatTensor(audio_feat).unsqueeze(0).to(self.device)
            input_v = torch.FloatTensor(video_feat).unsqueeze(0).to(self.device)
            embed_a = self._net.model.forward_audio_frontend(input_a)
            embed_v = self._net.model.forward_visual_frontend(input_v)
            out = self._net.model.forward_audio_visual_backend(embed_a, embed_v)
            logits = self._net.lossAV.FC(out)  # (n_v, 2)
            probs = F.softmax(logits, dim=-1)[:, 1]
            return [float(p) for p in probs.detach().cpu().numpy()]

    async def score(
        self, face_crops: list[Image], pcm16: bytes, sample_rate: int
    ) -> list[float]:
        return await asyncio.to_thread(self._score_sync, face_crops, pcm16, sample_rate)
