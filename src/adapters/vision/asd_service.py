"""LR-ASD active-speaker adapter (ONNX runtime).

Implements `IActiveSpeakerService` using the deeptalk re-export of LR-ASD as
three ONNX sessions (see `storage/models/asd/README.md`):

  audio_frontend.onnx  : MFCC (B, T*4, 13)        -> audio_embed (B, T, 128)
  visual_frontend.onnx : gray crops (B, T, 112,112) -> visual_embed (B, T, 128)
  av_backend.onnx      : (audio_embed, visual_embed) -> per-frame speaking logit

The visual frontend bakes the LR-ASD normalization (`(x/255 - 0.4161)/0.1688`),
so we feed raw 0..255 grayscale. The av-backend output is a single class logit
per frame; we apply sigmoid to get a probability in `[0, 1]`. Inference is
synchronous; callers reach it via `asyncio.to_thread`.
"""

import asyncio
import logging

import cv2
import numpy as np
import onnxruntime as ort

from src.config import settings
from src.domain.image import Image
from src.ports.vision import IActiveSpeakerService

logger = logging.getLogger(__name__)

_CROP_SIZE = 112
_MFCC_PER_FRAME = 4  # 100 fps audio MFCC / 25 fps video


class ActiveSpeakerService(IActiveSpeakerService):
    """Active speaker detection via the deeptalk ONNX re-export of LR-ASD."""

    def __init__(
        self,
        audio_model_path: str | None = None,
        visual_model_path: str | None = None,
        av_model_path: str | None = None,
        providers: list[str] | None = None,
        num_threads: int | None = None,
    ):
        import python_speech_features  # local import: optional dep, fail-soft upstream

        self._mfcc = python_speech_features.mfcc
        self.audio_path = audio_model_path or settings.asd_audio_model_path
        self.visual_path = visual_model_path or settings.asd_visual_model_path
        self.av_path = av_model_path or settings.asd_av_model_path
        self._providers = providers or ["CPUExecutionProvider"]
        self._num_threads = num_threads or settings.asd_num_threads

        # ASD runs in short bursts only at SPEECH_END (before LLM starts), so
        # claim threads to finish fast — single-thread was ~2× slower than
        # PyTorch on this hardware. 2 is the sweet spot; tune via ASD_NUM_THREADS.
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = self._num_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._audio = ort.InferenceSession(
            self.audio_path, sess_options=sess_opts, providers=self._providers,
        )
        self._visual = ort.InferenceSession(
            self.visual_path, sess_options=sess_opts, providers=self._providers,
        )
        self._av = ort.InferenceSession(
            self.av_path, sess_options=sess_opts, providers=self._providers,
        )
        logger.info(
            "ActiveSpeakerService initialized (ONNX): audio=%s visual=%s av=%s providers=%s threads=%d",
            self.audio_path, self.visual_path, self.av_path, self._providers, self._num_threads,
        )

    # --- preprocessing ---

    def _crops_to_array(self, face_crops: list[Image]) -> np.ndarray:
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

    def _score_sync(
        self, face_crops: list[Image], pcm16: bytes, sample_rate: int
    ) -> list[float]:
        if sample_rate != 16000:
            raise ValueError(f"ASD requires 16 kHz audio, got {sample_rate} Hz")
        if not pcm16 or not face_crops:
            return []

        video = self._crops_to_array(face_crops)
        if video.shape[0] == 0:
            return []

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        mfcc = self._mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

        n_v = min(video.shape[0], mfcc.shape[0] // _MFCC_PER_FRAME)
        if n_v < 1:
            return []
        audio_feat = mfcc[: n_v * _MFCC_PER_FRAME, :].astype(np.float32)[None, :, :]
        video_feat = video[:n_v, :, :][None, :, :, :]

        audio_embed = self._audio.run(None, {"audio_feature": audio_feat})[0]
        visual_embed = self._visual.run(None, {"visual_feature": video_feat})[0]
        logits = self._av.run(
            None, {"audio_embed": audio_embed, "visual_embed": visual_embed}
        )[0]  # shape (n_v,)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return [float(p) for p in probs]

    async def score(
        self, face_crops: list[Image], pcm16: bytes, sample_rate: int
    ) -> list[float]:
        return await asyncio.to_thread(
            self._score_sync, face_crops, pcm16, sample_rate
        )
