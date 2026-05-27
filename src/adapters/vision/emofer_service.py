"""EmoferService: emotion detection using VAD multi-task ONNX + optional Emotic fusion."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import settings
from src.domain.entities.vision import BoundingBox, DetectedEmotion
from src.domain.image import Image
from src.ports.vision import IEmotionDetectionService

logger = logging.getLogger(__name__)

# Bi-directional filter: suppress Emotic labels that contradict the face emotion
_SUPPRESS_RULES: dict[str, set[str]] = {
    "happy":   {"Sadness", "Suffering", "Pain", "Disquietment", "Shame", "Humiliation", "Aversion", "Disconnect"},
    "sad":     {"Happiness", "Pleasure", "Excitement", "Peace", "Affection", "Esteem", "Confidence", "Sensory Pleasure"},
    "angry":   {"Peace", "Affection", "Happiness", "Pleasure", "Sympathy"},
    "fear":    {"Confidence", "Peace", "Happiness", "Esteem"},
    "disgust": {"Affection", "Happiness", "Pleasure", "Peace"},
    "neutral": {"Excitement", "Pain", "Suffering", "Anger", "Fear", "Humiliation", "Pleasure", "Aversion", "Shame", "Sadness", "Disconnect"},
}


class EmoferService(IEmotionDetectionService):
    """Emotion detection using emofer ONNX models.

    - Primary: VADModel (face crop → 7-class emotion + VAD).
    - Optional: EmoticModel (full frame + body crop → 26 Emotic labels) when context is provided.
    """

    def __init__(
        self,
        vad_model_path: Optional[str] = None,
        emotic_model_path: Optional[str] = None,
    ):
        from src.adapters.vision._emofer.vad_model import VADModel
        from src.adapters.vision._emofer.emotic_model import EmoticModel

        vad_path = vad_model_path or settings.emofer_vad_model_path
        self._vad = VADModel(vad_path)
        logger.info("EmoferService: VADModel loaded from %s", vad_path)

        self._emotic: Optional[EmoticModel] = None
        emotic_path = emotic_model_path or settings.emofer_emotic_model_path
        if emotic_path and Path(emotic_path).exists():
            try:
                self._emotic = EmoticModel(emotic_path)
                logger.info("EmoferService: EmoticModel loaded from %s", emotic_path)
            except Exception as e:
                logger.warning(
                    "EmoferService: EmoticModel failed to load (running without it): %s", e
                )
        else:
            logger.info(
                "EmoferService: EmoticModel not found at %s, using VAD-only mode",
                emotic_path,
            )

    # --- Sync helpers ---

    def _detect_emotion_sync(self, face_crop_bytes: bytes) -> DetectedEmotion:
        try:
            arr = np.frombuffer(face_crop_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return DetectedEmotion(emotion="neutral", confidence=0.0)
            return self._run_vad(bgr)
        except Exception as e:
            logger.warning("EmoferService.detect_emotion failed: %s", e)
            return DetectedEmotion(emotion="neutral", confidence=0.0)

    def _detect_emotion_from_array_sync(self, face_image: np.ndarray) -> DetectedEmotion:
        try:
            return self._run_vad(face_image)
        except Exception as e:
            logger.warning("EmoferService.detect_emotion_from_array failed: %s", e)
            return DetectedEmotion(emotion="neutral", confidence=0.0)

    def _detect_emotion_with_context_sync(
        self,
        face_image: np.ndarray,
        full_frame: np.ndarray,
        bbox: BoundingBox,
    ) -> DetectedEmotion:
        """VAD + Emotic fusion with bi-directional filtering."""
        try:
            vad = self._vad.predict(face_image)
            if vad is None:
                return DetectedEmotion(emotion="neutral", confidence=0.0)

            if self._emotic is None:
                return DetectedEmotion(emotion=vad["emotion"], confidence=vad["confidence"])

            body_crop = self._body_crop(full_frame, bbox)
            if body_crop is None:
                return DetectedEmotion(emotion=vad["emotion"], confidence=vad["confidence"])

            emotic = self._emotic.predict(full_frame, body_crop, threshold=0.3)
            if emotic is None:
                return DetectedEmotion(emotion=vad["emotion"], confidence=vad["confidence"])

            self._log_fusion(vad, emotic)
            return DetectedEmotion(emotion=vad["emotion"], confidence=vad["confidence"])

        except Exception as e:
            logger.warning("EmoferService.detect_emotion_with_context failed: %s", e)
            return self._detect_emotion_from_array_sync(face_image)

    # --- Async port methods ---

    async def detect_emotion(self, face_crop_bytes: bytes) -> DetectedEmotion:
        return await asyncio.to_thread(self._detect_emotion_sync, face_crop_bytes)

    async def detect_emotion_from_image(self, face_image: Image) -> DetectedEmotion:
        return await asyncio.to_thread(self._detect_emotion_from_array_sync, face_image.raw)

    async def detect_emotion_with_context(
        self,
        face_image: Image,
        full_frame: Image,
        bbox: BoundingBox,
    ) -> DetectedEmotion:
        return await asyncio.to_thread(
            self._detect_emotion_with_context_sync,
            face_image.raw,
            full_frame.raw,
            bbox,
        )

    # --- Helpers ---

    def _run_vad(self, bgr_face: np.ndarray) -> DetectedEmotion:
        result = self._vad.predict(bgr_face)
        if result is None:
            return DetectedEmotion(emotion="neutral", confidence=0.0)
        return DetectedEmotion(emotion=result["emotion"], confidence=result["confidence"])

    def _body_crop(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """Approximate body region by extending face bbox 2x downward."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.to_xyxy()
        face_h = y2 - y1
        body_y2 = min(y2 + face_h * 2, h)
        crop = frame[y1:int(body_y2), max(0, x1):min(w, x2)]
        return crop if crop.size > 0 else None

    def _log_fusion(self, vad: dict, emotic: dict) -> None:
        fer_emotion = vad["emotion"]
        suppress = _SUPPRESS_RULES.get(fer_emotion, set())
        valence = vad["valence"]

        if fer_emotion == "neutral" and valence < -0.1:
            suppress = suppress - {"Sadness", "Shame", "Disconnect"}

        filtered = [e for e in emotic["emotions"] if e["label"] not in suppress]
        top_emotic = filtered[0]["label"] if filtered else "—"
        logger.debug(
            "Emofer fusion: fer=%s emotic_top=%s valence=%.2f arousal=%.2f",
            fer_emotion, top_emotic, valence, vad["arousal"],
        )
