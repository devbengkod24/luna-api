"""Emotion detection via EmotiEffLib (AffectNet-trained EfficientNet, ONNX).

The only emotion adapter. It replaced a custom FER2013-era ONNX model that was
badly miscalibrated — it returned things like `fear` at 1.00 confidence on
ordinary ID photos.
"""

import asyncio
import logging

import cv2
import numpy as np

from src.config import settings
from src.domain.entities.vision import BoundingBox, DetectedEmotion
from src.domain.image import Image
from src.ports.vision import IEmotionDetectionService

logger = logging.getLogger(__name__)

# EmotiEffLib returns capitalised AffectNet labels. Map onto the lowercase
# vocabulary the rest of Luna speaks — the FE overlay, base.txt's MATCH USER
# ENERGY rules and the stored conversation context all key on it. Localisation
# happens in the frontend i18n layer (`translateEmotion`); the app is bilingual,
# so translating here would show Indonesian labels in English mode and stop the
# prompt's rules from matching.
_LABEL_MAP = {
    "Anger": "angry",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprise",
    # Only produced by the 8-class models; folded onto the closest 7-class label
    # so a model swap can't emit a token nothing downstream understands.
    "Contempt": "disgust",
}

# AffectNet models are trained on detector bounding boxes that include forehead,
# chin and some cheek — not on the tight 5-point aligned crop the recogniser
# uses. Feeding the aligned crop is a domain mismatch that costs accuracy
# silently, so this adapter re-crops from the full frame instead.
_BBOX_MARGIN = 0.25


class EmotiEffLibService(IEmotionDetectionService):
    """Emotion detection via EmotiEffLib's ONNX recognizer."""

    def __init__(self, model_name: str | None = None):
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

        name = model_name or settings.emotiefflib_model
        available = get_model_list()
        if name not in available:
            raise ValueError(
                f"Unknown EmotiEffLib model {name!r}. Available: {available}"
            )
        # ONNX only: the torch engine pulls timm==0.9.* which this image does not
        # carry, and onnxruntime is already the runtime used for ASD/InsightFace.
        self._recognizer = EmotiEffLibRecognizer(engine="onnx", model_name=name)
        logger.info(
            "EmotiEffLibService: model=%s engine=onnx classes=%d",
            name, len(self._recognizer.idx_to_emotion_class),
        )

    # --- Inference ---

    def _predict(self, bgr: np.ndarray) -> DetectedEmotion:
        if bgr is None or bgr.size == 0:
            return DetectedEmotion(emotion="neutral", confidence=0.0)
        # The library's ONNX _preprocess does no colour conversion — whatever it
        # is handed is what the model sees, and these are RGB-trained. Measured
        # on 10 of storage/faces: feeding BGR shifted class probabilities by a
        # median of 0.03 (max 0.16) without flipping the argmax on those easy
        # frontal shots. So it is a bias, not a catastrophe — but it is free to
        # get right, and the margin is what decides the ambiguous kiosk frames.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        labels, scores = self._recognizer.predict_emotions(rgb, logits=False)
        label = labels[0] if isinstance(labels, (list, tuple)) else labels
        row = np.asarray(scores)[0] if np.asarray(scores).ndim > 1 else np.asarray(scores)
        # Multi-task models append valence/arousal after the class scores.
        if self._recognizer.is_mtl:
            row = row[:-2]
        confidence = float(np.max(row)) if row.size else 0.0
        return DetectedEmotion(
            emotion=_LABEL_MAP.get(label, "neutral"),
            confidence=round(confidence, 3),
        )

    def _safe_predict(self, bgr: np.ndarray) -> DetectedEmotion:
        try:
            return self._predict(bgr)
        except Exception as e:
            logger.warning("EmotiEffLibService prediction failed: %s", e)
            return DetectedEmotion(emotion="neutral", confidence=0.0)

    def _decode(self, face_crop_bytes: bytes) -> DetectedEmotion:
        arr = np.frombuffer(face_crop_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return DetectedEmotion(emotion="neutral", confidence=0.0)
        return self._safe_predict(bgr)

    def _with_context(
        self, face_image: np.ndarray, full_frame: np.ndarray, bbox: BoundingBox
    ) -> DetectedEmotion:
        crop = self._margin_crop(full_frame, bbox)
        # `face_image` (the aligned crop) is the fallback, not the preference —
        # see _BBOX_MARGIN.
        return self._safe_predict(crop if crop is not None else face_image)

    # --- Async port methods ---

    async def detect_emotion(self, face_crop_bytes: bytes) -> DetectedEmotion:
        return await asyncio.to_thread(self._decode, face_crop_bytes)

    async def detect_emotion_from_image(self, face_image: Image) -> DetectedEmotion:
        return await asyncio.to_thread(self._safe_predict, face_image.raw)

    async def detect_emotion_with_context(
        self, face_image: Image, full_frame: Image, bbox: BoundingBox
    ) -> DetectedEmotion:
        return await asyncio.to_thread(
            self._with_context, face_image.raw, full_frame.raw, bbox
        )

    # --- Helpers ---

    @staticmethod
    def _margin_crop(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
        """Face box widened by _BBOX_MARGIN, clamped to the frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.to_xyxy()
        mx = int((x2 - x1) * _BBOX_MARGIN)
        my = int((y2 - y1) * _BBOX_MARGIN)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None
