"""HuggingFace ViT emotion detection adapter."""

import asyncio
import io
import logging

import torch
from PIL import Image as PILImage

from src.config import settings
from src.domain.entities.vision import DetectedEmotion
from src.domain.image import Image
from src.ports.vision import IEmotionDetectionService

logger = logging.getLogger(__name__)


class EmotionDetectionService(IEmotionDetectionService):
    """Emotion detection using HuggingFace ViT image classification."""

    def __init__(self, model_name: str = None, device: str = None):
        model_name = model_name or settings.emotion_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

        logger.info(
            "EmotionDetection initialized: model=%s device=%s",
            model_name, self.device,
        )

    def _predict(self, image: PILImage.Image) -> DetectedEmotion:
        try:
            inputs = {
                k: v.to(self.device)
                for k, v in self.processor(images=image, return_tensors="pt").items()
            }
            with torch.no_grad():
                probs = torch.nn.functional.softmax(self.model(**inputs).logits, dim=-1)[0]
            cls = int(torch.argmax(probs).item())
            return DetectedEmotion(
                emotion=self.id2label[cls], confidence=float(probs[cls].item())
            )
        except Exception as e:
            logger.warning("EmotionDetection failed: %s", e)
            return DetectedEmotion(emotion="neutral", confidence=0.0)

    def _detect_emotion_sync(self, face_crop_bytes: bytes) -> DetectedEmotion:
        try:
            return self._predict(PILImage.open(io.BytesIO(face_crop_bytes)).convert("RGB"))
        except Exception as e:
            logger.warning("detect_emotion (bytes) failed: %s", e)
            return DetectedEmotion(emotion="neutral", confidence=0.0)

    def _detect_emotion_from_array_sync(self, face_image) -> DetectedEmotion:
        try:
            return self._predict(PILImage.fromarray(face_image[:, :, ::-1]))  # BGR → RGB
        except Exception as e:
            logger.warning("detect_emotion (array) failed: %s", e)
            return DetectedEmotion(emotion="neutral", confidence=0.0)

    async def detect_emotion(self, face_crop_bytes: bytes) -> DetectedEmotion:
        return await asyncio.to_thread(self._detect_emotion_sync, face_crop_bytes)

    async def detect_emotion_from_image(self, face_image: Image) -> DetectedEmotion:
        return await asyncio.to_thread(self._detect_emotion_from_array_sync, face_image.raw)
