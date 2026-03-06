"""Vision adapters (face recognition, tracking, emotion, vector store)."""

from .face_recognition_service import FaceRecognitionService
from .tracker_service import TrackerAdapter
from .emotion_service import EmotionDetectionAdapter
from .face_store import FaceStore

__all__ = [
    "FaceRecognition",
    "TrackerAdapter",
    "EmotionDetectionAdapter",
    "FaceStore",
]

