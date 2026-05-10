"""Vision port interfaces."""
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

import numpy as np

from src.domain.entities.vision import (
    DetectedFace,
    DetectedEmotion,
    FaceDetection,
    BoundingBox,
    Landmark,
    TrackedFaceData,
)


class IFaceRecognitionService(ABC):
    """Interface for face recognition adapter using InsightFace.
    
    Supports two modes:
    1. Detection + Recognition combined (legacy, for registration)
    2. Separate detection and recognition (for streaming pipeline)
    """
    
    @abstractmethod
    def detect_faces(self, image_bytes: bytes) -> List[DetectedFace]:
        """Detect faces and extract embeddings (for face registration)."""
        ...
    
    @abstractmethod
    def detect_faces_only(self, image_bytes: bytes) -> List[FaceDetection]:
        """Detect faces without embeddings (streaming pipeline)."""
        ...

    @abstractmethod
    def detect_faces_only_from_array(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces from BGR numpy array without embeddings."""
        ...

    @abstractmethod
    def align_face(
        self, image: np.ndarray, landmark: Landmark, output_size: Tuple[int, int] = (112, 112)
    ) -> Optional[np.ndarray]:
        """Align face using 5-point landmarks (ArcFace standard)."""
        ...

    @abstractmethod
    def extract_embedding(self, aligned_face: np.ndarray) -> Optional[List[float]]:
        """Extract 512-d normalized embedding from aligned face (112×112 BGR)."""
        ...

    @abstractmethod
    def crop_face(self, image: np.ndarray, bbox: BoundingBox, padding: float = 0.1) -> Optional[np.ndarray]:
        """Crop face region with padding."""
        ...

    @abstractmethod
    def decode_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Decode image bytes to BGR numpy array."""
        ...


class ITrackerService(ABC):
    """Interface for object/person tracking across frames."""
    
    @abstractmethod
    def track(
        self, 
        detections: List[FaceDetection], 
        frame_id: int
    ) -> Tuple[List["TrackedFaceData"], List[int]]:
        """Update tracker with new detections."""
        ...
    
    @abstractmethod
    def reset(self):
        """Reset tracker state (e.g., new video stream)."""
        ...
    
    @abstractmethod
    def get_active_track_ids(self) -> List[int]:
        """Get list of currently active track IDs."""
        ...

class IEmotionDetectionService(ABC):
    """Interface for emotion detection."""

    @abstractmethod
    def detect_emotion(
        self, face_crop_bytes: bytes,
    ) -> DetectedEmotion:
        """Detect emotion from face region (bytes input)."""
        ...

    @abstractmethod
    def detect_emotion_from_array(
        self, face_image: np.ndarray,
    ) -> DetectedEmotion:
        """Detect emotion from face region (numpy array input)."""
        ...

    def detect_emotion_with_context(
        self,
        face_image: np.ndarray,
        full_frame: np.ndarray,
        bbox: "BoundingBox",
    ) -> DetectedEmotion:
        """Detect emotion with optional scene context (full frame + bounding box).

        Default implementation ignores context and delegates to detect_emotion_from_array.
        Override in adapters that support contextual fusion (e.g. EmoferService).
        """
        return self.detect_emotion_from_array(face_image)
