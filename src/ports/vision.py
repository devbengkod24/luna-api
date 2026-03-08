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
    def detect_faces(self, image_bytes: bytes, include_crops: bool = False) -> List[DetectedFace]:
        """Detect all faces and extract embeddings (legacy method for registration).
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            include_crops: Whether to include cropped face images
            
        Returns:
            List[DetectedFace]: Detected faces with embeddings
        """
        ...
    
    @abstractmethod
    def detect_faces_only(self, image_bytes: bytes) -> List[FaceDetection]:
        """Detect faces WITHOUT extracting embeddings (for streaming pipeline).
        
        This is faster as it only runs SCRFD detection.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            
        Returns:
            List[FaceDetection]: Detected faces with bbox, confidence, landmark
        """
        ...
    
    @abstractmethod
    def align_face(
        self,
        image: np.ndarray,
        landmark: Landmark,
        output_size: Tuple[int, int] = (112, 112),
    ) -> Optional[np.ndarray]:
        """Align face using landmarks (for recognition).
        
        Uses affine transformation based on 5-point landmarks.
        
        Args:
            image: Full frame image (BGR numpy array)
            landmark: 5-point facial landmarks
            output_size: Output aligned face size (default 112x112 for ArcFace)
            
        Returns:
            Aligned face image or None if alignment fails
        """
        ...
    
    @abstractmethod
    def extract_embedding(self, aligned_face: np.ndarray) -> Optional[List[float]]:
        """Extract embedding from aligned face image.
        
        Args:
            aligned_face: Aligned face image (112x112 BGR)
            
        Returns:
            512-dimensional embedding vector or None if extraction fails
        """
        ...
    
    @abstractmethod
    def crop_face(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        padding: float = 0.1,
    ) -> Optional[np.ndarray]:
        """Crop face region from image with padding.
        
        Args:
            image: Full frame image (BGR numpy array)
            bbox: Face bounding box
            padding: Padding ratio (default 10%)
            
        Returns:
            Cropped face image or None if cropping fails
        """
        ...


class ITrackerService(ABC):
    """Interface for object/person tracking across frames."""
    
    @abstractmethod
    def track(
        self, 
        detections: List[FaceDetection], 
        frame_id: int
    ) -> Tuple[List["TrackedFaceData"], List[int]]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detected faces in current frame
            frame_id: Current frame number
            
        Returns:
            Tuple containing:
            - List[TrackedFaceData]: Active tracked faces with associated detection data
            - List[int]: Dead track IDs (tracks that have been lost)
        """
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
        """Detect emotion from face region (bytes input).
        
        Args:
            face_crop_bytes: Cropped face image bytes (JPEG/PNG)
            
        Returns:
            DetectedEmotion with emotion label and confidence
        """
        ...
    
    @abstractmethod
    def detect_emotion_from_array(
        self, face_image: np.ndarray,
    ) -> DetectedEmotion:
        """Detect emotion from face region (numpy array input).
        
        Args:
            face_image: Face image as BGR numpy array
            
        Returns:
            DetectedEmotion with emotion label and confidence
        """
        ...
