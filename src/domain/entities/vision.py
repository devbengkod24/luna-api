"""Vision domain entities."""
from dataclasses import dataclass, field
from typing import List, Optional
from time import time
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box coordinates for detected face/object."""
    
    x: int
    y: int
    width: int
    height: int
    
    def to_list(self) -> List[int]:
        """Convert to list format [x, y, w, h]."""
        return [self.x, self.y, self.width, self.height]
    
    def to_xyxy(self) -> List[int]:
        """Convert to [x1, y1, x2, y2] format."""
        return [self.x, self.y, self.x + self.width, self.y + self.height]
    
    def area(self) -> int:
        """Calculate bounding box area."""
        return self.width * self.height
    
    @classmethod
    def from_list(cls, box: List[int]) -> "BoundingBox":
        """Create BoundingBox from list [x, y, w, h]."""
        if len(box) != 4:
            raise ValueError("BoundingBox.from_list expects [x, y, w, h]")
        x, y, w, h = box
        if w < 0 or h < 0:
            raise ValueError("width/height must be non-negative")
        return cls(x=x, y=y, width=w, height=h)
    
    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BoundingBox":
        """Create BoundingBox from [x1, y1, x2, y2] format."""
        if x2 <= x1 or y2 <= y1:
            raise ValueError("x2 must be greater than x1, y2 must be greater than y1")
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


@dataclass
class Landmark:
    """Facial landmarks (5-point: left_eye, right_eye, nose, left_mouth, right_mouth)."""
    
    points: np.ndarray  # Shape: (5, 2)
    
    def to_list(self) -> List[List[float]]:
        """Convert to list format."""
        return self.points.tolist()
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Landmark":
        """Create Landmark from numpy array."""
        if arr.shape != (5, 2):
            raise ValueError(f"Landmark expects shape (5, 2), got {arr.shape}")
        return cls(points=arr.astype(np.float32))
    
    def transform_to_bbox(self, original_bbox: BoundingBox, target_bbox: BoundingBox) -> "Landmark":
        """Transform landmark coordinates from original bbox to target bbox.
        
        Used when tracked bbox differs from detection bbox.
        """
        # Calculate scale and offset
        scale_x = target_bbox.width / original_bbox.width
        scale_y = target_bbox.height / original_bbox.height
        offset_x = target_bbox.x - original_bbox.x * scale_x
        offset_y = target_bbox.y - original_bbox.y * scale_y
        
        # Transform points
        new_points = self.points.copy()
        new_points[:, 0] = self.points[:, 0] * scale_x + offset_x
        new_points[:, 1] = self.points[:, 1] * scale_y + offset_y
        
        return Landmark(points=new_points)


@dataclass
class FaceDetection:
    """Face detection result (without embedding)."""
    
    bbox: BoundingBox
    confidence: float
    landmark: Optional[Landmark] = None
    detection_index: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "landmark": self.landmark.to_list() if self.landmark else None,
            "detection_index": self.detection_index,
        }


@dataclass  
class TrackedFaceData:
    """Tracked face with associated detection data."""
    
    track_id: int
    bbox: BoundingBox  # Tracked bbox (stabilized by ByteTrack)
    confidence: float
    detection_index: Optional[int] = None
    landmark: Optional[Landmark] = None  # Landmark from detection, associated to this track
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "detection_index": self.detection_index,
        }


@dataclass
class Identity:
    """Recognized identity from database."""
    
    face_id: str
    name: str
    nim: str
    similarity_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "face_id": self.face_id,
            "name": self.name,
            "nim": self.nim,
            "similarity_score": self.similarity_score,
        }


@dataclass
class TrackState:
    """Per-track state: identity cache, recognition retry control, emotion cache."""

    track_id: int
    identity: Optional[Identity] = None
    recognition_attempts: int = 0
    last_recognition_time: float = 0.0
    last_recognition_quality: float = 0.0
    last_landmark: Optional[Landmark] = None
    last_embedding: Optional[List[float]] = None
    last_emotion: str = "neutral"
    last_emotion_confidence: float = 0.0
    last_emotion_frame: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    created_at: float = field(default_factory=time)

    def should_retry_recognition(
        self,
        current_quality: float,
        min_retry_interval: float = 2.0,
        max_attempts: int = 5,
        quality_threshold: float = 0.1,
    ) -> bool:
        if self.identity is not None or self.recognition_attempts >= max_attempts:
            return False
        if self.recognition_attempts == 0:
            return True
        time_passed = time() - self.last_recognition_time
        quality_improved = current_quality - self.last_recognition_quality >= quality_threshold
        return time_passed >= min_retry_interval or quality_improved

    def record_recognition_attempt(self, quality: float) -> None:
        self.recognition_attempts += 1
        self.last_recognition_time = time()
        self.last_recognition_quality = quality

    def bind_identity(self, identity: Identity) -> None:
        self.identity = identity

    def should_detect_emotion(self, current_frame: int, interval: int = 5) -> bool:
        return current_frame - self.last_emotion_frame >= interval

    def update_emotion(self, emotion: str, confidence: float, frame_id: int) -> None:
        self.last_emotion = emotion
        self.last_emotion_confidence = confidence
        self.last_emotion_frame = frame_id


@dataclass
class Confidence:
    """Confidence for a single face."""
    
    face: float = 0.0
    emotion: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "face": self.face,
            "emotion": self.emotion,
        }

@dataclass
class VisionResult:
    """Vision result for a single frame."""
    
    bbox: BoundingBox
    confidence: Confidence
    name: str = "Unknown"
    nim: Optional[str] = None
    emotion: str = "neutral"
    face_id: Optional[str] = None
    track_id: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "nim": self.nim,
            "emotion": self.emotion,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence.to_dict(),
            "face_id": self.face_id,
            "track_id": self.track_id,
        }
   
    
@dataclass
class DetectedFace:
    """Detected face with embedding for registration (legacy, used by faces use case)."""
    
    bbox: BoundingBox
    embedding: List[float]
    confidence: float = 0.0
    crop_bytes: Optional[bytes] = None
    detection_index: Optional[int] = None
    landmark: Optional[Landmark] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without embedding)."""
        return {
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "detection_index": self.detection_index,
        }


@dataclass
class DetectedEmotion:
    """Detected emotion with confidence."""
    
    emotion: str
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without embedding)."""
        return {
            "emotion": self.emotion,
            "confidence": self.confidence,
        }
