"""Vision domain entities."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import uuid as uuid_lib


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
class FaceData:
    """Registered face data stored in Qdrant."""
    
    id: str
    name: str
    nim: str
    embedding: List[float] = field(default_factory=list)
    
    @classmethod
    def create(cls, name: str, nim: str, embedding: List[float]) -> "FaceData":
        """Create a new FaceData with generated UUID."""
        return cls(
            id=str(uuid_lib.uuid4()),
            name=name,
            nim=nim,
            embedding=embedding,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without embedding for API response)."""
        return {
            "id": self.id,
            "name": self.name,
            "nim": self.nim,
        }
    
    def to_payload(self) -> dict:
        """Convert to Qdrant payload format."""
        return {
            "name": self.name,
            "nim": self.nim,
        }


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
class FaceRegistrationResult:
    """Result of face registration."""
    
    success: bool
    face_id: Optional[str] = None
    name: Optional[str] = None
    nim: Optional[str] = None
    message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "face_id": self.face_id,
            "name": self.name,
            "nim": self.nim,
            "message": self.message,
        }


@dataclass
class RegisterFaceInput:
    """Input data for face registration."""
    
    name: str
    nim: str
    image_base64: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without image for logging)."""
        return {
            "name": self.name,
            "nim": self.nim,
            "image_base64": f"{self.image_base64[:20]}..." if len(self.image_base64) > 20 else self.image_base64,
        }


@dataclass
class DetectedFace:
    """Detected face with embedding for registration."""
    
    bbox: BoundingBox
    embedding: List[float]
    confidence: float = 0.0
    crop_bytes: Optional[bytes] = None
    detection_index: Optional[int] = None
    
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

@dataclass
class TrackedFace:
    """Tracked face with bounding box and track ID."""
    
    track_id: int
    bbox: BoundingBox
    confidence: float = 0.0
    detection_index: Optional[int] = None
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "detection_index": self.detection_index,
        }