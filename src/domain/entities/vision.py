"""Vision domain entities."""
from dataclasses import dataclass, field
from typing import List, Optional
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
    
    @classmethod
    def from_list(cls, box: List[int]) -> "BoundingBox":
        """Create BoundingBox from list [x, y, w, h]."""
        return cls(x=box[0], y=box[1], width=box[2], height=box[3])


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
class FaceRecognitionResult:
    """Face recognition result for a single detected face."""
    
    name: str
    nim: Optional[str]
    box: BoundingBox
    confidence: float
    face_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "nim": self.nim,
            "box": self.box.to_list(),
            "confidence": self.confidence,
            "face_id": self.face_id,
        }


@dataclass
class EmotionRecognitionResult:
    """Emotion recognition result for a single detected face."""
    
    emotion: str
    box: BoundingBox
    confidence: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "emotion": self.emotion,
            "box": self.box.to_list(),
            "confidence": self.confidence,
        }


@dataclass
class VisionResult:
    """Vision analysis result - contains multiple face and emotion detections."""
    
    faces: List[FaceRecognitionResult]
    emotions: List[EmotionRecognitionResult]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "faces": [face.to_dict() for face in self.faces],
            "emotions": [emotion.to_dict() for emotion in self.emotions],
        }


@dataclass
class FaceRegistrationRequest:
    """Request for face registration."""
    
    name: str
    nim: str
    image_base64: str
    
    
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
class DetectedFace:
    """Detected face with embedding for registration."""
    
    box: BoundingBox
    embedding: List[float]
    confidence: float = 0.0  # Detection confidence score
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without embedding)."""
        return {
            "box": self.box.to_list(),
            "confidence": self.confidence,
        }

