"""Vision domain entities."""
from dataclasses import dataclass
from typing import List, Optional


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
class FaceRecognitionResult:
    """Face recognition result for a single detected face."""
    
    name: str
    box: BoundingBox
    confidence: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "box": self.box.to_list(),
            "confidence": self.confidence,
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

