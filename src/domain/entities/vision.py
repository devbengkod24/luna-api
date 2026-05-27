"""Vision domain entities."""
from dataclasses import dataclass, field
from time import time
from typing import List, Optional, Tuple


@dataclass
class BoundingBox:
    """Bounding box coordinates for detected face/object."""

    x: int
    y: int
    width: int
    height: int

    def to_list(self) -> List[int]:
        return [self.x, self.y, self.width, self.height]

    def to_xyxy(self) -> List[int]:
        return [self.x, self.y, self.x + self.width, self.y + self.height]

    def area(self) -> int:
        return self.width * self.height

    @classmethod
    def from_list(cls, box: List[int]) -> "BoundingBox":
        if len(box) != 4:
            raise ValueError("BoundingBox.from_list expects [x, y, w, h]")
        x, y, w, h = box
        if w < 0 or h < 0:
            raise ValueError("width/height must be non-negative")
        return cls(x=x, y=y, width=w, height=h)

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BoundingBox":
        if x2 <= x1 or y2 <= y1:
            raise ValueError("x2 must be greater than x1, y2 must be greater than y1")
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


@dataclass(frozen=True)
class Landmark:
    """Facial landmarks (5-point: left_eye, right_eye, nose, left_mouth, right_mouth).

    Stored as a tuple of (x, y) pairs to keep numpy out of the domain.
    """

    points: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) != 5:
            raise ValueError(f"Landmark expects 5 points, got {len(self.points)}")

    def to_list(self) -> List[List[float]]:
        return [[x, y] for (x, y) in self.points]

    @classmethod
    def from_xy_pairs(cls, pairs) -> "Landmark":
        return cls(points=tuple((float(x), float(y)) for (x, y) in pairs))

    def transform_to_bbox(
        self, original_bbox: BoundingBox, target_bbox: BoundingBox
    ) -> "Landmark":
        """Transform landmark coordinates from original bbox to target bbox."""
        scale_x = target_bbox.width / original_bbox.width
        scale_y = target_bbox.height / original_bbox.height
        offset_x = target_bbox.x - original_bbox.x * scale_x
        offset_y = target_bbox.y - original_bbox.y * scale_y

        new_points = tuple(
            (x * scale_x + offset_x, y * scale_y + offset_y) for (x, y) in self.points
        )
        return Landmark(points=new_points)


@dataclass
class FaceDetection:
    """Face detection result (without embedding)."""

    bbox: BoundingBox
    confidence: float
    landmark: Optional[Landmark] = None
    detection_index: int = 0


@dataclass
class TrackedFaceData:
    """Tracked face with associated detection data."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    detection_index: Optional[int] = None
    landmark: Optional[Landmark] = None


@dataclass
class Identity:
    """Recognized identity from database."""

    face_id: str
    name: str
    nim: Optional[str] = None
    similarity_score: float = 0.0


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
        quality_improved = (
            current_quality - self.last_recognition_quality >= quality_threshold
        )
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


@dataclass
class DetectedFace:
    """Detected face with embedding for registration."""

    bbox: BoundingBox
    embedding: List[float]
    confidence: float = 0.0
    crop_bytes: Optional[bytes] = None
    detection_index: Optional[int] = None
    landmark: Optional[Landmark] = None


@dataclass
class DetectedEmotion:
    """Detected emotion with confidence."""

    emotion: str
    confidence: float = 0.0
