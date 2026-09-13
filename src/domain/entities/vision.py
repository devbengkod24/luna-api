"""Vision domain entities.

What is left here is the *enrolment* vocabulary: a box, five landmarks, a
detection, a face with an embedding. Tracking, identity caching and per-frame
results moved to luna-edge along with the loops that produced them — the cloud
never sees a frame, so it has nothing to track across.
"""
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Bounding box coordinates for detected face/object."""

    x: int
    y: int
    width: int
    height: int

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]

    def to_xyxy(self) -> list[int]:
        return [self.x, self.y, self.x + self.width, self.y + self.height]

    def area(self) -> int:
        return self.width * self.height

    @classmethod
    def from_list(cls, box: list[int]) -> "BoundingBox":
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

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) != 5:
            raise ValueError(f"Landmark expects 5 points, got {len(self.points)}")

    def to_list(self) -> list[list[float]]:
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
    landmark: Landmark | None = None
    detection_index: int = 0


@dataclass
class DetectedFace:
    """Detected face with embedding for registration."""

    bbox: BoundingBox
    embedding: list[float]
    confidence: float = 0.0
    crop_bytes: bytes | None = None
    detection_index: int | None = None
    landmark: Landmark | None = None
