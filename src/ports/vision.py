"""Vision port interfaces (split per ISP).

`IFaceRecognitionService` is a composite that implementations may satisfy,
but use cases should depend on the narrower port they actually need.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from src.domain.entities.vision import (
    BoundingBox,
    DetectedEmotion,
    DetectedFace,
    FaceDetection,
    Landmark,
    TrackedFaceData,
)
from src.domain.image import Image


class IImageDecoder(ABC):
    """Decode raw bytes into a domain Image."""

    @abstractmethod
    async def decode(self, image_bytes: bytes) -> Optional[Image]:
        ...


class IFaceDetector(ABC):
    """Detect faces in an image."""

    @abstractmethod
    async def detect_with_embedding(self, image_bytes: bytes) -> List[DetectedFace]:
        """Detect faces and extract embeddings (registration path)."""
        ...

    @abstractmethod
    async def detect_from_bytes(self, image_bytes: bytes) -> List[FaceDetection]:
        """Detect faces without embeddings from raw bytes."""
        ...

    @abstractmethod
    async def detect(self, image: Image) -> List[FaceDetection]:
        """Detect faces without embeddings from a decoded image."""
        ...


class IFaceAligner(ABC):
    """Align a face to a canonical pose using landmarks."""

    @abstractmethod
    async def align(
        self,
        image: Image,
        landmark: Landmark,
        output_size: Tuple[int, int] = (112, 112),
    ) -> Optional[Image]:
        ...


class IFaceEmbedder(ABC):
    """Extract an embedding from an aligned face."""

    @abstractmethod
    async def embed(self, aligned_face: Image) -> Optional[List[float]]:
        ...


class IFaceCropper(ABC):
    """Crop a region of an image around a bbox."""

    @abstractmethod
    async def crop(
        self, image: Image, bbox: BoundingBox, padding: float = 0.1
    ) -> Optional[Image]:
        ...


class IFaceRecognitionService(
    IImageDecoder, IFaceDetector, IFaceAligner, IFaceEmbedder, IFaceCropper
):
    """Composite that bundles every face-recognition capability.

    Adapters that ship as one library (e.g. InsightFace) implement this; use
    cases should depend on the smaller ports above where possible.
    """


class ITrackerService(ABC):
    """Interface for object/person tracking across frames."""

    @abstractmethod
    async def track(
        self, detections: List[FaceDetection], frame_id: int
    ) -> Tuple[List[TrackedFaceData], List[int]]:
        ...

    @abstractmethod
    async def reset(self) -> None:
        ...

    @abstractmethod
    async def get_active_track_ids(self) -> List[int]:
        ...


class IEmotionDetectionService(ABC):
    """Interface for emotion detection."""

    @abstractmethod
    async def detect_emotion(self, face_crop_bytes: bytes) -> DetectedEmotion:
        ...

    @abstractmethod
    async def detect_emotion_from_image(self, face_image: Image) -> DetectedEmotion:
        ...

    async def detect_emotion_with_context(
        self,
        face_image: Image,
        full_frame: Image,
        bbox: BoundingBox,
    ) -> DetectedEmotion:
        """Optional contextual fusion; default falls back to face-only detection."""
        return await self.detect_emotion_from_image(face_image)
