"""Vision port interfaces (split per ISP).

`IFaceRecognitionService` is a composite that implementations may satisfy,
but use cases should depend on the narrower port they actually need.
"""

from abc import ABC, abstractmethod

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
    async def decode(self, image_bytes: bytes) -> Image | None:
        ...


class IFaceDetector(ABC):
    """Detect faces in an image."""

    @abstractmethod
    async def detect_with_embedding(self, image_bytes: bytes) -> list[DetectedFace]:
        """Detect faces and extract embeddings (registration path)."""
        ...

    @abstractmethod
    async def detect_from_bytes(self, image_bytes: bytes) -> list[FaceDetection]:
        """Detect faces without embeddings from raw bytes."""
        ...

    @abstractmethod
    async def detect(self, image: Image) -> list[FaceDetection]:
        """Detect faces without embeddings from a decoded image."""
        ...


class IFaceAligner(ABC):
    """Align a face to a canonical pose using landmarks."""

    @abstractmethod
    async def align(
        self,
        image: Image,
        landmark: Landmark,
        output_size: tuple[int, int] = (112, 112),
    ) -> Image | None:
        ...


class IFaceEmbedder(ABC):
    """Extract an embedding from an aligned face."""

    @abstractmethod
    async def embed(self, aligned_face: Image) -> list[float] | None:
        ...


class IFaceCropper(ABC):
    """Crop a region of an image around a bbox."""

    @abstractmethod
    async def crop(
        self, image: Image, bbox: BoundingBox, padding: float = 0.1
    ) -> Image | None:
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
        self, detections: list[FaceDetection], frame_id: int
    ) -> tuple[list[TrackedFaceData], list[int]]:
        ...

    @abstractmethod
    async def reset(self) -> None:
        ...

    @abstractmethod
    async def get_active_track_ids(self) -> list[int]:
        ...


class IActiveSpeakerService(ABC):
    """Audio-visual active-speaker detector (LR-ASD family).

    Given a temporally-ordered sequence of face crops for one track and the
    corresponding audio segment, returns a per-frame "speaking" score (0..1)
    aligned to `face_crops`. High score = this visible face produced this audio
    (lip motion correlates with the audio) — the core device-directed signal.
    """

    @abstractmethod
    async def score(
        self,
        face_crops: list[Image],
        pcm16: bytes,
        sample_rate: int,
        fps: float | None = None,
    ) -> list[float]:
        """`fps` is the real delivered video fps for `face_crops`; the adapter
        scales its MFCC window to it so lips and audio stay aligned. None falls
        back to a nominal fps."""
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
