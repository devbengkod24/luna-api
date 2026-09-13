"""Vision port interfaces (split per ISP).

`IFaceRecognitionService` is a composite that implementations may satisfy,
but use cases should depend on the narrower port they actually need.
"""

from abc import ABC, abstractmethod

from src.domain.entities.vision import (
    BoundingBox,
    DetectedFace,
    FaceDetection,
    Landmark,
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
