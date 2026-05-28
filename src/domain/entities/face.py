"""Face domain entities — used by face registration and CRUD."""

from dataclasses import dataclass, field


@dataclass
class FaceData:
    """Registered face stored in Qdrant."""

    id: str
    name: str
    nim: str | None = None
    embedding: list[float] = field(default_factory=list)


@dataclass
class RegisterFaceInput:
    """Input data for a single face registration."""

    name: str
    image_base64: str
    nim: str | None = None


@dataclass
class FaceRegistrationResult:
    """Result of a single face registration attempt."""

    success: bool
    face_id: str | None = None
    name: str | None = None
    nim: str | None = None
    message: str | None = None
