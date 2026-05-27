"""Face domain entities — used by face registration and CRUD."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FaceData:
    """Registered face stored in Qdrant."""

    id: str
    name: str
    nim: Optional[str] = None
    embedding: List[float] = field(default_factory=list)


@dataclass
class RegisterFaceInput:
    """Input data for a single face registration."""

    name: str
    image_base64: str
    nim: Optional[str] = None


@dataclass
class FaceRegistrationResult:
    """Result of a single face registration attempt."""

    success: bool
    face_id: Optional[str] = None
    name: Optional[str] = None
    nim: Optional[str] = None
    message: Optional[str] = None
