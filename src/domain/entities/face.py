"""Face domain entities — used by face registration and CRUD."""

from dataclasses import dataclass, field
from typing import List, Optional
import uuid


@dataclass
class FaceData:
    """Registered face stored in Qdrant."""
    id: str
    name: str
    nim: str
    embedding: List[float] = field(default_factory=list)

    @classmethod
    def create(cls, name: str, nim: str, embedding: List[float]) -> "FaceData":
        return cls(id=str(uuid.uuid4()), name=name, nim=nim, embedding=embedding)

    def to_payload(self) -> dict:
        """Qdrant payload (without embedding)."""
        return {"name": self.name, "nim": self.nim}


@dataclass
class RegisterFaceInput:
    """Input data for a single face registration."""
    name: str
    nim: str
    image_base64: str


@dataclass
class FaceRegistrationResult:
    """Result of a single face registration attempt."""
    success: bool
    face_id: Optional[str] = None
    name: Optional[str] = None
    nim: Optional[str] = None
    message: Optional[str] = None
