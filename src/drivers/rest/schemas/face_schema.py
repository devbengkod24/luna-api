import base64

from pydantic import BaseModel, Field, field_validator

from src.config import settings


class RegisterFaceItem(BaseModel):
    """Single face item for registration."""

    name: str = Field(..., min_length=1, max_length=100, description="Name")
    nim: str | None = Field(
        default=None, min_length=1, max_length=20, description="Student NIM (optional)"
    )
    image_base64: str = Field(
        ...,
        max_length=settings.face_max_image_size,
        description="Base64 encoded image of the face",
    )

    @field_validator("nim")
    @classmethod
    def validate_nim_format(cls, v: str | None) -> str | None:
        """Validate NIM format (optional)."""
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("image_base64")
    @classmethod
    def validate_base64_image(cls, v: str) -> str:
        """Validate base64 image format and size."""
        if not v:
            raise ValueError("Image data is required")

        if len(v) > settings.face_max_image_size:
            max_mb = settings.face_max_image_size / (1024 * 1024)
            raise ValueError(f"Image too large (max {max_mb}MB)")

        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 format: {str(e)}")

        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Strip whitespace from name."""
        if not v:
            raise ValueError("Name is required")
        return v.strip()


class RegisterFaceRequest(BaseModel):
    """Request model for face registration (supports multiple faces)."""

    faces: list[RegisterFaceItem] = Field(
        ...,
        min_length=1,
        max_length=settings.face_max_batch_size,
        description=f"List of faces to register (minimum 1, maximum {settings.face_max_batch_size})",
    )
    skip_existing: bool = Field(
        True, description="Skip faces with existing NIM if True, update if False"
    )


class FaceResponse(BaseModel):
    """Response model for a registered face."""

    id: str
    name: str
    nim: str | None = None


class ListFacesResponse(BaseModel):
    """Response model for paginated face list (cursor-based)."""

    items: list[FaceResponse]
    next_cursor: str | None = None


class SingleRegisterFaceResult(BaseModel):
    """Result for a single face registration attempt."""

    success: bool
    face_id: str | None = None
    name: str | None = None
    nim: str | None = None
    message: str | None = None


class RegisterFacesResponse(BaseModel):
    """Response model for face registration (one or many faces)."""

    total: int
    success_count: int
    failed_count: int
    results: list[SingleRegisterFaceResult]


class MatchFaceRequest(BaseModel):
    """An embedding computed on the kiosk, to be matched against the store.

    The edge computes the vector and the cloud owns the index: a new kiosk needs
    no sync step, and no biometric template is replicated onto a box standing in
    a public lobby. The embedding MUST come from the same ArcFace model faces
    were registered with — a version drift here misses silently, it does not
    error.
    """

    embedding: list[float] = Field(
        ..., min_length=1, max_length=1024, description="Face embedding vector"
    )
    threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum cosine score to accept"
    )


class MatchFaceResponse(BaseModel):
    """Never an error when nobody matches — an unknown visitor is a normal case."""

    matched: bool
    id: str | None = None
    name: str | None = None
    nim: str | None = None
    score: float | None = None
