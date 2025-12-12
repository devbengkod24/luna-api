from pydantic import BaseModel
from typing import List, Optional

class RegisterFaceItem(BaseModel):
    """Single face item for registration."""
    name: str
    nim: str
    image_base64: str

class RegisterFaceRequest(BaseModel):
    """Request model for face registration (supports multiple faces)."""
    faces: List[RegisterFaceItem]
    skip_existing: bool = True

class FaceResponse(BaseModel):
    """Response model for a registered face."""
    id: str
    name: str
    nim: str

class SingleRegisterFaceResult(BaseModel):
    """Result for a single face registration attempt."""
    success: bool
    face_id: Optional[str] = None
    name: Optional[str] = None
    nim: Optional[str] = None
    message: Optional[str] = None

class RegisterFacesResponse(BaseModel):
    """Response model for face registration (one or many faces)."""
    total: int
    success_count: int
    failed_count: int
    results: List[SingleRegisterFaceResult]
