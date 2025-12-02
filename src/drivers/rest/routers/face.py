"""Face registration REST routers."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.use_cases.face_registration import (
    RegisterFaceUseCase,
    RegisterBulkFacesUseCase,
    GetRegisteredFacesUseCase,
    DeleteFaceUseCase,
    BulkFaceRegistrationItem,
)
from src.drivers.rest import dependencies

router = APIRouter(prefix="/v1/face", tags=["face"])


# Request/Response models
class RegisterFaceRequest(BaseModel):
    """Request model for single face registration."""
    name: str
    nim: str
    image_base64: str
    update_if_exists: bool = False


class BulkFaceItem(BaseModel):
    """Single item for bulk registration."""
    name: str
    nim: str
    image_base64: str


class BulkRegisterFaceRequest(BaseModel):
    """Request model for bulk face registration."""
    faces: List[BulkFaceItem]
    skip_existing: bool = True


class FaceResponse(BaseModel):
    """Response model for face data."""
    id: str
    name: str
    nim: str


class RegisterFaceResponse(BaseModel):
    """Response model for face registration."""
    success: bool
    face_id: Optional[str] = None
    name: Optional[str] = None
    nim: Optional[str] = None
    message: Optional[str] = None


class BulkRegisterFaceResponse(BaseModel):
    """Response model for bulk face registration."""
    total: int
    success_count: int
    failed_count: int
    results: List[RegisterFaceResponse]


# Routes
@router.post("/register", response_model=RegisterFaceResponse)
async def register_face(
    request: RegisterFaceRequest,
    use_case: RegisterFaceUseCase = Depends(dependencies.get_register_face_use_case),
):
    """Register a single face.
    
    Takes a base64-encoded image containing a face and registers it
    with the provided name and NIM.
    
    The image should contain exactly one face for best results.
    """
    result = await use_case.execute(
        name=request.name,
        nim=request.nim,
        image_base64=request.image_base64,
        update_if_exists=request.update_if_exists,
    )
    
    return RegisterFaceResponse(
        success=result.success,
        face_id=result.face_id,
        name=result.name,
        nim=result.nim,
        message=result.message,
    )


@router.post("/register/bulk", response_model=BulkRegisterFaceResponse)
async def register_faces_bulk(
    request: BulkRegisterFaceRequest,
    use_case: RegisterBulkFacesUseCase = Depends(dependencies.get_bulk_register_face_use_case),
):
    """Register multiple faces in bulk.
    
    Useful for initial setup/seeding of the face database.
    Each item in the request should contain a base64-encoded image
    with exactly one face.
    """
    items = [
        BulkFaceRegistrationItem(
            name=face.name,
            nim=face.nim,
            image_base64=face.image_base64,
        )
        for face in request.faces
    ]
    
    results = await use_case.execute(
        faces=items,
        skip_existing=request.skip_existing,
    )
    
    success_count = sum(1 for r in results if r.success)
    
    return BulkRegisterFaceResponse(
        total=len(results),
        success_count=success_count,
        failed_count=len(results) - success_count,
        results=[
            RegisterFaceResponse(
                success=r.success,
                face_id=r.face_id,
                name=r.name,
                nim=r.nim,
                message=r.message,
            )
            for r in results
        ],
    )


@router.get("/list", response_model=List[FaceResponse])
async def list_registered_faces(
    use_case: GetRegisteredFacesUseCase = Depends(dependencies.get_list_faces_use_case),
):
    """Get all registered faces.
    
    Returns a list of all faces registered in the system,
    including their IDs, names, and NIMs.
    """
    faces = await use_case.execute()
    
    return [
        FaceResponse(
            id=face.id,
            name=face.name,
            nim=face.nim,
        )
        for face in faces
    ]


@router.delete("/{face_id}")
async def delete_face(
    face_id: str,
    use_case: DeleteFaceUseCase = Depends(dependencies.get_delete_face_use_case),
):
    """Delete a registered face by ID.
    
    Removes the face from the recognition database.
    Future detections will not match this face anymore.
    """
    success = await use_case.execute(face_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Face with ID {face_id} not found or could not be deleted",
        )
    
    return {"message": f"Face {face_id} deleted successfully"}


@router.get("/search/{nim}", response_model=Optional[FaceResponse])
async def search_face_by_nim(
    nim: str,
):
    """Search for a registered face by NIM.
    
    Returns the face data if found, or 404 if not found.
    """
    face_store = dependencies.get_face_store()
    
    if not face_store:
        raise HTTPException(
            status_code=503,
            detail="Face recognition service not available",
        )
    
    face = await face_store.get_face_by_nim(nim)
    
    if not face:
        raise HTTPException(
            status_code=404,
            detail=f"Face with NIM {nim} not found",
        )
    
    return FaceResponse(
        id=face.id,
        name=face.name,
        nim=face.nim,
    )

