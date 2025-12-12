"""Face registration REST routers."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException

from src.use_cases.face_registration import (
    RegisterFacesUseCase,
    GetRegisteredFacesUseCase,
    DeleteFaceUseCase,
)
from src.domain.entities.vision import RegisterFaceInput
from src.drivers import dependencies
from src.drivers.rest.schemas.face import RegisterFaceRequest, RegisterFacesResponse, FaceResponse, SingleRegisterFaceResult

router = APIRouter(prefix="/v1/face", tags=["face"])

@router.post("/register", response_model=RegisterFacesResponse)
async def register_faces(
    request: RegisterFaceRequest,
    use_case: RegisterFacesUseCase = Depends(dependencies.get_register_faces_use_case),
):
    """Register one or more faces.
    
    This endpoint accepts a list of faces, but you can also send just one item.
    Each item should contain a base64-encoded image with exactly one face.
    The skip_existing flag controls whether existing faces should be skipped (True) or updated (False).
    """
    # Convert schema to domain entities
    items = [
        RegisterFaceInput(
            name=face.name,
            nim=face.nim,
            image_base64=face.image_base64,
        )
        for face in request.faces
    ]
    
    # Execute use case
    results = await use_case.execute(
        faces=items,
        skip_existing=request.skip_existing,
    )
    
    # Calculate statistics
    success_count = sum(1 for r in results if r.success)
    
    # Convert domain results to schema response
    return RegisterFacesResponse(
        total=len(results),
        success_count=success_count,
        failed_count=len(results) - success_count,
        results=[
            SingleRegisterFaceResult(
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
    face_store = dependencies.face_store
    
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

