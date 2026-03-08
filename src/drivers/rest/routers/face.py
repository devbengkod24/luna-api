"""Face CRUD REST endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Query

from src.domain.entities.face import RegisterFaceInput
from src.drivers import dependencies
from src.drivers.rest.schemas.face import (
    FaceResponse,
    ListFacesResponse,
    RegisterFaceRequest,
    RegisterFacesResponse,
    SingleRegisterFaceResult,
)
from src.use_cases.face_use_case import (
    DeleteFaceUseCase,
    GetFaceUseCase,
    ListFacesUseCase,
    RegisterFacesUseCase,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/face", tags=["face"])


@router.post("", response_model=RegisterFacesResponse)
async def register_faces(
    request: RegisterFaceRequest,
    use_case: RegisterFacesUseCase = Depends(dependencies.get_register_faces_use_case),
):
    """Register one or more faces. Each image must contain exactly one face."""
    items = [
        RegisterFaceInput(name=f.name, nim=f.nim, image_base64=f.image_base64)
        for f in request.faces
    ]
    results = await use_case.execute(faces=items, skip_existing=request.skip_existing)
    success_count = sum(1 for r in results if r.success)
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


@router.get("", response_model=ListFacesResponse)
async def list_faces(
    limit: int = Query(20, ge=1, le=1000),
    cursor: str | None = Query(None, description="Cursor for next page (from previous response next_cursor)"),
    use_case: ListFacesUseCase = Depends(dependencies.get_list_faces_use_case),
):
    """List registered faces with cursor-based pagination."""
    faces, next_cursor = await use_case.execute(limit=limit, cursor=cursor)
    return ListFacesResponse(
        items=[FaceResponse(id=f.id, name=f.name, nim=f.nim) for f in faces],
        next_cursor=next_cursor,
    )


@router.get("/{face_id}", response_model=FaceResponse)
async def get_face(
    face_id: str,
    use_case: GetFaceUseCase = Depends(dependencies.get_face_use_case),
):
    """Get a face by ID."""
    face = await use_case.execute(face_id)
    if not face:
        raise HTTPException(status_code=404, detail=f"Face '{face_id}' not found")
    return FaceResponse(id=face.id, name=face.name, nim=face.nim)


@router.delete("/{face_id}")
async def delete_face(
    face_id: str,
    use_case: DeleteFaceUseCase = Depends(dependencies.get_delete_face_use_case),
):
    """Delete a registered face by ID."""
    success = await use_case.execute(face_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Face '{face_id}' not found")
    return {"success": True, "message": f"Face '{face_id}' deleted"}
