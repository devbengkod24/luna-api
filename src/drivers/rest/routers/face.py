"""Face registration REST routers."""
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Query

from src.use_cases.faces import (
    RegisterFacesUseCase,
    GetFacesUseCase,
    DeleteFaceUseCase,
)
from src.domain.entities.vision import RegisterFaceInput
from src.drivers import dependencies
from src.drivers.rest.schemas.face import (
    RegisterFaceRequest,
    RegisterFacesResponse,
    FaceResponse,
    SingleRegisterFaceResult,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/face", tags=["face"])

@router.post("/register", response_model=RegisterFacesResponse)
async def register_faces(
    request: RegisterFaceRequest,
    use_case: RegisterFacesUseCase = Depends(dependencies.get_register_faces_use_case),
):
    """Register one or more faces.
    
    Each item should contain a base64-encoded image with exactly one face.
    The skip_existing flag controls whether existing faces should be skipped (True) or updated (False).
    """
    try:
        items = [
            RegisterFaceInput(
                name=face.name,
                nim=face.nim,
                image_base64=face.image_base64,
            )
            for face in request.faces
        ]
        
        logger.info(f"Registering {len(items)} face(s)")
        
        results = await use_case.execute(
            faces=items,
            skip_existing=request.skip_existing,
        )
        
        success_count = sum(1 for r in results if r.success)
        failed_count = len(results) - success_count
        
        logger.info(
            f"Face registration completed: {success_count} successful, "
            f"{failed_count} failed out of {len(results)} total"
        )
        
        return RegisterFacesResponse(
            total=len(results),
            success_count=success_count,
            failed_count=failed_count,
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
    
    except Exception as e:
        logger.error(f"Error registering faces: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/list", response_model=list[FaceResponse])
async def list_registered_faces(
    limit: int | None = Query(None, ge=1, le=1000, description="Maximum number of faces to return"),
    offset: int | None = Query(0, ge=0, description="Number of faces to skip"),
    use_case: GetFacesUseCase = Depends(dependencies.get_list_faces_use_case),
):
    """Get all registered faces."""
    try:
        faces = await use_case.execute()
        
        if limit is not None:
            total = len(faces)
            faces = faces[offset:offset + limit]
            logger.info(f"Listed {len(faces)} face(s) (offset={offset}, limit={limit}, total={total})")
        else:
            logger.info(f"Listed {len(faces)} face(s)")
        
        return [
            FaceResponse(
                id=face.id,
                name=face.name,
                nim=face.nim,
            )
            for face in faces
        ]
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error listing faces: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/{face_id}", response_model=FaceResponse)
async def get_face_by_id(
    face_id: str,
    use_case: GetFacesUseCase = Depends(dependencies.get_list_faces_use_case),
):
    """Get a single face by ID."""
    try:
        faces = await use_case.execute()
        face = next((f for f in faces if f.id == face_id), None)
        
        if not face:
            raise HTTPException(
                status_code=404,
                detail=f"Face with ID '{face_id}' not found"
            )
        
        logger.info(f"Retrieved face: {face.name} ({face.nim})")
        
        return FaceResponse(
            id=face.id,
            name=face.name,
            nim=face.nim,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting face by ID: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete("/{face_id}")
async def delete_face(
    face_id: str,
    use_case: DeleteFaceUseCase = Depends(dependencies.get_delete_face_use_case),
):
    """Delete a registered face by ID."""
    try:
        success = await use_case.execute(face_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Face with ID '{face_id}' not found"
            )
        
        logger.info(f"Deleted face with ID: {face_id}")
        
        return {
            "success": True,
            "message": f"Face with ID '{face_id}' deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
