"""Face registration REST routers."""
from typing import List, Optional
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Query
from src.use_cases.faces import (
    RegisterFacesUseCase,
    GetFacesUseCase,
    DeleteFaceUseCase
)
from src.domain.entities.vision import RegisterFaceInput
from src.drivers import dependencies
from src.drivers.rest.schemas.face import RegisterFaceRequest, RegisterFacesResponse, FaceResponse, SingleRegisterFaceResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/face", tags=["face"])

# Constants
MAX_FACES_PER_REQUEST = 100
MAX_BASE64_SIZE = 10 * 1024 * 1024  # 10MB
MIN_NAME_LENGTH = 1
MAX_NAME_LENGTH = 200
MIN_NIM_LENGTH = 1
MAX_NIM_LENGTH = 50

def validate_base64(data: str) -> bool:
    """Validate base64 string."""
    import base64
    try:
        if not data or not isinstance(data, str):
            return False
        if len(data) > MAX_BASE64_SIZE:
            return False
        base64.b64decode(data, validate=True)
        return True
    except Exception:
        return False

def validate_name(name: str) -> bool:
    """Validate name format."""
    if not name or not isinstance(name, str):
        return False
    name_stripped = name.strip()
    return MIN_NAME_LENGTH <= len(name_stripped) <= MAX_NAME_LENGTH


def validate_nim(nim: str) -> bool:
    """Validate NIM format."""
    if not nim or not isinstance(nim, str):
        return False
    nim_stripped = nim.strip()
    return MIN_NIM_LENGTH <= len(nim_stripped) <= MAX_NIM_LENGTH


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

    if not request.faces:
        raise HTTPException(
            status_code=400,
            detail="At least one face is required"
        )
    
    if len(request.faces) > MAX_FACES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FACES_PER_REQUEST} faces per request"
        )

    validation_errors = []
    for i, face in enumerate(request.faces):
        if not validate_name(face.name):
            validation_errors.append(
                f"Face {i+1}: Invalid name (must be {MIN_NAME_LENGTH}-{MAX_NAME_LENGTH} characters)"
            )
        
        if not validate_nim(face.nim):
            validation_errors.append(
                f"Face {i+1}: Invalid NIM (must be {MIN_NIM_LENGTH}-{MAX_NIM_LENGTH} characters)"
            )
        
        if not validate_base64(face.image_base64):
            validation_errors.append(
                f"Face {i+1}: Invalid base64 image or image too large (max {MAX_BASE64_SIZE / 1024 / 1024}MB)"
            )
    
    if validation_errors:
        raise HTTPException(
            status_code=400,
            detail="Validation errors: " + "; ".join(validation_errors)
        )

    nims = [face.nim.strip() for face in request.faces]
    if len(nims) != len(set(nims)):
        raise HTTPException(
            status_code=400,
            detail="Duplicate NIMs found in request. Each face must have a unique NIM."
        )


    try:
        items = [
            RegisterFaceInput(
                name=face.name.strip(),
                nim=face.nim.strip(),
                image_base64=face.image_base64,
            )
            for face in request.faces
        ]
        
        logger.info(f"Registering {len(items)} face(s)")
        
        # Execute use case
        results = await use_case.execute(
            faces=items,
            skip_existing=request.skip_existing,
        )
        
        # Calculate statistics
        success_count = sum(1 for r in results if r.success)
        failed_count = len(results) - success_count
        
        logger.info(
            f"Face registration completed: {success_count} successful, "
            f"{failed_count} failed out of {len(results)} total"
        )
        
        # Convert domain results to schema response
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


@router.get("/list", response_model=List[FaceResponse])
async def list_registered_faces(
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum number of faces to return"),
    offset: Optional[int] = Query(0, ge=0, description="Number of faces to skip"),
    use_case: GetFacesUseCase = Depends(dependencies.get_list_faces_use_case),
):
    """Get all registered faces.
    
    Returns a list of all faces registered in the system,
    including their IDs, names, and NIMs.
    """
    try:
        faces = await use_case.execute()
        
        # Apply pagination if requested
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
    
    except Exception as e:
        logger.error(f"Error listing faces: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/{face_id}", response_model=FaceResponse)
async def get_face_by_id(
    face_id: str,
    use_case: GetFacesUseCase = Depends(dependencies.get_list_faces_use_case),
):
    """Get a single face by ID.
    
    Returns face details including ID, name, and NIM.
    """
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
    """Delete a registered face by ID.
    
    Returns success status.
    """
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
