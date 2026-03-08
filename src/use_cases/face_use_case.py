"""Face use cases — register, list, get, delete."""

import base64
import logging
from typing import List, Optional, Tuple

from src.domain.entities.face import FaceData, FaceRegistrationResult, RegisterFaceInput
from src.ports.services import IFaceService
from src.ports.vision import IFaceRecognitionService

logger = logging.getLogger(__name__)


class RegisterFacesUseCase:
    """Register one or multiple faces: detect embedding then store in Qdrant."""

    def __init__(self, recognition: IFaceRecognitionService, face_service: IFaceService):
        self.recognition = recognition
        self.face_service = face_service

    async def execute(
        self,
        faces: List[RegisterFaceInput],
        skip_existing: bool = True,
    ) -> List[FaceRegistrationResult]:
        results = []
        for face_input in faces:
            result = await self._register_single(face_input, skip_existing)
            results.append(result)
        success = sum(1 for r in results if r.success)
        logger.info(f"Registration complete: {success}/{len(results)} successful")
        return results

    async def _register_single(
        self, face_input: RegisterFaceInput, skip_existing: bool
    ) -> FaceRegistrationResult:
        name, nim = face_input.name, face_input.nim
        try:
            image_bytes = base64.b64decode(face_input.image_base64, validate=True)
        except Exception:
            return FaceRegistrationResult(success=False, name=name, nim=nim, message="Invalid base64 image")
        try:
            detected = self.recognition.detect_faces(image_bytes)
        except Exception as e:
            logger.error(f"Face detection error for {nim}: {e}", exc_info=True)
            return FaceRegistrationResult(success=False, name=name, nim=nim, message="Face detection failed")

        if not detected:
            return FaceRegistrationResult(success=False, name=name, nim=nim, message="No face detected in image")
        if len(detected) > 1:
            return FaceRegistrationResult(
                success=False, name=name, nim=nim,
                message=f"Multiple faces detected ({len(detected)}). Use a single-face image."
            )

        embedding = detected[0].embedding
        if not embedding:
            return FaceRegistrationResult(success=False, name=name, nim=nim, message="Failed to extract embedding")
        try:
            existing = await self.face_service.get_by_nim(nim)
        except Exception as e:
            logger.error(f"Qdrant query error for {nim}: {e}", exc_info=True)
            return FaceRegistrationResult(success=False, name=name, nim=nim, message="Storage query failed")

        if existing and skip_existing:
            return FaceRegistrationResult(
                success=False, face_id=existing.id, name=existing.name, nim=existing.nim,
                message=f"Face with NIM {nim} already registered."
            )
        face_id = existing.id if existing else str(__import__("uuid").uuid4())
        face = FaceData(id=face_id, name=name, nim=nim, embedding=embedding)
        try:
            await self.face_service.add_face(face)
        except Exception as e:
            logger.error(f"Failed to store face for {nim}: {e}", exc_info=True)
            return FaceRegistrationResult(success=False, name=name, nim=nim, message="Failed to store face")

        action = "updated" if existing else "registered"
        logger.info(f"Face {action}: {name} ({nim})")
        return FaceRegistrationResult(success=True, face_id=face_id, name=name, nim=nim, message=f"Face {action} successfully")


class ListFacesUseCase:
    """List registered faces with cursor-based pagination."""

    def __init__(self, face_service: IFaceService):
        self.face_service = face_service

    async def execute(
        self,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> Tuple[List[FaceData], Optional[str]]:
        return await self.face_service.get_all_cursor(limit=limit, cursor=cursor)


class GetFaceUseCase:
    """Get a single face by ID."""

    def __init__(self, face_service: IFaceService):
        self.face_service = face_service

    async def execute(self, face_id: str) -> Optional[FaceData]:
        return await self.face_service.get_by_id(face_id)


class DeleteFaceUseCase:
    """Delete a face by ID."""

    def __init__(self, face_service: IFaceService):
        self.face_service = face_service

    async def execute(self, face_id: str) -> bool:
        return await self.face_service.delete(face_id)
