"""Face use cases — register, list, get, delete."""

import base64
import logging

from src.domain.entities.face import FaceData, FaceRegistrationResult, RegisterFaceInput
from src.ports.identifiers import IUuidGenerator
from src.ports.services import IFaceService
from src.ports.vision import IFaceDetector

logger = logging.getLogger(__name__)


class RegisterFacesUseCase:
    """Register one or multiple faces: detect embedding then store in Qdrant."""

    def __init__(
        self,
        face_detector: IFaceDetector,
        face_service: IFaceService,
        uuid_generator: IUuidGenerator,
    ):
        self.face_detector = face_detector
        self.face_service = face_service
        self.uuid_generator = uuid_generator

    async def execute(
        self,
        faces: list[RegisterFaceInput],
        skip_existing: bool = True,
    ) -> list[FaceRegistrationResult]:
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
            detected = await self.face_detector.detect_with_embedding(image_bytes)
        except Exception as e:
            logger.error("Face detection error: %s", e, exc_info=True)
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

        existing: FaceData | None = None
        if nim:
            try:
                existing = await self.face_service.get_by_nim(nim)
            except Exception as e:
                logger.error("Qdrant query error: %s", e, exc_info=True)
                return FaceRegistrationResult(success=False, name=name, nim=nim, message="Storage query failed")

        if existing and skip_existing:
            return FaceRegistrationResult(
                success=False, face_id=existing.id, name=existing.name, nim=existing.nim,
                message=f"Face with NIM {nim} already registered."
            )
        face_id = existing.id if existing else self.uuid_generator.new_id()
        face = FaceData(id=face_id, name=name, nim=nim, embedding=embedding)
        try:
            await self.face_service.add_face(face)
        except Exception as e:
            logger.error("Failed to store face: %s", e, exc_info=True)
            return FaceRegistrationResult(success=False, name=name, nim=nim, message="Failed to store face")

        action = "updated" if existing else "registered"
        logger.info("Face %s (id=%s)", action, face_id)
        return FaceRegistrationResult(success=True, face_id=face_id, name=name, nim=nim, message=f"Face {action} successfully")


class ListFacesUseCase:
    """List registered faces with cursor-based pagination."""

    def __init__(self, face_service: IFaceService):
        self.face_service = face_service

    async def execute(
        self,
        limit: int = 20,
        cursor: str | None = None,
    ) -> tuple[list[FaceData], str | None]:
        return await self.face_service.get_all_cursor(limit=limit, cursor=cursor)


class GetFaceUseCase:
    """Get a single face by ID."""

    def __init__(self, face_service: IFaceService):
        self.face_service = face_service

    async def execute(self, face_id: str) -> FaceData | None:
        return await self.face_service.get_by_id(face_id)


class DeleteFaceUseCase:
    """Delete a face by ID."""

    def __init__(self, face_service: IFaceService):
        self.face_service = face_service

    async def execute(self, face_id: str) -> bool:
        return await self.face_service.delete(face_id)


class MatchFaceUseCase:
    """Match a pre-computed embedding against the registered faces.

    Distinct from RegisterFacesUseCase's detect-then-embed path: the vector
    arrives already computed, because the detector and the embedder now run on
    the kiosk. Nothing here decodes an image — no picture of anyone reaches this
    service on this route.
    """

    def __init__(self, face_service: IFaceService):
        self.face_service = face_service

    async def execute(
        self, embedding: list[float], threshold: float = 0.6
    ) -> tuple[FaceData, float] | None:
        if not embedding:
            return None
        return await self.face_service.search_face(embedding=embedding, threshold=threshold)
