"""Face registration use cases."""
import base64
from typing import List, Optional
from dataclasses import dataclass

from src.ports.vision import IFaceRecognitionAdapter, IFaceVectorStore
from src.domain.entities.vision import (
    FaceData,
    FaceRegistrationResult,
    DetectedFace,
)


@dataclass
class BulkFaceRegistrationItem:
    """Item for bulk face registration."""
    name: str
    nim: str
    image_base64: str


class RegisterFaceUseCase:
    """Use case for registering a single face."""
    
    def __init__(
        self,
        face_recognition: IFaceRecognitionAdapter,
        face_store: IFaceVectorStore,
    ):
        """Initialize use case.
        
        Args:
            face_recognition: Face recognition adapter
            face_store: Face vector store
        """
        self.face_recognition = face_recognition
        self.face_store = face_store
    
    async def execute(
        self,
        name: str,
        nim: str,
        image_base64: str,
        update_if_exists: bool = False,
    ) -> FaceRegistrationResult:
        """Register a face from an image.
        
        Args:
            name: Person's name
            nim: Student NIM
            image_base64: Base64-encoded image
            update_if_exists: Update existing face if NIM already registered
            
        Returns:
            FaceRegistrationResult: Registration result
        """
        # Check if NIM already exists
        existing = await self.face_store.get_face_by_nim(nim)
        if existing and not update_if_exists:
            return FaceRegistrationResult(
                success=False,
                message=f"NIM {nim} sudah terdaftar dengan nama {existing.name}",
            )
        
        # Decode image
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            return FaceRegistrationResult(
                success=False,
                message=f"Gagal decode gambar: {str(e)}",
            )
        
        # Get face embedding
        embedding = self.face_recognition.get_embedding(image_bytes)
        
        if not embedding:
            return FaceRegistrationResult(
                success=False,
                message="Tidak ada wajah terdeteksi dalam gambar",
            )
        
        # Create or update face data
        if existing and update_if_exists:
            # Update existing face
            face_data = FaceData(
                id=existing.id,
                name=name,
                nim=nim,
                embedding=embedding,
            )
            await self.face_store.delete_face(existing.id)
            face_id = await self.face_store.add_face(face_data)
            message = f"Wajah {name} (NIM: {nim}) berhasil diperbarui"
        else:
            # Create new face
            face_data = FaceData.create(name=name, nim=nim, embedding=embedding)
            face_id = await self.face_store.add_face(face_data)
            message = f"Wajah {name} (NIM: {nim}) berhasil didaftarkan"
        
        return FaceRegistrationResult(
            success=True,
            face_id=face_id,
            name=name,
            nim=nim,
            message=message,
        )


class RegisterBulkFacesUseCase:
    """Use case for bulk face registration (initialization)."""
    
    def __init__(
        self,
        face_recognition: IFaceRecognitionAdapter,
        face_store: IFaceVectorStore,
    ):
        """Initialize use case.
        
        Args:
            face_recognition: Face recognition adapter
            face_store: Face vector store
        """
        self.face_recognition = face_recognition
        self.face_store = face_store
        self.register_single = RegisterFaceUseCase(face_recognition, face_store)
    
    async def execute(
        self,
        faces: List[BulkFaceRegistrationItem],
        skip_existing: bool = True,
    ) -> List[FaceRegistrationResult]:
        """Register multiple faces.
        
        Args:
            faces: List of faces to register
            skip_existing: Skip faces with existing NIM
            
        Returns:
            List[FaceRegistrationResult]: Results for each face
        """
        results = []
        
        for face in faces:
            result = await self.register_single.execute(
                name=face.name,
                nim=face.nim,
                image_base64=face.image_base64,
                update_if_exists=not skip_existing,
            )
            results.append(result)
        
        return results


class GetRegisteredFacesUseCase:
    """Use case for getting all registered faces."""
    
    def __init__(self, face_store: IFaceVectorStore):
        """Initialize use case.
        
        Args:
            face_store: Face vector store
        """
        self.face_store = face_store
    
    async def execute(self) -> List[FaceData]:
        """Get all registered faces.
        
        Returns:
            List[FaceData]: List of registered faces
        """
        return await self.face_store.get_all_faces()


class DeleteFaceUseCase:
    """Use case for deleting a registered face."""
    
    def __init__(self, face_store: IFaceVectorStore):
        """Initialize use case.
        
        Args:
            face_store: Face vector store
        """
        self.face_store = face_store
    
    async def execute(self, face_id: str) -> bool:
        """Delete a registered face.
        
        Args:
            face_id: Face ID to delete
            
        Returns:
            bool: True if deleted successfully
        """
        return await self.face_store.delete_face(face_id)


class RecognizeFaceUseCase:
    """Use case for recognizing a face from current camera frame."""
    
    def __init__(
        self,
        face_recognition: IFaceRecognitionAdapter,
        face_store: IFaceVectorStore,
    ):
        """Initialize use case.
        
        Args:
            face_recognition: Face recognition adapter
            face_store: Face vector store
        """
        self.face_recognition = face_recognition
        self.face_store = face_store
    
    async def get_current_face_embedding(
        self, image_base64: str
    ) -> Optional[List[float]]:
        """Get embedding from current camera frame.
        
        Args:
            image_base64: Base64-encoded image
            
        Returns:
            Optional[List[float]]: Face embedding or None
        """
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            return None
        
        return self.face_recognition.get_embedding(image_bytes)

