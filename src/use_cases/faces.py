"""Face registration use cases."""
import base64
from typing import List, Optional

from src.ports.vision import IFaceRecognitionAdapter, IFaceVectorStore
from src.domain.entities.vision import (
    FaceData,
    FaceRegistrationResult,
    RegisterFaceInput,
)


class RegisterFacesUseCase:
    """Use case for registering one or multiple faces.
    
    This is the main use case for face registration, supporting both
    single and batch registration through the same interface.
    """
    
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
        faces: List[RegisterFaceInput],
        skip_existing: bool = True,
    ) -> List[FaceRegistrationResult]:
        """Register one or multiple faces.
        
        Accepts a list of faces and registers each one. Can handle both
        single face registration (list with 1 item) and batch registration.
        
        Args:
            faces: List of faces to register (can be a single item)
            skip_existing: If True, skip faces with existing NIM; if False, update them
            
        Returns:
            List[FaceRegistrationResult]: Results for each face registration attempt
        """
        results = []
        
        for face_input in faces:
            result = await self._register_single_face(
                name=face_input.name,
                nim=face_input.nim,
                image_base64=face_input.image_base64,
                update_if_exists=not skip_existing,
            )
            results.append(result)
        
        return results
    
    async def _register_single_face(
        self,
        name: str,
        nim: str,
        image_base64: str,
        update_if_exists: bool = False,
    ) -> FaceRegistrationResult:
        """Register a single face (internal method).
        
        Workflow:
        1. Decode base64 image
        2. Detect faces in the image
        3. Validate exactly one face is detected
        4. Extract face embedding
        5. Check if NIM already exists
        6. Store or update face data in vector store
        
        Args:
            name: Person's name
            nim: Person's NIM (student ID)
            image_base64: Base64-encoded image
            update_if_exists: If True, update existing face; if False, skip
            
        Returns:
            FaceRegistrationResult: Result with success status and details
        """
        # Step 1: Decode image
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            return FaceRegistrationResult(
                success=False,
                message=f"Invalid base64 image: {str(e)}"
            )
        
        # Step 2 & 3: Detect faces and validate
        try:
            detected_faces = self.face_recognition.detect_faces(
                image_bytes,
                include_crops=False
            )
        except Exception as e:
            return FaceRegistrationResult(
                success=False,
                message=f"Face detection failed: {str(e)}"
            )
        
        if not detected_faces:
            return FaceRegistrationResult(
                success=False,
                message="No face detected in the image"
            )
        
        if len(detected_faces) > 1:
            return FaceRegistrationResult(
                success=False,
                message=f"Multiple faces detected ({len(detected_faces)}). Please provide an image with exactly one face."
            )
        
        # Step 4: Extract embedding
        detected_face = detected_faces[0]
        embedding = detected_face.embedding
        
        if not embedding or len(embedding) == 0:
            return FaceRegistrationResult(
                success=False,
                message="Failed to extract face embedding"
            )
        
        # Step 5: Check if NIM already exists
        try:
            existing_face = await self.face_store.get_face_by_nim(nim)
        except Exception as e:
            return FaceRegistrationResult(
                success=False,
                message=f"Database query failed: {str(e)}"
            )
        
        # Step 6: Store or update face data
        try:
            if existing_face:
                if not update_if_exists:
                    return FaceRegistrationResult(
                        success=False,
                        face_id=existing_face.id,
                        name=existing_face.name,
                        nim=existing_face.nim,
                        message=f"Face with NIM {nim} already exists. Use update_if_exists=True to update."
                    )
                
                # Update existing face
                face_data = FaceData(
                    id=existing_face.id,
                    name=name,
                    nim=nim,
                    embedding=embedding,
                )
                await self.face_store.add_face(face_data)  # upsert
                
                return FaceRegistrationResult(
                    success=True,
                    face_id=existing_face.id,
                    name=name,
                    nim=nim,
                    message="Face updated successfully"
                )
            else:
                # Create new face
                face_data = FaceData.create(
                    name=name,
                    nim=nim,
                    embedding=embedding,
                )
                face_id = await self.face_store.add_face(face_data)
                
                return FaceRegistrationResult(
                    success=True,
                    face_id=face_id,
                    name=name,
                    nim=nim,
                    message="Face registered successfully"
                )
        except Exception as e:
            return FaceRegistrationResult(
                success=False,
                message=f"Failed to store face data: {str(e)}"
            )
    
    async def initialize_from_storage(
        self, 
        storage_path: str = "storage/faces"
    ) -> tuple[int, int, list[str]]:
        """Initialize faces from storage folder.
        
        Scans the storage folder and registers all face images found.
        Filename format: Name__NIM.ext (double underscore separator)
        
        Args:
            storage_path: Path to folder containing face images
            
        Returns:
            tuple[int, int, list[str]]: (total_found, success_count, error_messages)
        """
        from pathlib import Path
        
        SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
        storage = Path(storage_path)
        
        print(f"🔍 Scanning for faces in: {storage}")
        
        if not storage.exists():
            print(f"⚠️  Storage path does not exist: {storage}")
            return (0, 0, [f"Storage path not found: {storage}"])
        
        # Scan for image files
        face_inputs = []
        errors = []
        
        for file_path in storage.iterdir():
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            
            try:
                # Parse filename: Name__NIM
                name, nim = self._parse_filename(file_path.stem)
                
                # Read and encode image
                with open(file_path, 'rb') as f:
                    image_bytes = f.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                face_inputs.append(RegisterFaceInput(
                    name=name,
                    nim=nim,
                    image_base64=image_base64,
                ))
                
            except Exception as e:
                error_msg = f"Failed to process {file_path.name}: {str(e)}"
                errors.append(error_msg)
                print(f"⚠️  {error_msg}")
        
        total_found = len(face_inputs)
        
        if total_found == 0:
            print("ℹ️  No valid face images found in storage")
            return (0, 0, errors)
        
        print(f"📦 Found {total_found} face image(s) to register")
        
        # Register all faces (skip existing by default)
        try:
            results = await self.execute(
                faces=face_inputs,
                skip_existing=True,
            )
            
            # Count successes and collect errors
            success_count = 0
            for i, result in enumerate(results):
                if result.success:
                    success_count += 1
                    print(f"✅ Registered: {result.name} ({result.nim})")
                else:
                    error_msg = f"{face_inputs[i].name} ({face_inputs[i].nim}): {result.message}"
                    errors.append(error_msg)
                    print(f"❌ {error_msg}")
            
            print(f"🎉 Face initialization complete: {success_count}/{total_found} successful")
            
            return (total_found, success_count, errors)
            
        except Exception as e:
            error_msg = f"Face registration failed: {str(e)}"
            errors.append(error_msg)
            print(f"❌ {error_msg}")
            return (total_found, 0, errors)
    
    def _parse_filename(self, filename: str) -> tuple[str, str]:
        """Parse filename to extract name and NIM.
        
        Expected format: Name_With_Underscores__NIM
        - Underscores in name represent spaces
        - Double underscore separates name from NIM
        
        Args:
            filename: Filename without extension
            
        Returns:
            tuple[str, str]: (name, nim)
            
        Raises:
            ValueError: If filename format is invalid
        """
        if '__' not in filename:
            raise ValueError(
                f"Invalid filename format. Expected 'Name__NIM', got '{filename}'. "
                "Use double underscore to separate name from NIM."
            )
        
        # Split by double underscore
        parts = filename.split('__')
        
        if len(parts) != 2:
            raise ValueError(
                f"Invalid filename format. Expected exactly one '__' separator, "
                f"found {len(parts) - 1} in '{filename}'"
            )
        
        name_part, nim_part = parts
        
        # Convert underscores to spaces in name
        name = name_part.replace('_', ' ').strip()
        nim = nim_part.strip()
        
        if not name:
            raise ValueError(f"Name is empty in filename: {filename}")
        
        if not nim:
            raise ValueError(f"NIM is empty in filename: {filename}")
        
        return (name, nim)


class GetFacesUseCase:
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
