"""Face registration use cases."""
import base64
from typing import List
import logging
from pathlib import Path
import aiofiles

from src.ports.vision import IFaceRecognitionAdapter, IFaceVectorStore
from src.domain.entities.vision import (
    FaceData,
    FaceRegistrationResult,
    RegisterFaceInput,
)

MAX_BASE64_SIZE = 10 * 1024 * 1024
MAX_EMBEDDING_LENGTH = 512
MIN_EMBEDDING_LENGTH = 128 

logger = logging.getLogger(__name__)

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
        if not faces:
            raise ValueError("Faces list cannot be empty")
        
        logger.info(f"Registering {len(faces)} face(s), skip_existing={skip_existing}")

        results = []
        
        for i, face_input in enumerate(faces, 1):
            logger.debug(f"Processing face {i}/{len(faces)}: {face_input.name} ({face_input.nim})")
            
            result = await self._register_single_face(
                name=face_input.name,
                nim=face_input.nim,
                image_base64=face_input.image_base64,
                update_if_exists=not skip_existing,
            )
            results.append(result)
            
            if result.success:
                logger.info(f"Successfully registered face {i}/{len(faces)}: {result.name} ({result.nim})")
            else:
                logger.warning(f"Failed to register face {i}/{len(faces)}: {result.message}")
        
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Registration complete: {success_count}/{len(faces)} successful")
        
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
        name = name.strip() if name else ""
        nim = nim.strip() if nim else ""
        # Step 1: Decode image
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
            
            if len(image_bytes) == 0:
                return FaceRegistrationResult(
                    success=False,
                    message="Image data is empty"
                )
        except base64.binascii.Error as e:
            logger.debug(f"Base64 decode error for {name} ({nim}): {e}")
            return FaceRegistrationResult(
                success=False,
                message="Invalid base64 image format"
            )
        except Exception as e:
            logger.error(f"Unexpected error decoding base64 for {name} ({nim}): {e}", exc_info=True)
            return FaceRegistrationResult(
                success=False,
                message="Failed to decode image"
            )
        
        # Step 2 & 3: Detect faces and validate
        try:
            detected_faces = self.face_recognition.detect_faces(
                image_bytes,
                include_crops=False
            )
        except ValueError as e:
            logger.warning(f"Face detection validation error for {name} ({nim}): {e}")
            return FaceRegistrationResult(
                success=False,
                message=f"Face detection validation failed: {str(e)}"
            )
        except RuntimeError as e:
            logger.error(f"Face detection runtime error for {name} ({nim}): {e}", exc_info=True)
            return FaceRegistrationResult(
                success=False,
                message="Face detection service error"
            )
        except Exception as e:
            logger.error(f"Unexpected face detection error for {name} ({nim}): {e}", exc_info=True)
            return FaceRegistrationResult(
                success=False,
                message="Face detection failed"
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
        
        if not embedding:
            logger.warning(f"Empty embedding for {name} ({nim})")
            return FaceRegistrationResult(
                success=False,
                message="Failed to extract face embedding"
            )
        
        # Validate embedding length
        if not isinstance(embedding, list) or len(embedding) < MIN_EMBEDDING_LENGTH:
            logger.warning(f"Invalid embedding length for {name} ({nim}): {len(embedding) if isinstance(embedding, list) else 'not a list'}")
            return FaceRegistrationResult(
                success=False,
                message="Invalid face embedding extracted"
            )
        
        if len(embedding) > MAX_EMBEDDING_LENGTH:
            logger.warning(f"Embedding too long for {name} ({nim}): {len(embedding)}")
            return FaceRegistrationResult(
                success=False,
                message="Invalid face embedding extracted"
            )
        
        # Step 5: Check if NIM already exists
        try:
            existing_face = await self.face_store.get_face_by_nim(nim)
        except ValueError as e:
            logger.warning(f"Database query validation error for {name} ({nim}): {e}")
            return FaceRegistrationResult(
                success=False,
                message="Invalid query parameters"
            )
        except ConnectionError as e:
            logger.error(f"Database connection error for {name} ({nim}): {e}", exc_info=True)
            return FaceRegistrationResult(
                success=False,
                message="Database connection error"
            )
        except Exception as e:
            logger.error(f"Database query error for {name} ({nim}): {e}", exc_info=True)
            return FaceRegistrationResult(
                success=False,
                message="Database query failed"
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
                        message=f"Face with NIM {nim} already exists. Use skip_existing=False to update."
                    )
                
                # Update existing face
                face_data = FaceData(
                    id=existing_face.id,
                    name=name,
                    nim=nim,
                    embedding=embedding,
                )
                await self.face_store.add_face(face_data)  # upsert
                
                logger.info(f"Updated face: {name} ({nim}), ID: {existing_face.id}")
                
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
                
                logger.info(f"Registered new face: {name} ({nim}), ID: {face_id}")
                
                return FaceRegistrationResult(
                    success=True,
                    face_id=face_id,
                    name=name,
                    nim=nim,
                    message="Face registered successfully"
                )
        except ValueError as e:
            logger.warning(f"Data validation error storing face {name} ({nim}): {e}")
            return FaceRegistrationResult(
                success=False,
                message=f"Invalid face data: {str(e)}"
            )
        except ConnectionError as e:
            logger.error(f"Database connection error storing face {name} ({nim}): {e}", exc_info=True)
            return FaceRegistrationResult(
                success=False,
                message="Database connection error"
            )
        except Exception as e:
            logger.error(f"Error storing face data for {name} ({nim}): {e}", exc_info=True)
            return FaceRegistrationResult(
                success=False,
                message="Failed to store face data"
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
        
        SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
        storage = Path(storage_path)
        
        logger.info(f"Scanning for faces in: {storage}")
        
        if not storage.exists():
            error_msg = f"Storage path does not exist: {storage}"
            logger.error(error_msg)
            return (0, 0, [error_msg])
        
        if not storage.is_dir():
            error_msg = f"Storage path is not a directory: {storage}"
            logger.error(error_msg)
            return (0, 0, [error_msg])
        
        # Scan for image files
        face_inputs = []
        errors = []
        
        for file_path in storage.iterdir():
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.debug(f"Skipping unsupported file: {file_path.name}")
                continue
            
            try:
                # Parse filename: Name__NIM
                name, nim = self._parse_filename(file_path.stem)
                
                # Read and encode image using async I/O
                async with aiofiles.open(file_path, 'rb') as f:
                    image_bytes = await f.read()
                    
                    if len(image_bytes) == 0:
                        error_msg = f"Empty file: {file_path.name}"
                        errors.append(error_msg)
                        logger.warning(error_msg)
                        continue
                    
                    # Check file size before encoding
                    if len(image_bytes) > MAX_BASE64_SIZE:
                        error_msg = f"File too large: {file_path.name} ({len(image_bytes)} bytes)"
                        errors.append(error_msg)
                        logger.warning(error_msg)
                        continue
                    
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                face_inputs.append(RegisterFaceInput(
                    name=name,
                    nim=nim,
                    image_base64=image_base64,
                ))
                
            except ValueError as e:
                error_msg = f"Failed to process {file_path.name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
            except OSError as e:
                error_msg = f"File I/O error processing {file_path.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
            except Exception as e:
                error_msg = f"Unexpected error processing {file_path.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
        
        total_found = len(face_inputs)
        
        if total_found == 0:
            logger.info("No valid face images found in storage")
            return (0, 0, errors)
        
        logger.info(f"Found {total_found} face image(s) to register")
        
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
                    logger.info(f"Registered: {result.name} ({result.nim})")
                else:
                    error_msg = f"{face_inputs[i].name} ({face_inputs[i].nim}): {result.message}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
            
            logger.info(f"Face initialization complete: {success_count}/{total_found} successful")
            
            return (total_found, success_count, errors)
            
        except ValueError as e:
            error_msg = f"Face registration validation failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return (total_found, 0, errors)
        except Exception as e:
            error_msg = f"Face registration failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
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
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")
        
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
        try:
            faces = await self.face_store.get_all_faces()
            logger.info(f"Retrieved {len(faces)} registered face(s)")
            return faces
        except ConnectionError as e:
            logger.error(f"Database connection error retrieving faces: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error retrieving faces: {e}", exc_info=True)
            raise


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
        if not face_id or not face_id.strip():
            raise ValueError("Face ID cannot be empty")
        
        face_id = face_id.strip()
        
        logger.info(f"Deleting face: {face_id}")
        
        try:
            result = await self.face_store.delete_face(face_id)
            
            if result:
                logger.info(f"Successfully deleted face: {face_id}")
            else:
                logger.warning(f"Face not found or already deleted: {face_id}")
            
            return result
        except ValueError as e:
            logger.warning(f"Invalid face ID: {face_id}, error: {e}")
            raise
        except ConnectionError as e:
            logger.error(f"Database connection error deleting face {face_id}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error deleting face {face_id}: {e}", exc_info=True)
            raise
