# """General file storage use case - simple business logic only."""
# import uuid
# import logging
# from pathlib import Path
# from typing import Dict, Optional

# from src.ports.storage import IFileStorage, StorageType
# from src.config import settings

# logger = logging.getLogger(__name__)


# class StorageUseCase:
#     """General use case for file storage - works for both RAG documents and face images.
    
#     This use case assumes all validation has been done at the schema level.
#     It only handles:
#     - File path generation
#     - File storage operations
#     """
    
#     def __init__(self, file_storage: IFileStorage, storage_type: StorageType):
#         """Initialize storage use case.
        
#         Args:
#             file_storage: File storage adapter
#             storage_type: Type of storage (RAG or FACES)
#         """
#         self.file_storage = file_storage
#         self.storage_type = storage_type
    
#     def _generate_file_path(self, filename: str, file_id: Optional[str] = None) -> Path:
#         """Generate file path for storage.
        
#         Args:
#             filename: Sanitized filename (already validated)
#             file_id: Optional unique file ID (UUID). If not provided, will be generated.
            
#         Returns:
#             Path: Filename with ID for storage
#         """
#         # Get extension from filename
#         file_path_obj = Path(filename)
#         extension = file_path_obj.suffix.lower()
        
#         # Generate unique file ID if not provided
#         if not file_id:
#             file_id = str(uuid.uuid4())
        
#         # Build final filename: {file_id}{extension}
#         final_filename = f"{file_id}{extension}"
        
#         return Path(final_filename)
    
#     async def upload_file(
#         self,
#         file_content: bytes,
#         filename: str,
#         file_id: Optional[str] = None
#     ) -> Dict:
#         """Upload a file to storage.
        
#         Args:
#             file_content: File content as bytes (already validated)
#             filename: Sanitized filename (already validated)
#             file_id: Optional unique file ID. If not provided, UUID will be generated.
            
#         Returns:
#             Dict: Upload result with success status, message, and file info
#         """
#         try:
#             # Generate file path (just filename with ID)
#             file_path = self._generate_file_path(filename, file_id)
            
#             # Get base storage path from adapter
#             base_path = getattr(self.file_storage, 'base_path', None)
#             if not base_path:
#                 base_path = getattr(self.file_storage, 'storage_path', None)
#             if not base_path:
#                 # Fallback: construct from storage type
#                 if self.storage_type == StorageType.RAG:
#                     base_path = Path(getattr(settings, 'rag_storage_path', 'storage/rag'))
#                 else:
#                     base_path = Path(getattr(settings, 'faces_dir', 'storage/faces'))
            
#             # Construct full path
#             full_path = base_path / file_path
            
#             # Save file
#             saved_path = await self.file_storage.save_file(file_content, full_path)
            
#             logger.info(
#                 f"Successfully uploaded file: {saved_path.name} "
#                 f"({len(file_content)} bytes) to {self.storage_type.value} storage"
#             )
            
#             final_file_id = file_id or Path(file_path).stem
#             return {
#                 "success": True,
#                 "message": f"File {filename} berhasil diupload",
#                 "file_id": final_file_id,
#                 "filename": filename,
#                 "file_path": str(saved_path),
#                 "file_size": len(file_content),
#             }
            
#         except Exception as e:
#             logger.error(f"Error uploading file {filename}: {e}", exc_info=True)
#             return {
#                 "success": False,
#                 "message": f"Gagal mengupload file: {str(e)}",
#             }
    
#     async def delete_file(self, file_id: str, allowed_extensions: set) -> Dict:
#         """Delete a file from storage.
        
#         Args:
#             file_id: File ID (UUID) to delete (already validated)
#             allowed_extensions: Set of allowed extensions for this storage type
            
#         Returns:
#             Dict: Delete result with success status and message
#         """
#         try:
#             # Get base path
#             base_path = getattr(self.file_storage, 'base_path', None)
#             if not base_path:
#                 base_path = getattr(self.file_storage, 'storage_path', None)
#             if not base_path:
#                 if self.storage_type == StorageType.RAG:
#                     base_path = Path(getattr(settings, 'rag_storage_path', 'storage/rag'))
#                 else:
#                     base_path = Path(getattr(settings, 'faces_dir', 'storage/faces'))
            
#             # Try to find file with any allowed extension
#             file_path = None
#             for ext in allowed_extensions:
#                 potential_path = base_path / f"{file_id}{ext}"
#                 if await self.file_storage.file_exists(potential_path):
#                     file_path = potential_path
#                     break
            
#             if not file_path:
#                 return {
#                     "success": False,
#                     "message": f"File dengan ID {file_id} tidak ditemukan",
#                 }
            
#             # Delete file
#             deleted = await self.file_storage.delete_file(file_path)
            
#             if deleted:
#                 logger.info(f"Successfully deleted file: {file_id}")
#                 return {
#                     "success": True,
#                     "message": f"File {file_id} berhasil dihapus",
#                 }
#             else:
#                 return {
#                     "success": False,
#                     "message": f"Gagal menghapus file {file_id}",
#                 }
                
#         except Exception as e:
#             logger.error(f"Error deleting file {file_id}: {e}", exc_info=True)
#             return {
#                 "success": False,
#                 "message": f"Gagal menghapus file: {str(e)}",
#             }
        
#     async def file_exists(self, file_id: str, allowed_extensions: set) -> Optional[Path]:
#         """Check if file exists by file ID.
        
#         Args:
#             file_id: File ID (UUID) to check (already validated)
#             allowed_extensions: Set of allowed extensions for this storage type
            
#         Returns:
#             Optional[Path]: File path if exists, None otherwise
#         """
#         # Get base path
#         base_path = getattr(self.file_storage, 'base_path', None)
#         if not base_path:
#             base_path = getattr(self.file_storage, 'storage_path', None)
#         if not base_path:
#             if self.storage_type == StorageType.RAG:
#                 base_path = Path(getattr(settings, 'rag_storage_path', 'storage/rag'))
#             else:
#                 base_path = Path(getattr(settings, 'faces_dir', 'storage/faces'))
        
#         # Try to find file with any allowed extension
#         for ext in allowed_extensions:
#             potential_path = base_path / f"{file_id}{ext}"
#             if await self.file_storage.file_exists(potential_path):
#                 return potential_path
        
#         return None
    
#     async def list_files(self) -> Dict:
#         """List all files in storage.
        
#         Returns:
#             Dict: List result with file information
#         """
#         try:
#             files = await self.file_storage.list_files()
            
#             file_list = []
#             for file_path in files:
#                 # Extract file ID from filename (remove extension)
#                 file_id = file_path.stem
#                 file_list.append({
#                     "file_id": file_id,
#                     "filename": file_path.name,
#                     "file_path": str(file_path),
#                 })
            
#             return {
#                 "success": True,
#                 "files": file_list,
#                 "count": len(file_list),
#             }
            
#         except Exception as e:
#             logger.error(f"Error listing files: {e}", exc_info=True)
#             return {
#                 "success": False,
#                 "message": f"Gagal mendapatkan daftar file: {str(e)}",
#                 "files": [],
#                 "count": 0,
#             }
