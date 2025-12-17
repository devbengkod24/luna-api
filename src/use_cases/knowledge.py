"""Document management use cases for RAG knowledge base."""

import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import aiofiles
import uuid

from src.config import settings
from src.ports.rag import IRAGStore
from src.ports.storage import IFileStorage

logger = logging.getLogger(__name__)


class UploadKnowledgeUseCase:
    """Use case for uploading a knowledge to RAG knowledge base."""
    
    def __init__(
        self,
        file_storage: IFileStorage,
        rag_store: IRAGStore,
    ):
        """Initialize use case."""
        self.file_storage = file_storage
        self.rag_store = rag_store
    
    async def execute(
        self,
        file_content: bytes,
        filename: str,
    ) -> dict[str, Any]:
        """Upload and process a document."""
        try:
            file_path_obj = Path(filename)
            original_name = file_path_obj.stem.lower()
            extension = file_path_obj.suffix.lower()
            
            file_id = str(uuid.uuid4().hex[:8])
            filename = f"{original_name}_{file_id}{extension}"

            # Step 1: Save file to storage
            file_path = await self.file_storage.save_file(
                file_content=file_content,
                filename=filename,
            )
            
            logger.info(f"Saved document file: {file_path}")
            
            # Step 2: Add document to RAG vector store
            num_chunks = await self.rag_store.add_document(
                file_path=file_path,
            )
            
            logger.info(f"Added document to vector store: {filename} ({num_chunks} chunks)")
            
            return {
                "success": True,
                "message": f"Dokumen {filename} berhasil diupload",
                "filename": filename,
                "num_chunks": num_chunks,
            }
            
        except Exception as e:
            logger.error(f"Error uploading document {filename}: {e}", exc_info=True)
            
            try:
                await self.file_storage.delete_file(filename)
                await self.rag_store.delete_document(filename)
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}", exc_info=True)
            
            return {
                "success": False,
                "message": f"Gagal mengupload dokumen: {str(e)}",
            }
    
    async def seed_knowledge(
        self,
        storage_path: str = None,
        skip_existing: bool = True,
    ) -> tuple[int, int, list[str]]:
        """Initialize knowledge documents from storage folder."""
        if storage_path is None:
            storage_path = settings.rag_dir
        
        storage = Path(storage_path)
        
        logger.info(f"Scanning for knowledge documents in: {storage}")
        
        if not storage.exists():
            error_msg = f"Storage path does not exist: {storage}"
            logger.error(error_msg)
            return (0, 0, [error_msg])
        
        if not storage.is_dir():
            error_msg = f"Storage path is not a directory: {storage}"
            logger.error(error_msg)
            return (0, 0, [error_msg])
        
        indexed_filenames = set()
        if skip_existing:
            try:
                documents_metadata = await self.rag_store.get_all_documents_metadata()
                indexed_filenames = {doc.get("file_name", "") for doc in documents_metadata}
                logger.info(f"Found {len(indexed_filenames)} already indexed documents")
            except Exception as e:
                logger.warning(f"Failed to get indexed documents: {e}, will process all files")
        
        document_files = []
        errors = []
        
        for file_path in storage.rglob('*'):
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in settings.rag_allowed_extensions:
                logger.debug(f"Skipping unsupported file: {file_path.name}")
                continue
            
            if skip_existing and file_path.name in indexed_filenames:
                logger.debug(f"Skipping already indexed file: {file_path.name}")
                continue
            
            document_files.append(file_path)
        
        total_found = len(document_files)
        
        if total_found == 0:
            logger.info("No valid document files found in storage")
            return (0, 0, errors)
        
        logger.info(f"Found {total_found} document file(s) to upload")
        
        # Upload all documents
        success_count = 0
        for i, file_path in enumerate(document_files, 1):
            logger.info(f"[{i}/{total_found}] Processing: {file_path.name}")
            
            try:
                async with aiofiles.open(file_path, 'rb') as f:
                    file_content = await f.read()
                
                if len(file_content) == 0:
                    error_msg = f"Empty file: {file_path.name}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                    continue
                
                if len(file_content) > settings.rag_max_file_size:
                    error_msg = f"File too large: {file_path.name} ({len(file_content)} bytes)"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                    continue
                
                result = await self.execute(
                    file_content=file_content,
                    filename=file_path.name,
                )
                
                if result["success"]:
                    success_count += 1
                    logger.info(
                        f"✓ Successfully uploaded: {file_path.name} "
                        f"({result.get('num_chunks', 0)} chunks)"
                    )
                else:
                    error_msg = f"{file_path.name}: {result.get('message', 'Upload failed')}"
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
        
        logger.info(f"Knowledge seeding complete: {success_count}/{total_found} successful")
        
        return (total_found, success_count, errors)


class ListKnowledgeUseCase:
    """Use case for listing all documents in RAG knowledge base."""
    
    def __init__(
        self,
        rag_store: IRAGStore,
    ):
        """Initialize use case."""
        self.rag_store = rag_store
    
    async def execute(self) -> list[dict[str, Any]]:
        """List all knowledge."""
        try:
            documents = await self.rag_store.get_all_documents_metadata()
            
            for doc in documents:
                file_name = doc.get("file_name", "")
                encoded_filename = quote(file_name, safe='')
                doc["download_url"] = f"/v1/knowledge/download/{encoded_filename}"
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {e}", exc_info=True)
            return []


class DownloadKnowledgeUseCase:
    """Use case for downloading a knowledge file."""
    
    def __init__(
        self,
        file_storage: IFileStorage,
        rag_store: IRAGStore,
    ):
        """Initialize use case."""
        self.file_storage = file_storage
        self.rag_store = rag_store
    
    async def execute(self, filename: str) -> dict[str, Any]:
        """Download a document file by filename."""
        try:
            decoded_filename = unquote(filename)
            
            documents_metadata = await self.rag_store.get_all_documents_metadata()
            doc_metadata = None
            for doc in documents_metadata:
                if doc.get("file_name") == decoded_filename:
                    doc_metadata = doc
                    break
            
            if not doc_metadata:
                return {
                    "success": False,
                    "message": f"Dokumen dengan nama {decoded_filename} tidak ditemukan",
                }
            
            document_id = doc_metadata.get("document_id")
            if not document_id:
                return {
                    "success": False,
                    "message": f"Document ID tidak ditemukan untuk file {decoded_filename}",
                }
            
            files = await self.file_storage.list_files()
            storage_filename = None
            for f in files:
                file_stem = Path(f).stem
                if file_stem == document_id:
                    storage_filename = f
                    break
            
            if not storage_filename:
                return {
                    "success": False,
                    "message": f"File untuk dokumen {decoded_filename} tidak ditemukan di storage",
                }
            
            file_content = await self.file_storage.read_file(storage_filename)
            
            file_ext = Path(decoded_filename).suffix.lower()
            content_type_map = {
                '.pdf': 'application/pdf',
                '.txt': 'text/plain',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
            }
            content_type = content_type_map.get(file_ext, 'application/octet-stream')
            
            return {
                "success": True,
                "file_content": file_content,
                "filename": decoded_filename,
                "content_type": content_type,
            }
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return {
                "success": False,
                "message": f"File tidak ditemukan: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Error downloading document {filename}: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Gagal mengunduh dokumen: {str(e)}",
            }


class DeleteKnowledgeUseCase:
    """Use case for deleting a knowledge from RAG knowledge base."""
    
    def __init__(
        self,
        file_storage: IFileStorage,
        rag_store: IRAGStore,
    ):
        """Initialize use case."""
        self.file_storage = file_storage
        self.rag_store = rag_store
    
    async def execute(self, document_id: str) -> dict[str, Any]:
        """Delete a document."""
        try:
            indexed_ids = await self.rag_store.get_all_document_ids()
            if document_id not in indexed_ids:
                return {
                    "success": False,
                    "message": f"Dokumen dengan ID {document_id} tidak ditemukan",
                }
            
            files = await self.file_storage.list_files()
            filename = None
            for f in files:
                file_stem = Path(f).stem
                if file_stem == document_id:
                    filename = f
                    break
            
            deleted_from_vector = await self.rag_store.delete_document(document_id)
            if deleted_from_vector:
                logger.info(f"Deleted document from vector store: {document_id}")
            
            if filename and await self.file_storage.file_exists(filename):
                await self.file_storage.delete_file(filename)
                logger.info(f"Deleted document file: {filename}")
            
            return {
                "success": True,
                "message": f"Dokumen {document_id} berhasil dihapus",
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Gagal menghapus dokumen: {str(e)}",
            }