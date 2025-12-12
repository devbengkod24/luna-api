"""Document management use cases for RAG."""
import uuid
from typing import List, Optional
from pathlib import Path

from src.ports.rag import IRAGAdapter, IDocumentStorage
from src.domain.entities.document import Document, DocumentType, RetrievalResult, NodeMetadata


class UploadDocumentUseCase:
    """Use case for uploading a document."""
    
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc"}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(
        self,
        rag_adapter: IRAGAdapter,
        document_storage: IDocumentStorage,
    ):
        """Initialize use case.
        
        Args:
            rag_adapter: RAG adapter for vector store
            document_storage: Document storage for file system
        """
        self.rag_adapter = rag_adapter
        self.document_storage = document_storage
    
    async def execute(
        self,
        file_content: bytes,
        filename: str,
        tags: Optional[List[str]] = None,
        source: str = "",
        document_type: str = "",
        section: str = "",
        language: str = "id",
        year: Optional[int] = None,
    ) -> dict:
        """Upload and process a document.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            tags: List of tags for the document
            source: Source system/repository name
            document_type: Document function/purpose (not file format)
            section: Document section
            language: Language code
            year: Year of document
            
        Returns:
            dict: Upload result
        """
        # Validate file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            return {
                "success": False,
                "message": f"Format file tidak didukung. Hanya mendukung: {', '.join(self.ALLOWED_EXTENSIONS)}",
            }
        
        # Validate file size
        if len(file_content) > self.MAX_FILE_SIZE:
            return {
                "success": False,
                "message": f"Ukuran file terlalu besar. Maksimal {self.MAX_FILE_SIZE // (1024 * 1024)}MB",
            }
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        try:
            # Save file to storage
            file_path = await self.document_storage.save_file(
                file_content=file_content,
                filename=filename,
                document_id=document_id,
            )
            
            # Get document metadata
            document_metadata = await self.document_storage.get_document_metadata(document_id)
            if not document_metadata:
                raise RuntimeError("Failed to get document metadata")
            
            # Create node metadata template if any metadata fields provided
            node_metadata_template = None
            if tags or source or document_type or section or language != "id" or year is not None:
                node_metadata_template = NodeMetadata(
                    doc_id=document_id,  # Will be overridden per chunk
                    chunk_id="",  # Will be auto-generated per chunk
                    tags=tags or [],
                    source=source,
                    document_type=document_type,
                    section=section,
                    language=language,
                    year=year,
                    position=0,  # Will be auto-generated per chunk
                )
            
            # Add to RAG vector store
            num_chunks = await self.rag_adapter.add_document(
                document_id=document_id,
                file_path=file_path,
                metadata=document_metadata,
                node_metadata_template=node_metadata_template,
            )
            
            # Update metadata with chunk count
            await self.document_storage.update_metadata_chunks(document_id, num_chunks)
            
            return {
                "success": True,
                "message": f"Dokumen {filename} berhasil diupload",
                "document_id": document_id,
                "filename": filename,
                "num_chunks": num_chunks,
            }
            
        except Exception as e:
            # Cleanup on error
            try:
                await self.document_storage.delete_file(Path(file_path) if 'file_path' in locals() else None)
                await self.rag_adapter.delete_document(document_id)
            except:
                pass
            
            return {
                "success": False,
                "message": f"Gagal mengupload dokumen: {str(e)}",
            }


class ListDocumentsUseCase:
    """Use case for listing all documents."""
    
    def __init__(self, document_storage: IDocumentStorage):
        """Initialize use case.
        
        Args:
            document_storage: Document storage
        """
        self.document_storage = document_storage
    
    async def execute(self) -> List[Document]:
        """List all documents.
        
        Returns:
            List[Document]: List of all documents
        """
        return await self.document_storage.list_all_documents()


class DeleteDocumentUseCase:
    """Use case for deleting a document."""
    
    def __init__(
        self,
        rag_adapter: IRAGAdapter,
        document_storage: IDocumentStorage,
    ):
        """Initialize use case.
        
        Args:
            rag_adapter: RAG adapter
            document_storage: Document storage
        """
        self.rag_adapter = rag_adapter
        self.document_storage = document_storage
    
    async def execute(self, document_id: str) -> dict:
        """Delete a document.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            dict: Delete result
        """
        # Check if document exists
        metadata = await self.document_storage.get_document_metadata(document_id)
        if not metadata:
            return {
                "success": False,
                "message": f"Dokumen dengan ID {document_id} tidak ditemukan",
            }
        
        try:
            # Get file path
            documents = await self.document_storage.list_all_documents()
            doc = next((d for d in documents if d.id == document_id), None)
            
            if not doc:
                return {
                    "success": False,
                    "message": f"Dokumen dengan ID {document_id} tidak ditemukan",
                }
            
            # Delete from vector store
            await self.rag_adapter.delete_document(document_id)
            
            # Delete from file storage
            await self.document_storage.delete_file(Path(doc.file_path))
            
            return {
                "success": True,
                "message": f"Dokumen {metadata.filename} berhasil dihapus",
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Gagal menghapus dokumen: {str(e)}",
            }