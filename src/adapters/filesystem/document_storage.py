"""Document storage adapter for file system."""
import json
import aiofiles
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from src.config import settings
from src.ports.rag import IDocumentStorage
from src.domain.entities.document import Document, DocumentMetadata, DocumentType


class FileSystemDocumentStorage(IDocumentStorage):
    """Document storage using file system."""
    
    def __init__(self):
        """Initialize file system storage."""
        self.storage_path = Path(settings.rag_storage_path)
        self.metadata_path = Path(settings.rag_metadata_path)
        
        # Create directories if not exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Document storage initialized at: {self.storage_path}")
    
    def _get_file_path(self, document_id: str, filename: str) -> Path:
        """Get file path for a document.
        
        Args:
            document_id: Document ID
            filename: Original filename
            
        Returns:
            Path: File path
        """
        # Preserve file extension
        extension = Path(filename).suffix
        return self.storage_path / f"{document_id}{extension}"
    
    def _get_metadata_path(self, document_id: str) -> Path:
        """Get metadata file path.
        
        Args:
            document_id: Document ID
            
        Returns:
            Path: Metadata file path
        """
        return self.metadata_path / f"{document_id}.json"
    
    def _detect_file_type(self, filename: str) -> DocumentType:
        """Detect file type from filename.
        
        Args:
            filename: Filename
            
        Returns:
            DocumentType: Detected file type
        """
        ext = Path(filename).suffix.lower()
        
        type_map = {
            ".pdf": DocumentType.PDF,
            ".txt": DocumentType.TXT,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOCX,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
            ".md": DocumentType.MD,
        }
        
        return type_map.get(ext, DocumentType.TXT)
    
    async def save_file(
        self,
        file_content: bytes,
        filename: str,
        document_id: str,
    ) -> Path:
        """Save uploaded file to storage.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            document_id: Unique document ID
            
        Returns:
            Path: Path where file was saved
        """
        file_path = self._get_file_path(document_id, filename)
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        # Create and save metadata
        file_type = self._detect_file_type(filename)
        metadata = DocumentMetadata(
            filename=filename,
            file_type=file_type,
            file_size=len(file_content),
            uploaded_at=datetime.utcnow(),
        )
        
        metadata_path = self._get_metadata_path(document_id)
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps(metadata.to_dict(), indent=2))
        
        print(f"Saved document {document_id}: {filename} ({file_type.value})")
        return file_path
    
    async def delete_file(self, file_path: Path) -> bool:
        """Delete file from storage.
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            # Extract document_id from filename
            document_id = file_path.stem
            
            # Delete file
            if file_path.exists():
                file_path.unlink()
            
            # Delete metadata
            metadata_path = self._get_metadata_path(document_id)
            if metadata_path.exists():
                metadata_path.unlink()
            
            print(f"Deleted document {document_id}")
            return True
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return False
    
    async def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata.
        
        Args:
            document_id: Document ID
            
        Returns:
            Optional[DocumentMetadata]: Document metadata or None
        """
        metadata_path = self._get_metadata_path(document_id)
        
        if not metadata_path.exists():
            return None
        
        try:
            async with aiofiles.open(metadata_path, 'r') as f:
                data = json.loads(await f.read())
            
            return DocumentMetadata(
                filename=data["filename"],
                file_type=DocumentType(data["file_type"]),
                file_size=data["file_size"],
                uploaded_at=datetime.fromisoformat(data["uploaded_at"]),
                num_chunks=data.get("num_chunks"),
            )
        except Exception as e:
            print(f"Error reading metadata for {document_id}: {e}")
            return None
    
    async def list_all_documents(self) -> List[Document]:
        """List all documents in storage.
        
        Returns:
            List[Document]: List of all documents
        """
        documents = []
        
        for metadata_file in self.metadata_path.glob("*.json"):
            document_id = metadata_file.stem
            metadata = await self.get_document_metadata(document_id)
            
            if metadata:
                # Find file path
                file_path = None
                for ext in [".pdf", ".txt", ".docx", ".html", ".md"]:
                    potential_path = self.storage_path / f"{document_id}{ext}"
                    if potential_path.exists():
                        file_path = potential_path
                        break
                
                if file_path:
                    documents.append(Document(
                        id=document_id,
                        metadata=metadata,
                        file_path=str(file_path),
                    ))
        
        return documents
    
    async def file_exists(self, document_id: str) -> bool:
        """Check if document file exists.
        
        Args:
            document_id: Document ID
            
        Returns:
            bool: True if file exists
        """
        metadata_path = self._get_metadata_path(document_id)
        return metadata_path.exists()
    
    async def update_metadata_chunks(self, document_id: str, num_chunks: int):
        """Update number of chunks in metadata.
        
        Args:
            document_id: Document ID
            num_chunks: Number of chunks
        """
        metadata = await self.get_document_metadata(document_id)
        if metadata:
            metadata.num_chunks = num_chunks
            metadata_path = self._get_metadata_path(document_id)
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata.to_dict(), indent=2))

