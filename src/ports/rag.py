"""RAG (Retrieval Augmented Generation) port interfaces."""
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

from src.domain.entities.document import Document, DocumentMetadata, RetrievalResult, NodeMetadata


class IRAGAdapter(ABC):
    """Interface for RAG adapter using vector database."""
    
    @abstractmethod
    async def retrieve(self, query: str, limit: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant documents.
        
        Args:
            query: Retrieval query
            limit: Maximum number of results
            
        Returns:
            List[RetrievalResult]: Relevant document nodes with scores
        """
        pass
    
    @abstractmethod
    async def add_document(
        self,
        document_id: str,
        file_path: Path,
        metadata: DocumentMetadata,
        node_metadata_template: Optional[NodeMetadata] = None,
    ) -> int:
        """Add a document to the vector store.
        
        Args:
            document_id: Unique document ID
            file_path: Path to the document file
            metadata: Document metadata
            node_metadata_template: Template for node metadata (optional)
            
        Returns:
            int: Number of chunks created
        """
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            bool: True if deleted successfully
        """
        pass
    
    @abstractmethod
    async def get_all_document_ids(self) -> List[str]:
        """Get all document IDs in the vector store.
        
        Returns:
            List[str]: List of document IDs
        """
        pass


class IDocumentStorage(ABC):
    """Interface for document storage (file system)."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: Path) -> bool:
        """Delete file from storage.
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if deleted successfully
        """
        pass
    
    @abstractmethod
    async def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata.
        
        Args:
            document_id: Document ID
            
        Returns:
            Optional[DocumentMetadata]: Document metadata or None
        """
        pass
    
    @abstractmethod
    async def list_all_documents(self) -> List[Document]:
        """List all documents in storage.
        
        Returns:
            List[Document]: List of all documents
        """
        pass
    
    @abstractmethod
    async def file_exists(self, document_id: str) -> bool:
        """Check if document file exists.
        
        Args:
            document_id: Document ID
            
        Returns:
            bool: True if file exists
        """
        pass
