"""RAG (Retrieval Augmented Generation) port interfaces."""
from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path

from src.domain.entities.knowledge import RetrievalResult

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
        file_path: Path,
    ) -> int:
        """Add a document to the vector store.
        
        Args:
            file_path: Path to the document file
            
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
    
    @abstractmethod
    async def get_all_documents_metadata(self) -> List[Dict]:
        """Get all documents with their metadata from vector store.
        
        Returns:
            List[Dict]: List of documents with metadata (document_id, file_name, file_type, file_size, creation_date)
        """
        pass