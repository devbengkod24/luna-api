"""RAG (Retrieval Augmented Generation) port interfaces."""
from abc import ABC, abstractmethod
from typing import List


class IRAGAdapter(ABC):
    """Interface for RAG adapter using vector database."""
    
    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> List[str]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List[str]: Relevant document chunks
        """
        pass
    
    @abstractmethod
    async def add_document(self, text: str, metadata: dict = None) -> str:
        """Add a document to the vector store.
        
        Args:
            text: Document text
            metadata: Optional metadata
            
        Returns:
            str: Document ID
        """
        pass

