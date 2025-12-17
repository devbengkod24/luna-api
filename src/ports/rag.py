"""RAG (Retrieval Augmented Generation) port interfaces."""
from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities.knowledge import RetrievalResultEntity


class IRAGAdapter(ABC):
    """Interface untuk RAG *retriever* (hanya baca / search).

    Concern kelas ini: melakukan retrieval dari vector store
    untuk dipakai oleh agent / LLM.
    """

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 5) -> list[RetrievalResultEntity]:
        """Retrieve relevant documents.

        Args:
            query: Retrieval query
            limit: Maximum number of results

        Returns:
            list[RetrievalResultEntity]: Relevant document nodes with scores
        """
        ...


class IRAGStore(ABC):
    """Interface untuk RAG *store* (tulis & metadata ke vector DB).

    Concern kelas ini: komunikasi ke vector database (Qdrant) untuk:
    - add_document
    - delete_document
    - get_all_document_ids
    - get_all_documents_metadata
    """

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
        ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store.

        Args:
            document_id: Document ID to delete

        Returns:
            bool: True if deleted successfully
        """
        ...

    @abstractmethod
    async def get_all_document_ids(self) -> list[str]:
        """Get all document IDs in the vector store.

        Returns:
            list[str]: List of document IDs
        """
        ...

    @abstractmethod
    async def get_all_documents_metadata(self) -> list[dict]:
        """Get all documents with their metadata from vector store.

        Returns:
            list[dict]: List of documents with metadata
                       (document_id, file_name, file_type, file_size, creation_date)
        """
        ...
