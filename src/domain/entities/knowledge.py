"""Document domain entities for RAG knowledge base."""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RetrievalResultEntity:
    """RAG retrieval result."""
    content: str
    score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
        }


@dataclass
class KnowledgeMetadataEntity:
    """Document-level metadata for storage and API responses."""
    filename: str
    file_type: str
    file_size: int
    uploaded_at: datetime
    num_chunks: int | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "uploaded_at": self.uploaded_at.isoformat(),
        }
        if self.num_chunks is not None:
            result["num_chunks"] = self.num_chunks
        return result


@dataclass
class KnowledgeEntity:
    """RAG knowledge."""
    id: str
    file_path: str
    metadata: KnowledgeMetadataEntity
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "metadata": self.metadata.to_dict(),
        }

