"""Document domain entities for RAG."""
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    HTML = "html"
    MD = "md"

@dataclass
class NodeMetadata:
    """Node metadata for RAG."""
    doc_id: str
    chunk_id: str
    tags: List[str] = field(default_factory=list)
    source: str = "" # asal dokumen (nama sistem atau repository), bukan nama file.
    document_type: str = "" # fungsi dokumen, bukan format
    position: int = 0
    section: str = ""
    language: str = "id"
    year: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "document_type": self.document_type,
            "position": self.position,
            "section": self.section,
            "language": self.language,
            "year": self.year,
            "tags": self.tags,
        }

@dataclass
class RetrievalResult:
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
class DocumentMetadata:
    """Document-level metadata (for API/storage)."""
    filename: str
    file_type: DocumentType
    file_size: int  # in bytes
    uploaded_at: datetime
    num_chunks: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "filename": self.filename,
            "file_type": self.file_type.value,
            "file_size": self.file_size,
            "uploaded_at": self.uploaded_at.isoformat(),
        }
        if self.num_chunks is not None:
            result["num_chunks"] = self.num_chunks
        return result

@dataclass
class Document:
    """RAG document."""
    id: str
    file_path: str
    metadata: DocumentMetadata
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "metadata": self.metadata.to_dict(),
            "file_path": self.file_path,
        }

