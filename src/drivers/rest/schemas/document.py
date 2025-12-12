from pydantic import BaseModel
from typing import Optional, List

class UploadDocumentRequest(BaseModel):
    """Request model for document upload metadata (NodeMetadata fields)."""
    tags: Optional[List[str]] = []
    source: str = ""
    document_type: str = ""  # fungsi dokumen, bukan format
    section: str = ""
    language: str = "id"
    year: Optional[int] = None

class UploadDocumentResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    message: str
    document_id: Optional[str] = None
    filename: Optional[str] = None
    num_chunks: Optional[int] = None


class DocumentResponse(BaseModel):
    """Response model for document."""
    id: str
    filename: str
    file_type: str
    file_size: int
    uploaded_at: str
    num_chunks: Optional[int] = None


class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion."""
    success: bool
    message: str
