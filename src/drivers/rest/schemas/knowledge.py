"""Knowledge/document schemas with validators for RAG."""

from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

from src.config import settings

DANGEROUS_PATTERNS = [
    '..',
    '/',
    '\\',
    '\x00',
]

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues."""
    if not filename:
        return "file"
    
    sanitized = filename
    for pattern in DANGEROUS_PATTERNS:
        sanitized = sanitized.replace(pattern, '_')
    
    sanitized = sanitized.strip('. ')
    
    if len(sanitized) > 255:
        file_path = Path(sanitized)
        name_without_ext = file_path.stem
        extension = file_path.suffix
        max_name_length = 255 - len(extension)
        sanitized = name_without_ext[:max_name_length] + extension
    
    return sanitized or "file"


class UploadKnowledgeRequest(BaseModel):
    """Request model for knowledge upload with validation."""
    filename: str
    file_content: bytes

    @field_validator('filename')
    @classmethod
    def validate_and_sanitize_filename(cls, v: str) -> str:
        """Validate and sanitize filename."""
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        
        sanitized = sanitize_filename(v)
        
        if not sanitized or sanitized == "file":
            raise ValueError("Filename becomes empty after sanitization")
        
        return sanitized
    
    @field_validator('file_content')
    @classmethod
    def validate_file_content(cls, v: bytes) -> bytes:
        """Validate file content is not empty."""
        if not v or len(v) == 0:
            raise ValueError("File content cannot be empty")
        return v
    
    @model_validator(mode='after')
    def validate_extension_and_size(self) -> 'UploadKnowledgeRequest':
        """Validate file extension and size for RAG knowledge."""
        file_path = Path(self.filename)
        extension = file_path.suffix.lower()
        
        if not extension:
            raise ValueError(
                f"File must have an extension. "
                f"Allowed extensions: {', '.join(settings.rag_allowed_extensions)}"
            )
        
        if extension not in settings.rag_allowed_extensions:
            raise ValueError(
                f"File extension '{extension}' is not allowed. "
                f"Allowed extensions: {', '.join(settings.rag_allowed_extensions)}"
            )
        
        file_size = len(self.file_content)
        if file_size > settings.rag_max_file_size:
            raise ValueError(
                f"File size ({file_size / (1024 * 1024):.2f}MB) exceeds maximum "
                f"allowed size ({settings.rag_max_file_size / (1024 * 1024):.2f}MB)"
            )
        
        return self
    
    class Config:
        arbitrary_types_allowed = True


class UploadKnowledgeResponse(BaseModel):
    """Response model for knowledge upload."""
    success: bool
    message: str
    filename: str | None = None
    num_chunks: int | None = None


class KnowledgeResponse(BaseModel):
    """Response model for knowledge."""
    document_id: str
    file_name: str
    file_type: str
    file_size: int
    creation_date: str
    download_url: str


class DeleteKnowledgeResponse(BaseModel):
    """Response model for knowledge deletion."""
    success: bool
    message: str
