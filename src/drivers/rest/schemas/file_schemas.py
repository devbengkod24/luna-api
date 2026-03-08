"""File schemas for CRUD + ingest endpoints."""
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import settings
from src.shared.constants import FileStatus

DANGEROUS_PATTERNS = ["..", "/", "\\", "\x00"]


def sanitize_filename(filename: str) -> str:
    sanitized = filename
    for pattern in DANGEROUS_PATTERNS:
        sanitized = sanitized.replace(pattern, "_")
    sanitized = sanitized.strip(". ")
    if len(sanitized) > 255:
        p = Path(sanitized)
        sanitized = p.stem[: 255 - len(p.suffix)] + p.suffix
    return sanitized or "file"


class UploadFileRequest(BaseModel):
    """Built from multipart upload in the router."""
    filename: str
    file_content: bytes

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        return sanitize_filename(v)

    @field_validator("file_content")
    @classmethod
    def validate_content(cls, v: bytes) -> bytes:
        if not v:
            raise ValueError("File content cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_extension_and_size(self) -> "UploadFileRequest":
        ext = Path(self.filename).suffix.lower()
        if not ext:
            raise ValueError(f"File must have an extension. Allowed: {', '.join(settings.rag_allowed_extensions)}")
        if ext not in settings.rag_allowed_extensions:
            raise ValueError(f"Extension '{ext}' not allowed. Allowed: {', '.join(settings.rag_allowed_extensions)}")
        size = len(self.file_content)
        if size > settings.rag_max_file_size:
            raise ValueError(f"File too large ({size / 1024 / 1024:.1f}MB). Max: {settings.rag_max_file_size / 1024 / 1024:.0f}MB")
        return self

    model_config = {"arbitrary_types_allowed": True}


class FileResponse(BaseModel):
    """Single file item."""
    id: str
    filename: str
    size_bytes: int
    content_type: str
    storage_path: str
    status: FileStatus
    tags: list[str]
    indexed_at: str | None
    created_at: str
    updated_at: str


class IngestRequest(BaseModel):
    """Request body for POST /file/ingest."""
    file_id: Optional[str] = None
    tags: Optional[list[str]] = Field(default=None, description="Tags for filtering context (e.g. admisi, akademik)")


class IngestResponse(BaseModel):
    """Response for POST /file/ingest."""
    success: bool
    message: str
    indexed: int
    failed: int
