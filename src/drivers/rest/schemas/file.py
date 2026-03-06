"""File/knowledge schemas for RAG CRUD + ingest."""

from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

from src.config import settings

DANGEROUS_PATTERNS = [
    "..",
    "/",
    "\\",
    "\x00",
]


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues."""
    if not filename:
        return "file"

    sanitized = filename
    for pattern in DANGEROUS_PATTERNS:
        sanitized = sanitized.replace(pattern, "_")

    sanitized = sanitized.strip(". ")

    if len(sanitized) > 255:
        file_path = Path(sanitized)
        name_without_ext = file_path.stem
        extension = file_path.suffix
        max_name_length = 255 - len(extension)
        sanitized = name_without_ext[:max_name_length] + extension

    return sanitized or "file"


# ----- Internal (built from multipart in router) -----


class UploadFileRequest(BaseModel):
    """Request body shape for upload/ingest (filename + content from multipart)."""

    filename: str
    file_content: bytes

    @field_validator("filename")
    @classmethod
    def validate_and_sanitize_filename(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        sanitized = sanitize_filename(v)
        if not sanitized or sanitized == "file":
            raise ValueError("Filename becomes empty after sanitization")
        return sanitized

    @field_validator("file_content")
    @classmethod
    def validate_file_content(cls, v: bytes) -> bytes:
        if not v or len(v) == 0:
            raise ValueError("File content cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_extension_and_size(self) -> "UploadFileRequest":
        file_path = Path(self.filename)
        extension = file_path.suffix.lower()

        if not extension:
            raise ValueError(
                f"File must have an extension. "
                f"Allowed: {', '.join(settings.rag_allowed_extensions)}"
            )
        if extension not in settings.rag_allowed_extensions:
            raise ValueError(
                f"File extension '{extension}' is not allowed. "
                f"Allowed: {', '.join(settings.rag_allowed_extensions)}"
            )
        file_size = len(self.file_content)
        if file_size > settings.rag_max_file_size:
            raise ValueError(
                f"File size ({file_size / (1024 * 1024):.2f}MB) exceeds maximum "
                f"({settings.rag_max_file_size / (1024 * 1024):.2f}MB)"
            )
        return self

    model_config = {"arbitrary_types_allowed": True}


# ----- Responses -----


class FileUploadResponse(BaseModel):
    """Response for POST /files/upload."""

    success: bool
    message: str
    filename: str | None = None
    num_chunks: int | None = None


class FileListItem(BaseModel):
    """Single item in GET /files/list."""

    document_id: str
    file_name: str
    file_type: str
    file_size: int
    creation_date: str
    download_url: str


class FileDeleteResponse(BaseModel):
    """Response for DELETE /files/{document_id}."""

    success: bool
    message: str


class FileIngestResponse(BaseModel):
    """Response for POST /files/ingest (store to vector DB)."""

    success: bool
    message: str
    filename: str | None = None
    num_chunks: int | None = None
    status: str = "indexed"
