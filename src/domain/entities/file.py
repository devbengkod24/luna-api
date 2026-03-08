"""File domain entity."""
from dataclasses import dataclass
from datetime import datetime

from src.shared.constants import FileStatus


@dataclass
class File:
    """File domain entity."""
    filename: str
    size_bytes: int
    content_type: str
    storage_path: str
    status: FileStatus
    tags: list[str] | None = None
    indexed_at: datetime | None = None
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
