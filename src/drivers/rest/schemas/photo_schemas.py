"""Photobooth photo schemas."""
from pydantic import BaseModel

from src.config import settings

# Every JPEG begins with these three bytes. Checked on the content itself
# because the multipart content type is whatever the client chose to claim.
JPEG_MAGIC = b"\xff\xd8\xff"


def validate_photo_content(content: bytes) -> bytes:
    if not content:
        raise ValueError("Photo cannot be empty")
    if len(content) > settings.photo_max_image_size:
        raise ValueError(
            f"Photo too large. Max: {settings.photo_max_image_size / 1024 / 1024:.0f}MB"
        )
    if not content.startswith(JPEG_MAGIC):
        raise ValueError("Photo must be a JPEG image")
    return content


class PhotoResponse(BaseModel):
    """Single photo item. The image itself is fetched from /photo/{id}."""
    id: str
    size_bytes: int
    created_at: str
