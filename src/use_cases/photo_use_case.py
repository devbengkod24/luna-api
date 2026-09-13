"""Photobooth photos: save, list, fetch, delete.

Stored under their own prefix in the same bucket as RAG documents. Documents
are keyed by their original filename at the bucket root, so a prefix is what
keeps the two from ever colliding — and what lets a listing return photos
without walking every document.
"""

import logging

from src.domain.entities import Photo, StoredObject
from src.ports.services import IStorageService
from src.shared.exceptions import InternalServerException

logger = logging.getLogger(__name__)

PHOTO_PREFIX = "photos/"
PHOTO_SUFFIX = ".jpg"
PHOTO_CONTENT_TYPE = "image/jpeg"


def photo_path(photo_id: str) -> str:
    return f"{PHOTO_PREFIX}{photo_id}{PHOTO_SUFFIX}"


def _to_photo(obj: StoredObject) -> Photo | None:
    """A stored object as a photo, or None if it is not one of ours.

    Anything else under the prefix — a stray upload, a nested key — is skipped
    rather than surfaced with an id the other endpoints cannot resolve.
    """
    name = obj.path.removeprefix(PHOTO_PREFIX)
    if name == obj.path or "/" in name or not name.endswith(PHOTO_SUFFIX):
        return None
    return Photo(
        id=name.removesuffix(PHOTO_SUFFIX),
        size_bytes=obj.size_bytes,
        created_at=obj.modified_at,
    )


class SavePhotoUseCase:
    """Store one composed photobooth frame, under the id the kiosk chose.

    The kiosk names the photo, not this service, because the kiosk is the one
    that retries. It keeps every photo on its own disk until an upload
    succeeds, and an upload that landed but lost its reply is indistinguishable
    from one that never arrived. With the id fixed on the kiosk, the retry
    overwrites the same object; minted here, it would store a second copy.
    """

    def __init__(self, storage: IStorageService):
        self.storage = storage

    async def execute(self, photo_id: str, content: bytes) -> Photo:
        path = photo_path(photo_id)
        await self.storage.upload_file(
            file_content=content, filename=path, content_type=PHOTO_CONTENT_TYPE
        )
        # Read back rather than stamped locally, so the time a photo reports
        # here is the same one every later listing will report.
        stored = await self.storage.stat_file(path)
        if stored is None:
            raise InternalServerException("Photo was not stored")
        logger.info("Saved photo %s (%d bytes)", photo_id, len(content))
        return _to_photo(stored)


class ListPhotosUseCase:
    """Every stored photo, newest first."""

    def __init__(self, storage: IStorageService):
        self.storage = storage

    async def execute(self) -> list[Photo]:
        objects = await self.storage.list_files(PHOTO_PREFIX)
        photos = [p for p in (_to_photo(o) for o in objects) if p is not None]
        return sorted(photos, key=lambda p: p.created_at, reverse=True)


class GetPhotoUseCase:
    """One photo and its bytes, or None if it does not exist."""

    def __init__(self, storage: IStorageService):
        self.storage = storage

    async def execute(self, photo_id: str) -> tuple[Photo, bytes] | None:
        path = photo_path(photo_id)
        stored = await self.storage.stat_file(path)
        if stored is None:
            return None
        return _to_photo(stored), await self.storage.download_file(path)


class DeletePhotoUseCase:
    """Delete one photo. False if there was nothing to delete."""

    def __init__(self, storage: IStorageService):
        self.storage = storage

    async def execute(self, photo_id: str) -> bool:
        path = photo_path(photo_id)
        # Checked first because deleting a missing object succeeds in S3.
        # Without this a wrong id answered 204 and looked like it had worked.
        if await self.storage.stat_file(path) is None:
            return False
        deleted = await self.storage.delete_file(path)
        if deleted:
            logger.info("Deleted photo %s", photo_id)
        return deleted
