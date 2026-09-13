"""Photobooth photo REST router: save, list, view/download, delete."""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, Response, UploadFile

from src.config import settings
from src.drivers import dependencies
from src.drivers.rest.schemas.photo_schemas import PhotoResponse, validate_photo_content
from src.use_cases.photo_use_case import (
    DeletePhotoUseCase,
    GetPhotoUseCase,
    ListPhotosUseCase,
    SavePhotoUseCase,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/photo", tags=["photo"])

# The shape `StdUuidGenerator` produces, and nothing else. The id becomes part
# of a storage key, so it is held to exactly that shape at the boundary.
PHOTO_ID = Path(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _download_name(taken: datetime) -> str:
    """The kiosk's own wall clock, not UTC.

    Storage stamps objects in UTC, and a file saved at 09:26 in Semarang came
    down named 02:26 — which reads as the middle of the night to whoever
    downloads it.
    """
    try:
        local = taken.astimezone(ZoneInfo(settings.app_timezone))
    except (ZoneInfoNotFoundError, ValueError):
        local = taken.astimezone(timezone.utc)
    return f"luna-{local:%Y%m%d-%H%M%S}.jpg"


def _to_response(photo) -> PhotoResponse:
    return PhotoResponse(
        id=photo.id,
        size_bytes=photo.size_bytes,
        created_at=photo.created_at.isoformat(),
    )


@router.put("/{photo_id}", response_model=PhotoResponse, status_code=201)
async def save_photo(
    photo_id: str = PHOTO_ID,
    file: UploadFile = File(...),
    use_case: SavePhotoUseCase = Depends(dependencies.get_save_photo_use_case),
):
    """Store a composed photobooth frame under the kiosk's id.

    PUT, so sending the same photo twice is harmless. The kiosk retries until
    it hears success, and it cannot tell a lost request from a lost reply.
    """
    try:
        # One byte past the limit is enough to know it is over, and stops an
        # oversized body being buffered whole just to be rejected.
        content = await file.read(settings.photo_max_image_size + 1)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading photo") from e

    try:
        content = validate_photo_content(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return _to_response(await use_case.execute(photo_id, content))


@router.get("", response_model=list[PhotoResponse])
async def list_photos(
    use_case: ListPhotosUseCase = Depends(dependencies.get_list_photos_use_case),
):
    """All photos, newest first."""
    return [_to_response(p) for p in await use_case.execute()]


@router.get("/{photo_id}")
async def get_photo(
    photo_id: str = PHOTO_ID,
    download: bool = Query(False),
    use_case: GetPhotoUseCase = Depends(dependencies.get_get_photo_use_case),
):
    """The image. `?download=true` asks the browser to save it instead.

    The download is decided here, by header, rather than by an `<a download>`
    in the dashboard: browsers ignore that attribute on a cross-origin link and
    simply open the image, and the dashboard is served from a different origin
    than this API.
    """
    found = await use_case.execute(photo_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Photo not found")
    photo, content = found
    name = _download_name(photo.created_at)
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'{"attachment" if download else "inline"}; filename="{name}"',
            # An id always names the same bytes, so a thumbnail grid need not
            # refetch every image each time the list refreshes.
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.delete("/{photo_id}", status_code=204)
async def delete_photo(
    photo_id: str = PHOTO_ID,
    use_case: DeletePhotoUseCase = Depends(dependencies.get_delete_photo_use_case),
):
    """Delete one photo."""
    if not await use_case.execute(photo_id):
        raise HTTPException(status_code=404, detail="Photo not found")
