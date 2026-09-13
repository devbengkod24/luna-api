"""Unit tests for photobooth photo use cases."""

import pytest

from src.use_cases.photo_use_case import (
    DeletePhotoUseCase,
    GetPhotoUseCase,
    ListPhotosUseCase,
    SavePhotoUseCase,
    photo_path,
)

from .fakes import FakeStorageService

JPEG = b"\xff\xd8\xff\xe0 photobooth frame"
ID_A = "11111111-1111-4111-8111-111111111111"
ID_B = "22222222-2222-4222-8222-222222222222"


@pytest.mark.asyncio
async def test_save_stores_under_the_photo_prefix():
    storage = FakeStorageService()
    photo = await SavePhotoUseCase(storage).execute(ID_A, JPEG)

    assert photo.id == ID_A
    assert photo.size_bytes == len(JPEG)
    assert storage.uploaded[photo_path(ID_A)] == JPEG


@pytest.mark.asyncio
async def test_list_returns_only_photos_newest_first():
    storage = FakeStorageService()
    save = SavePhotoUseCase(storage)
    older = await save.execute(ID_A, JPEG)
    newer = await save.execute(ID_B, JPEG)
    # A RAG document lives in the same bucket, keyed at the root. It must never
    # appear as a photo.
    await storage.upload_file(b"%PDF", "handbook.pdf", "application/pdf")

    photos = await ListPhotosUseCase(storage).execute()

    assert [p.id for p in photos] == [newer.id, older.id]


@pytest.mark.asyncio
async def test_get_returns_bytes_or_none():
    storage = FakeStorageService()
    await SavePhotoUseCase(storage).execute(ID_A, JPEG)
    get = GetPhotoUseCase(storage)

    photo, content = await get.execute(ID_A)
    assert photo.id == ID_A and content == JPEG
    assert await get.execute(ID_B) is None


@pytest.mark.asyncio
async def test_delete_reports_a_missing_photo_as_not_deleted():
    storage = FakeStorageService()
    await SavePhotoUseCase(storage).execute(ID_A, JPEG)
    delete = DeletePhotoUseCase(storage)

    assert await delete.execute(ID_B) is False
    assert photo_path(ID_B) not in storage.deleted     # never even attempted
    assert await delete.execute(ID_A) is True
    assert await ListPhotosUseCase(storage).execute() == []


@pytest.mark.asyncio
async def test_saving_the_same_id_twice_keeps_one_photo():
    """The kiosk's retry after a lost reply. It must overwrite, not duplicate."""
    storage = FakeStorageService()
    save = SavePhotoUseCase(storage)
    await save.execute(ID_A, JPEG)
    await save.execute(ID_A, JPEG)

    assert [p.id for p in await ListPhotosUseCase(storage).execute()] == [ID_A]

