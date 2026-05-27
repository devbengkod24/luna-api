"""Unit tests for file lifecycle use cases."""

import pytest

from src.shared.constants import FileStatus
from src.use_cases.file_use_case import (
    DeleteFileUseCase,
    IngestFileUseCase,
    UploadFileUseCase,
)

from .fakes import FakeFileRepo, FakeRagService, FakeStorageService


@pytest.mark.asyncio
async def test_upload_file_writes_to_storage_and_persists_metadata():
    storage = FakeStorageService()
    repo = FakeFileRepo()
    use_case = UploadFileUseCase(repo, storage)

    file = await use_case.execute(file_content=b"hello world", filename="hello.txt")

    assert file.id is not None
    assert file.filename == "hello.txt"
    assert file.size_bytes == len(b"hello world")
    assert file.status == FileStatus.UPLOADED
    assert storage.uploaded[file.storage_path] == b"hello world"


@pytest.mark.asyncio
async def test_delete_file_removes_storage_rag_and_repo():
    storage = FakeStorageService()
    repo = FakeFileRepo()
    rag = FakeRagService()

    upload = UploadFileUseCase(repo, storage)
    file = await upload.execute(file_content=b"x", filename="x.txt")

    delete = DeleteFileUseCase(repo, storage, rag)
    ok = await delete.execute(file_id=file.id)

    assert ok is True
    assert file.storage_path in storage.deleted
    assert file.id in rag.deleted
    assert await repo.get_by_id(file.id) is None


@pytest.mark.asyncio
async def test_delete_file_returns_false_for_missing():
    delete = DeleteFileUseCase(FakeFileRepo(), FakeStorageService(), FakeRagService())
    assert await delete.execute(file_id="missing") is False


@pytest.mark.asyncio
async def test_ingest_file_marks_completed_on_success():
    storage = FakeStorageService()
    repo = FakeFileRepo()
    rag = FakeRagService()

    upload = UploadFileUseCase(repo, storage)
    file = await upload.execute(file_content=b"data", filename="doc.txt")

    ingest = IngestFileUseCase(repo, storage, rag)
    result = await ingest.execute(file_id=file.id, tags=["udinus"])

    assert result["indexed"] == 1
    assert result["failed"] == 0
    refreshed = await repo.get_by_id(file.id)
    assert refreshed.status == FileStatus.COMPLETED
    assert refreshed.tags == ["udinus"]
    assert refreshed.indexed_at is not None
