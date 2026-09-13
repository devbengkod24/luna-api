"""Unit tests for file lifecycle use cases."""

import pytest

from src.domain.entities.file import File
from src.shared.constants import FileStatus
from src.use_cases.file_use_case import (
    DOCUMENT_PREFIX,
    DeleteFileUseCase,
    IngestFileUseCase,
    UploadFileUseCase,
)

from .fakes import FakeFileRepo, FakeRagService, FakeStorageService, FakeUuidGenerator


def uploader(repo, storage):
    return UploadFileUseCase(repo, storage, FakeUuidGenerator())


@pytest.mark.asyncio
async def test_upload_file_writes_to_storage_and_persists_metadata():
    storage = FakeStorageService()
    repo = FakeFileRepo()
    use_case = uploader(repo, storage)

    file = await use_case.execute(file_content=b"hello world", filename="hello.txt")

    assert file.id is not None
    assert file.filename == "hello.txt"
    assert file.size_bytes == len(b"hello world")
    assert file.status == FileStatus.UPLOADED
    assert storage.uploaded[file.storage_path] == b"hello world"
    assert file.storage_path.startswith(DOCUMENT_PREFIX)
    assert file.storage_path.endswith("/hello.txt")


@pytest.mark.asyncio
async def test_delete_file_removes_storage_rag_and_repo():
    storage = FakeStorageService()
    repo = FakeFileRepo()
    rag = FakeRagService()

    upload = uploader(repo, storage)
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

    upload = uploader(repo, storage)
    file = await upload.execute(file_content=b"data", filename="doc.txt")

    ingest = IngestFileUseCase(repo, storage, rag)
    result = await ingest.execute(file_id=file.id, tags=["udinus"])

    assert result["indexed"] == 1
    assert result["failed"] == 0
    refreshed = await repo.get_by_id(file.id)
    assert refreshed.status == FileStatus.COMPLETED
    assert refreshed.tags == ["udinus"]
    assert refreshed.indexed_at is not None


@pytest.mark.asyncio
async def test_documents_with_the_same_name_keep_their_own_content():
    """The seeder's everyday case: `admisi/biaya.pdf` and `akademik/biaya.pdf`."""
    storage, repo = FakeStorageService(), FakeFileRepo()
    upload = uploader(repo, storage)
    first = await upload.execute(file_content=b"biaya FIK", filename="biaya.pdf")
    second = await upload.execute(file_content=b"biaya FEB", filename="biaya.pdf")

    assert first.storage_path != second.storage_path
    assert storage.uploaded[first.storage_path] == b"biaya FIK"
    assert storage.uploaded[second.storage_path] == b"biaya FEB"


@pytest.mark.asyncio
async def test_deleting_one_same_named_document_leaves_the_other():
    storage, repo, rag = FakeStorageService(), FakeFileRepo(), FakeRagService()
    upload = uploader(repo, storage)
    kept = await upload.execute(file_content=b"biaya FIK", filename="biaya.pdf")
    gone = await upload.execute(file_content=b"biaya FEB", filename="biaya.pdf")

    assert await DeleteFileUseCase(repo, storage, rag).execute(gone.id) is True

    assert storage.uploaded.get(kept.storage_path) == b"biaya FIK"
    assert gone.storage_path not in storage.uploaded


@pytest.mark.asyncio
async def test_a_legacy_key_shared_by_two_rows_survives_until_the_last_one_goes():
    """Rows written before keys were unique. Deleting one must not empty the
    other; deleting the last one must still clean up."""
    storage, repo, rag = FakeStorageService(), FakeFileRepo(), FakeRagService()
    await storage.upload_file(b"isi terakhir", "biaya.pdf", "application/pdf")
    rows = [
        await repo.create(File(filename="biaya.pdf", size_bytes=12, content_type="application/pdf",
                               storage_path="biaya.pdf", status=FileStatus.COMPLETED))
        for _ in range(2)
    ]
    delete = DeleteFileUseCase(repo, storage, rag)

    await delete.execute(rows[0].id)
    assert storage.uploaded.get("biaya.pdf") == b"isi terakhir"
    assert rows[0].id in rag.deleted

    await delete.execute(rows[1].id)
    assert "biaya.pdf" not in storage.uploaded

