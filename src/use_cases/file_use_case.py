"""File use cases — upload to MinIO, CRUD via DB, and ingest to vector store."""

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.domain.entities.file import File
from src.ports.identifiers import IUuidGenerator
from src.ports.repositories import IFileRepository
from src.ports.services import IStorageService, IRAGService
from src.shared.constants import FileStatus

logger = logging.getLogger(__name__)


def _detect_content_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".md": "text/markdown",
    }.get(ext, "application/octet-stream")


DOCUMENT_PREFIX = "documents/"


def document_path(key_id: str, filename: str) -> str:
    """Where one uploaded document lives in storage.

    Unique per upload, not per name. The key used to be the filename alone, so
    a second "biaya.pdf" overwrote the first one's bytes while both rows still
    pointed at it, and deleting either took the other's content too. That was
    easy to reach without trying: the seeder takes documents from one folder
    per tag, and the same filename under two tags is ordinary.

    The filename stays at the end so the object is still recognisable when
    browsing the bucket. The database row, not this key, is what records it.
    """
    return f"{DOCUMENT_PREFIX}{key_id}/{filename}"


class UploadFileUseCase:
    """Upload file to MinIO and record metadata in DB."""

    def __init__(
        self,
        file_repo: IFileRepository,
        storage: IStorageService,
        uuid_generator: IUuidGenerator,
    ):
        self.file_repo = file_repo
        self.storage = storage
        self.uuid_generator = uuid_generator

    async def execute(self, file_content: bytes, filename: str) -> File:
        content_type = _detect_content_type(filename)

        storage_path = await self.storage.upload_file(
            file_content=file_content,
            filename=document_path(self.uuid_generator.new_id(), filename),
            content_type=content_type,
        )

        file = await self.file_repo.create(
            File(
                filename=filename,
                size_bytes=len(file_content),
                content_type=content_type,
                storage_path=storage_path,
                status=FileStatus.UPLOADED,
            )
        )

        logger.info("Uploaded file: %s (id=%s, %d bytes)", filename, file.id, len(file_content))
        return file


class ListFilesUseCase:
    """List all files from DB."""

    def __init__(self, file_repo: IFileRepository):
        self.file_repo = file_repo

    async def execute(self) -> list[File]:
        return await self.file_repo.list()


class DeleteFileUseCase:
    """Delete file from MinIO, vector store, and DB."""

    def __init__(
        self,
        file_repo: IFileRepository,
        storage: IStorageService,
        rag_service: IRAGService,
    ):
        self.file_repo = file_repo
        self.storage = storage
        self.rag_service = rag_service

    async def execute(self, file_id: str) -> bool:
        file = await self.file_repo.get_by_id(file_id)
        if not file:
            return False

        # New uploads never share a storage key. Rows written before keys were
        # unique can — two documents uploaded under one name — and for those,
        # deleting the object would empty the survivor too. The survivor's
        # bytes are already the later upload's; keeping the object is the
        # only thing left that can be saved.
        sharing = [
            other for other in await self.file_repo.list()
            if other.id != file.id and other.storage_path == file.storage_path
        ]
        if sharing:
            logger.warning(
                "Kept storage object %s: %d other document(s) still point at it "
                "(uploaded under the same name before storage keys were unique)",
                file.storage_path, len(sharing),
            )
        else:
            await self.storage.delete_file(file.storage_path)
        await self.rag_service.delete_by_file_id(file_id)
        return await self.file_repo.delete(file_id)


class IngestFileUseCase:
    """Vectorize file(s): download from MinIO, index into Qdrant, update DB status."""

    def __init__(
        self,
        file_repo: IFileRepository,
        storage: IStorageService,
        rag_service: IRAGService,
    ):
        self.file_repo = file_repo
        self.storage = storage
        self.rag_service = rag_service

    async def execute(
        self,
        file_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        if file_id:
            files = [await self.file_repo.get_by_id(file_id)]
            if not files[0]:
                return {"success": False, "message": f"File {file_id} not found", "indexed": 0, "failed": 0}
        else:
            all_files = await self.file_repo.list()
            files = [f for f in all_files if f.status != FileStatus.INDEXING]

        indexed, failed = 0, 0

        for file in files:
            try:
                file.status = FileStatus.INDEXING
                await self.file_repo.update(file)

                file_bytes = await self.storage.download_file(file.storage_path)

                with tempfile.NamedTemporaryFile(
                    suffix=Path(file.filename).suffix, delete=False
                ) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                try:
                    effective_tags = tags if tags is not None else (file.tags or [])
                    ok = await self.rag_service.index_document(
                        file_id=file.id,
                        file_path=tmp_path,
                        tags=effective_tags or None,
                    )
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

                file.status = FileStatus.COMPLETED if ok else FileStatus.FAILED
                file.tags = effective_tags or file.tags
                file.indexed_at = datetime.now(timezone.utc) if ok else file.indexed_at
                await self.file_repo.update(file)

                if ok:
                    indexed += 1
                    logger.info("Indexed file: %s (id=%s)", file.filename, file.id)
                else:
                    failed += 1
                    logger.warning("Index returned False for file: %s", file.filename)

            except Exception as e:
                failed += 1
                logger.error("Error indexing file %s: %s", file.filename, e, exc_info=True)
                try:
                    file.status = FileStatus.FAILED
                    await self.file_repo.update(file)
                except Exception as update_err:
                    logger.warning("Failed to update file status to FAILED: %s", update_err)

        return {
            "success": failed == 0,
            "message": f"Indexed {indexed} file(s), {failed} failed.",
            "indexed": indexed,
            "failed": failed,
        }
