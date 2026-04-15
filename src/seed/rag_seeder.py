"""RAG seeder — upload + vectorize documents from storage/rag/ into Qdrant.

Folder structure:
    storage/rag/<tag>/file.pdf            → tags=[tag]
    storage/rag/<tag>/nested/file.pdf     → tags=[tag]  (always top-level subfolder)
    storage/rag/file.pdf                  → tags=["default"] (no subfolder)

If a file with the same filename already exists, it is deleted first (re-index).
"""

import logging
from pathlib import Path

from src.adapters.persistence.database import DatabaseConnection
from src.adapters.persistence.repositories import FileRepository
from src.config import settings
from src.ports.services import IRAGService, IStorageService
from src.use_cases.file_use_case import (
    DeleteFileUseCase,
    IngestFileUseCase,
    ListFilesUseCase,
    UploadFileUseCase,
)

logger = logging.getLogger(__name__)


async def run(
    db: DatabaseConnection,
    storage: IStorageService,
    rag: IRAGService,
) -> None:
    """Seed and vectorize all documents from settings.rag_dir subfolders."""
    rag_dir = Path("storage/rag")
    if not rag_dir.exists():
        logger.warning("RAG seed skipped: directory not found (%s)", rag_dir)
        return

    async with db.async_session_maker() as session:
        file_repo = FileRepository(session)
        list_uc = ListFilesUseCase(file_repo)
        upload_uc = UploadFileUseCase(file_repo, storage)
        delete_uc = DeleteFileUseCase(file_repo, storage, rag)
        ingest_uc = IngestFileUseCase(file_repo, storage, rag)

        existing = {f.filename: f.id for f in await list_uc.execute()}

        indexed, failed = 0, 0

        for path in sorted(rag_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in settings.rag_allowed_extensions:
                continue

            # Determine tag from subfolder or use "default" if directly in rag_dir
            if path.parent == rag_dir:
                tag = "default"
            else:
                # Find the top-level subfolder inside rag_dir
                rel_path = path.relative_to(rag_dir)
                tag = rel_path.parts[0]

            # Delete old entry to re-index
            if path.name in existing:
                await delete_uc.execute(existing[path.name])
                logger.debug("Removed old entry: %s", path.name)

            file = await upload_uc.execute(
                file_content=path.read_bytes(),
                filename=path.name,
            )

            result = await ingest_uc.execute(file_id=file.id, tags=[tag])
            if result["indexed"]:
                indexed += 1
                logger.debug("Indexed [%s]: %s", tag, path.name)
            else:
                failed += 1
                logger.warning("Failed [%s]: %s", tag, path.name)

        await session.commit()

    logger.info("RAG seed done: %d indexed, %d failed", indexed, failed)
