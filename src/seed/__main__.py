"""Standalone seeder CLI.

Usage:
    python -m src.seed face    # seed faces from storage/faces/
    python -m src.seed rag     # upload + vectorize docs from storage/rag/
    python -m src.seed all     # seed both
"""

import argparse
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("seed")


async def _run_face() -> None:
    from src.adapters import FaceStore, FaceRecognitionAdapter
    from src.seed import face_seeder

    face_service = FaceStore()
    await face_service.initialize()
    recognition = FaceRecognitionAdapter()
    await face_seeder.run(face_service, recognition)


async def _run_rag() -> None:
    from src.adapters import MinIOStorageService, LlamaIndexRAGService
    from src.adapters.persistence.database import DatabaseConnection
    from src.config import settings
    from src.seed import rag_seeder

    db = DatabaseConnection(url=settings.database_url)
    storage = MinIOStorageService()
    rag = LlamaIndexRAGService()
    await rag.initialize()

    try:
        await rag_seeder.run(db, storage, rag)
    finally:
        await db.close()


async def _main(target: str) -> None:
    if target in ("face", "all"):
        logger.info("=== Face seeder ===")
        await _run_face()
    if target in ("rag", "all"):
        logger.info("=== RAG seeder ===")
        await _run_rag()


def main() -> None:
    parser = argparse.ArgumentParser(description="VSA data seeders")
    parser.add_argument(
        "target",
        choices=["face", "rag", "all"],
        help="Which seeder to run",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.target))


if __name__ == "__main__":
    main()
