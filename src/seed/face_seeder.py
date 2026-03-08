"""Face seeder — register faces from storage/faces into Qdrant.

Filename format:
  Nama_Nama__NIM.ext   → name="Nama Nama", nim="NIM"
  Nama_Nama.ext        → name="Nama Nama", nim=""
"""

import base64
import logging
from pathlib import Path

from src.domain.entities.face import RegisterFaceInput
from src.ports.services import IFaceService
from src.ports.vision import IFaceRecognitionService
from src.use_cases.face_use_case import RegisterFacesUseCase

logger = logging.getLogger(__name__)


def _parse_filename(stem: str) -> tuple[str, str]:
    """Parse stem into (name, nim). Uses '__' as separator."""
    if "__" in stem:
        name_part, nim = stem.split("__", 1)
    else:
        name_part, nim = stem, ""
    return name_part.replace("_", " ").strip(), nim.strip()


async def run(face_service: IFaceService, recognition: IFaceRecognitionService) -> None:
    """Seed all faces from settings.faces_dir. Overwrites existing by NIM."""
    faces_dir = Path("storage/faces")
    if not faces_dir.exists():
        logger.warning("Face seed skipped: directory not found (%s)", faces_dir)
        return

    items: list[RegisterFaceInput] = []
    for path in sorted(faces_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue

        name, nim = _parse_filename(path.stem)
        image_b64 = base64.b64encode(path.read_bytes()).decode()
        items.append(RegisterFaceInput(name=name, nim=nim, image_base64=image_b64))
        logger.debug("Queued: %s (nim=%s)", name, nim or "-")

    if not items:
        logger.info("No face images found in %s", faces_dir)
        return

    logger.info("Seeding %d face(s) from %s ...", len(items), faces_dir)
    use_case = RegisterFacesUseCase(recognition=recognition, face_service=face_service)
    results = await use_case.execute(faces=items, skip_existing=False)

    success = sum(1 for r in results if r.success)
    failed = [r for r in results if not r.success]
    logger.info("Face seed done: %d/%d registered", success, len(results))
    for r in failed:
        logger.warning("  Failed [%s / %s]: %s", r.name, r.nim, r.message)
