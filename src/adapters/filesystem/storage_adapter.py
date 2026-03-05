"""General file storage adapter - simple file system operations only."""

import logging
from pathlib import Path
from typing import List

import aiofiles
import aiofiles.os

from src.config import settings
from src.domain.entities.storage import StorageType
from src.ports.services import IFileStorage

logger = logging.getLogger(__name__)


class FileStorageAdapter(IFileStorage):
    """Simple file storage adapter - only file system operations, no business logic."""

    def __init__(self, storage_type: StorageType):
        """Initialize file storage adapter.

        Args:
            storage_type: Type of storage (RAG or FACES)
            base_path: Optional custom base path (overrides settings)

        Raises:
            ValueError: If storage type is unknown
        """
        self.storage_type = storage_type

        # Use settings-based paths
        if storage_type == StorageType.RAG:
            self.storage_path = Path(settings.rag_dir)
        elif storage_type == StorageType.FACES:
            self.storage_path = Path(settings.faces_dir)
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

        # Create directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized {storage_type.value} storage at: {self.storage_path}")

    @property
    def base_path(self) -> Path:
        """Get base storage path."""
        return self.storage_path

    async def save_file(self, file_content: bytes, filename: str) -> Path:
        """Save file to storage.

        Args:
            file_content: File content as bytes
            file_path: Full path where file should be saved

        Returns:
            Path: Path where file was saved
        """
        # Ensure parent directory exists
        file_path = self.storage_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)

        logger.debug(f"Saved file: {file_path} ({len(file_content)} bytes)")
        return file_path

    async def delete_file(self, filename: str) -> bool:
        """Delete file from storage.

        Args:
            filename: Filename to delete

        Returns:
            bool: True if deleted successfully
        """
        try:
            file_path = self.storage_path / filename
            if await aiofiles.os.path.exists(file_path):
                await aiofiles.os.remove(file_path)
                logger.debug(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    async def file_exists(self, filename: str) -> bool:
        """Check if file exists.

        Args:
            filename: Filename to check

        Returns:
            bool: True if file exists
        """
        file_path = self.storage_path / filename
        return await aiofiles.os.path.exists(file_path)

    async def list_files(self) -> List[str]:
        """List all files in storage.

        Returns:
            List[str]: List of filenames
        """
        files: List[str] = []
        if not self.storage_path.exists():
            return files

        # Walk recursively and return relative paths so nested folders are
        # represented (e.g. "BIMA/kurikulum/file.txt"). Use forward slashes
        # for consistency across platforms.
        for item in self.storage_path.rglob("*"):
            try:
                if item.is_file():
                    rel = item.relative_to(self.storage_path)
                    files.append(str(rel).replace("\\", "/"))
            except Exception:
                # Fallback to name if relative path computation fails
                try:
                    if item.is_file():
                        files.append(item.name)
                except Exception:
                    continue

        return files

    async def read_file(self, filename: str) -> bytes:
        """Read file from storage.

        Args:
            filename: Filename to read

        Returns:
            bytes: File content as bytes

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = self.storage_path / filename
        if not await aiofiles.os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {filename}")

        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()

        logger.debug(f"Read file: {file_path} ({len(content)} bytes)")
        return content
