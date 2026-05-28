"""MinIO storage service implementation."""
import asyncio
import io
import logging

from minio import Minio
from minio.error import S3Error

from src.ports.services import IStorageService
from src.config import settings
from src.shared.exceptions import InternalServerException

logger = logging.getLogger(__name__)


class MinIOStorageService(IStorageService):
    """MinIO storage service implementation."""

    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket_name = settings.minio_bucket_name
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            logger.error("MinIO bucket initialization failed: %s", e)
            raise InternalServerException("Storage service initialization failed")

    async def upload_file(self, file_content: bytes, filename: str, content_type: str) -> str:
        try:
            await asyncio.to_thread(
                self.client.put_object,
                self.bucket_name,
                filename,
                io.BytesIO(file_content),
                len(file_content),
                content_type=content_type,
            )
            return filename
        except S3Error as e:
            logger.error("MinIO upload failed for %s: %s", filename, e)
            raise InternalServerException("File upload failed")

    async def download_file(self, storage_path: str) -> bytes:
        def _read() -> bytes:
            response = self.client.get_object(self.bucket_name, storage_path)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        try:
            return await asyncio.to_thread(_read)
        except S3Error as e:
            logger.error("MinIO download failed for %s: %s", storage_path, e)
            raise InternalServerException("File download failed")

    async def delete_file(self, storage_path: str) -> bool:
        try:
            await asyncio.to_thread(self.client.remove_object, self.bucket_name, storage_path)
            return True
        except S3Error:
            return False
