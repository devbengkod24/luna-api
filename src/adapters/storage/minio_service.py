"""MinIO storage service implementation."""
from typing import BinaryIO
import io

from minio import Minio
from minio.error import S3Error

from src.ports.services import IStorageService
from src.config import settings
from src.shared.exceptions import InternalServerException


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

    def _ensure_bucket(self):
        """Ensure bucket exists."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            raise InternalServerException(f"MinIO error: {str(e)}")

    async def upload_file(
        self, file_data: BinaryIO, filename: str, content_type: str
    ) -> str:
        """Upload file and return storage path."""
        try:
            # Read file data
            file_bytes = file_data.read()
            file_size = len(file_bytes)
            
            # Upload to MinIO
            self.client.put_object(
                self.bucket_name,
                filename,
                io.BytesIO(file_bytes),
                file_size,
                content_type=content_type,
            )
            
            return filename
        except S3Error as e:
            raise InternalServerException(f"Upload failed: {str(e)}")

    async def download_file(self, storage_path: str) -> bytes:
        """Download file from storage."""
        try:
            response = self.client.get_object(self.bucket_name, storage_path)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            raise InternalServerException(f"Download failed: {str(e)}")

    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from storage."""
        try:
            self.client.remove_object(self.bucket_name, storage_path)
            return True
        except S3Error:
            return False

    async def get_file_url(self, storage_path: str, expires: int = 3600) -> str:
        """Get pre-signed URL for file access."""
        try:
            url = self.client.presigned_get_object(
                self.bucket_name, storage_path, expires=expires
            )
            return url
        except S3Error as e:
            raise InternalServerException(f"Get URL failed: {str(e)}")

