import asyncio
from functools import cached_property
from io import BytesIO
from pathlib import Path

import boto3
from botocore.config import Config

from app.core.config import settings


class StorageService:
    """Stores uploaded product files in Cloudflare R2, with a local disk fallback.

    The local fallback keeps the processing pipeline usable when R2 credentials
    are not configured (development and tests).
    """

    @cached_property
    def r2_configured(self) -> bool:
        return all(
            [
                settings.r2_account_id,
                settings.r2_access_key_id,
                settings.r2_secret_access_key,
                settings.r2_bucket_name,
            ]
        )

    @cached_property
    def client(self):  # type: ignore[no-untyped-def]
        if not self.r2_configured:
            raise RuntimeError("Cloudflare R2 credentials are not configured")
        return boto3.client(
            "s3",
            endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    async def save(self, object_key: str, data: bytes, content_type: str) -> tuple[str, str]:
        """Persist the file and return the backend name and its URL."""
        if not self.r2_configured:
            return "local", self._save_local(object_key, data)
        await asyncio.to_thread(
            self.client.upload_fileobj,
            BytesIO(data),
            settings.r2_bucket_name,
            object_key,
            ExtraArgs={"ContentType": content_type},
        )
        return "r2", self._r2_url(object_key)

    async def delete(self, object_key: str) -> None:
        if not self.r2_configured:
            Path(settings.local_storage_dir, object_key).unlink(missing_ok=True)
            return
        await asyncio.to_thread(
            self.client.delete_object, Bucket=settings.r2_bucket_name, Key=object_key
        )

    def _save_local(self, object_key: str, data: bytes) -> str:
        path = Path(settings.local_storage_dir, object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path.resolve())

    def _r2_url(self, object_key: str) -> str:
        if settings.r2_public_base_url:
            return f"{settings.r2_public_base_url.rstrip('/')}/{object_key}"
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.r2_bucket_name, "Key": object_key},
            ExpiresIn=settings.r2_presigned_url_expire_seconds,
        )


storage_service = StorageService()
