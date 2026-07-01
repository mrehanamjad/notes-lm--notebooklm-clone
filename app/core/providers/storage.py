"""Storage provider abstraction with ImageKit implementation."""

from abc import ABC, abstractmethod
from typing import Optional
import os
import shutil

from app.core.config import settings
from app.core.logger import logger

from imagekitio import ImageKit


class BaseStorageProvider(ABC):
    @abstractmethod
    def upload_file(self, file_path: str, file_name: str, folder: str = "/") -> dict:
        """Upload file, return dict with url and file_id."""
        ...

    @abstractmethod
    def delete_file(self, file_id: str) -> bool:
        """Delete a single file by its storage provider file ID."""
        ...

    @abstractmethod
    def delete_many_files(self, file_ids: list[str]) -> bool:
        """Delete multiple files in a single bulk operation."""
        ...


class ImageKitProvider(BaseStorageProvider):
    def __init__(self):
        self.client = ImageKit(
            private_key=settings.IMAGEKIT_PRIVATE_KEY,
            # public_key=settings.IMAGEKIT_PUBLIC_KEY,
            # url_endpoint=settings.IMAGEKIT_URL_ENDPOINT,
        )

    def upload_file(self, file_path: str, file_name: str, folder: str = "/") -> dict:
        try:
            with open(file_path, "rb") as f:
                response = self.client.files.upload(
                    file=f,
                    file_name=file_name,
                    folder=folder,
                    use_unique_file_name=True,
                )
            logger.info(f"ImageKit upload successful: {response.file_id}")
            return {
                "url": response.url,
                "file_id": response.file_id,
            }
        except Exception as e:
            logger.error(f"ImageKit upload failed: {e}")
            raise

    def delete_file(self, file_id: str) -> bool:
        try:
            self.client.files.delete(file_id=file_id)
            logger.info(f"ImageKit delete successful: {file_id}")
            return True
        except Exception as e:
            logger.error(f"ImageKit delete failed (file_id={file_id}): {e}")
            return False

    def delete_many_files(self, file_ids: list[str]) -> bool:
        try:
            response = self.client.files.bulk.delete(file_ids=file_ids)
            logger.info(f"ImageKit bulk delete successful ({len(file_ids)} files): {response}")
            return True
        except Exception as e:
            logger.error(f"ImageKit bulk delete failed ({len(file_ids)} files): {e}")
            return False


class LocalStorageProvider(BaseStorageProvider):
    def upload_file(self, file_path: str, file_name: str, folder: str = "/") -> dict:
        dest_dir = os.path.join("uploads", folder.strip("/"))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, file_name)
        shutil.copy2(file_path, dest)
        return {"url": dest, "file_id": dest, "file_name": file_name}

    def delete_file(self, file_id: str) -> bool:
        try:
            if os.path.exists(file_id):
                os.remove(file_id)
                logger.info(f"Local delete successful: {file_id}")
                return True
            logger.warning(f"Local delete skipped — file not found: {file_id}")
            return False
        except Exception as e:
            logger.error(f"Local delete failed (file_id={file_id}): {e}")
            return False

    def delete_many_files(self, file_ids: list[str]) -> bool:
        """Delete each file individually; local storage has no native bulk API."""
        all_ok = True
        for file_id in file_ids:
            if not self.delete_file(file_id):
                all_ok = False
        return all_ok


_PROVIDERS = {
    "imagekit": ImageKitProvider,
    "local": LocalStorageProvider,
}


def get_storage_provider(provider_name: Optional[str] = None) -> BaseStorageProvider:
    name = (provider_name or settings.STORAGE_PROVIDER).lower()
    provider_cls = _PROVIDERS.get(name)
    if not provider_cls:
        raise ValueError(f"Unknown storage provider: {name}")
    return provider_cls()