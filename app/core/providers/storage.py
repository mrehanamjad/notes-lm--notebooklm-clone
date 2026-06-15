"""Storage provider abstraction layer.

Supports: ImageKit. Add new providers by subclassing BaseStorageProvider.
"""

from abc import ABC, abstractmethod
from typing import Optional
from app.core.config import settings
from app.core.logger import logger


class BaseStorageProvider(ABC):
    """Abstract base for file storage providers."""

    @abstractmethod
    def upload_file(self, file_path: str, file_name: str, folder: str = "/") -> dict:
        """Upload a file and return metadata including URL."""
        ...

    @abstractmethod
    def delete_file(self, file_id: str) -> bool:
        """Delete a file by its provider-specific ID."""
        ...


class ImageKitProvider(BaseStorageProvider):
    def upload_file(self, file_path: str, file_name: str, folder: str = "/") -> dict:
        logger.info(f"ImageKit upload: {file_name} to {folder}")
        # ImageKit integration via MCP or SDK — placeholder for future
        return {"url": "", "file_id": "", "file_name": file_name}

    def delete_file(self, file_id: str) -> bool:
        logger.info(f"ImageKit delete: {file_id}")
        return True


class LocalStorageProvider(BaseStorageProvider):
    """Fallback: stores files on the local filesystem."""

    def upload_file(self, file_path: str, file_name: str, folder: str = "/") -> dict:
        import shutil, os
        dest_dir = os.path.join("uploads", folder.strip("/"))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, file_name)
        shutil.copy2(file_path, dest)
        return {"url": dest, "file_id": dest, "file_name": file_name}

    def delete_file(self, file_id: str) -> bool:
        import os
        if os.path.exists(file_id):
            os.remove(file_id)
            return True
        return False


_PROVIDERS: dict[str, type[BaseStorageProvider]] = {
    "imagekit": ImageKitProvider,
    "local": LocalStorageProvider,
}


def get_storage_provider(provider_name: Optional[str] = None) -> BaseStorageProvider:
    name = (provider_name or "local").lower()
    provider_cls = _PROVIDERS.get(name)
    if not provider_cls:
        raise ValueError(f"Unknown storage provider: {name}. Available: {list(_PROVIDERS.keys())}")
    return provider_cls()
