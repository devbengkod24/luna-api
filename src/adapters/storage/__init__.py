"""Storage adapters (file/object storage).

Organized by *function* (storage) instead of framework/vendor.
"""

from src.adapters.filesystem.storage_adapter import FileStorageAdapter

__all__ = ["FileStorageAdapter"]

