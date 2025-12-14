"""Cache port interfaces."""
from abc import ABC, abstractmethod
from typing import Optional, Any


class ICacheAdapter(ABC):
    """Interface for cache adapter."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value or None
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        pass

