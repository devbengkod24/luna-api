"""Redis cache adapter implementation."""
import json
from typing import Optional, Any
import redis.asyncio as redis

from src.ports.cache import ICacheAdapter


class RedisCacheAdapter(ICacheAdapter):
    """Cache adapter using Redis."""
    
    def __init__(self, url: str):
        """Initialize Redis client.
        
        Args:
            url: Redis connection URL (e.g., redis://localhost:6379)
        """
        self.client = redis.from_url(url, decode_responses=True)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value or None
        """
        value = await self.client.get(key)
        if value is None:
            return None
        
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
            
        Returns:
            bool: True if successful
        """
        try:
            serialized = json.dumps(value) if not isinstance(value, str) else value
            if ttl:
                await self.client.setex(key, ttl, serialized)
            else:
                await self.client.set(key, serialized)
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        result = await self.client.delete(key)
        return result > 0
    
    async def close(self):
        """Close Redis connection."""
        await self.client.close()

