"""Vision streaming use case."""
from typing import Optional

from src.ports.vision import IVisionAdapter
from src.ports.cache import ICacheAdapter
from src.domain.entities.vision import VisionResult


class VisionStreamUseCase:
    """Use case for real-time vision analysis."""
    
    def __init__(
        self,
        vision_adapter: IVisionAdapter,
        cache_adapter: Optional[ICacheAdapter] = None,
    ):
        """Initialize use case with dependencies.
        
        Args:
            vision_adapter: Vision adapter for emotion and face recognition
            cache_adapter: Optional cache adapter for caching results
        """
        self.vision_adapter = vision_adapter
        self.cache_adapter = cache_adapter
    
    async def execute(self, frame_base64: str) -> Optional[VisionResult]:
        """Analyze a video frame.
        
        Args:
            frame_base64: Base64-encoded JPEG frame
            
        Returns:
            Optional[VisionResult]: Analysis result or None if no face detected
        """
        # Analyze frame
        result = await self.vision_adapter.analyze_frame(frame_base64)
        
        # Cache result if available (optional - for recent user state)
        if result and self.cache_adapter:
            await self.cache_adapter.set(
                key="latest_vision_result",
                value=result.to_dict(),
                ttl=60,  # Cache for 60 seconds
            )
        
        return result

