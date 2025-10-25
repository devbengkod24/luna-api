"""Vision port interfaces."""
from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities.vision import VisionResult


class IVisionAdapter(ABC):
    """Interface for vision adapter (emotion detection + face recognition)."""
    
    @abstractmethod
    async def analyze_frame(self, frame_base64: str) -> Optional[VisionResult]:
        """Analyze a video frame for emotion and face recognition.
        
        Args:
            frame_base64: Base64-encoded JPEG frame
            
        Returns:
            Optional[VisionResult]: Analysis result or None if no face detected
        """
        pass

