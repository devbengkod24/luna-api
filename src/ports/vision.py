"""Vision port interfaces."""
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict

from src.domain.entities.vision import (
    DetectedFace,
    FaceData,
    DetectedEmotion,
)

class IFaceRecognitionAdapter(ABC):
    """Interface for face recognition adapter using InsightFace."""
    
    @abstractmethod
    def detect_faces(self, image_bytes: bytes, include_crops: bool = False) -> List[DetectedFace]:
        """Detect all faces in an image and extract embeddings.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            include_crops: Whether to include cropped face images in the result
        Returns:
            List[DetectedFace]: List of detected faces with embeddings
        """
        pass
    
    @abstractmethod
    def get_embedding(self, image_bytes: bytes) -> Optional[List[float]]:
        """Get face embedding from an image with single face.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            
        Returns:
            Optional[List[float]]: Face embedding vector or None if no face detected
        """
        pass
    
    @abstractmethod
    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Similarity score (0-1)
        """
        pass


class IFaceVectorStore(ABC):
    """Interface for face vector storage (Qdrant)."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store collection."""
        pass
    
    @abstractmethod
    async def add_face(self, face_data: FaceData) -> str:
        """Add a face to the vector store.
        
        Args:
            face_data: Face data with embedding
            
        Returns:
            str: Face ID
        """
        pass
    
    @abstractmethod
    async def search_face(
        self, embedding: List[float], threshold: float = 0.6
    ) -> Optional[Tuple[FaceData, float]]:
        """Search for a matching face in the vector store.
        
        Args:
            embedding: Face embedding vector
            threshold: Minimum similarity threshold
            
        Returns:
            Optional[Tuple[FaceData, float]]: Matching face data and score, or None
        """
        pass
    
    @abstractmethod
    async def get_all_faces(self) -> List[FaceData]:
        """Get all registered faces.
        
        Returns:
            List[FaceData]: List of all face data (without embeddings)
        """
        pass
    
    @abstractmethod
    async def delete_face(self, face_id: str) -> bool:
        """Delete a face from the vector store.
        
        Args:
            face_id: Face ID to delete
            
        Returns:
            bool: True if deleted successfully
        """
        pass
    
    @abstractmethod
    async def get_face_by_nim(self, nim: str) -> Optional[FaceData]:
        """Get face data by NIM.
        
        Args:
            nim: Student NIM
            
        Returns:
            Optional[FaceData]: Face data if found
        """
        pass

class ITrackerAdapter(ABC):
    """Interface for object/person tracking across frames."""
    
    @abstractmethod
    def track(self, detections: List[DetectedFace], frame_id: int) -> List[Dict]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detected faces in current frame
            frame_id: Current frame number
            
        Returns:
            List[Dict]: Tracked objects with track_id, box, confidence, etc.
            Format: [
                {
                    "track_id": int,
                    "box": BoundingBox,
                    "confidence": float,
                    "embedding": List[float],  # Optional
                    "age": int,  # Frames since first detection
                }
            ]
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset tracker state (e.g., new video stream)."""
        pass

class IEmotionDetectionAdapter(ABC):
    """Interface for emotion detection."""
    
    @abstractmethod
    def detect_emotion(
        self, face_crop_bytes: bytes,
    ) -> DetectedEmotion:
        """Detect emotion from face region.
        
        Args:
            face_crop_bytes: Cropped face image bytes
            box: Face bounding box
            
        Returns:
            Dict with "emotion" (str) and "confidence" (float)
        """
        pass
