"""Vision adapter implementation with InsightFace integration.

This adapter handles:
- Face detection using SCRFD
- Face recognition using ArcFace embeddings
- Face matching against registered faces in Qdrant
- Emotion detection (placeholder for future implementation)
"""
import base64
from typing import Optional, List, Tuple

from src.config import settings
from src.ports.vision import IVisionAdapter, IFaceRecognitionAdapter, IFaceVectorStore
from src.domain.entities.vision import (
    VisionResult,
    FaceRecognitionResult,
    EmotionRecognitionResult,
    BoundingBox,
    DetectedFace,
)


class VisionAdapter(IVisionAdapter):
    """Vision adapter for face detection and recognition using InsightFace."""
    
    def __init__(
        self,
        face_recognition: Optional[IFaceRecognitionAdapter] = None,
        face_store: Optional[IFaceVectorStore] = None,
        similarity_threshold: float = None,
    ):
        """Initialize vision adapter.
        
        Args:
            face_recognition: InsightFace adapter for face detection/recognition
            face_store: Qdrant face store for registered faces
            similarity_threshold: Minimum similarity for face matching (from config if None)
        """
        self.face_recognition = face_recognition
        self.face_store = face_store
        self.similarity_threshold = similarity_threshold or settings.face_similarity_threshold
        
        self._log_init_status()
    
    def _log_init_status(self):
        """Log initialization status."""
        if self.face_recognition:
            print(f"VisionAdapter initialized with face recognition")
        else:
            print("VisionAdapter initialized WITHOUT face recognition (placeholder mode)")
        
        if self.face_store:
            print(f"VisionAdapter connected to face store")
        else:
            print("VisionAdapter: No face store connected")
        
        print(f"VisionAdapter similarity threshold: {self.similarity_threshold}")
    
    async def analyze_frame(self, frame_base64: str) -> Optional[VisionResult]:
        """Analyze a video frame for face detection and recognition.
        
        Args:
            frame_base64: Base64-encoded JPEG frame
            
        Returns:
            Optional[VisionResult]: Analysis result with detected faces,
                                   or None if no face detected
        """
        # Decode base64 to bytes
        try:
            image_bytes = base64.b64decode(frame_base64)
        except Exception as e:
            print(f"Error decoding base64: {e}")
            return None
        
        # If no face recognition adapter, return placeholder
        if not self.face_recognition:
            return self._placeholder_response()
        
        # Detect faces using InsightFace (SCRFD + ArcFace)
        detected_faces = self.face_recognition.detect_faces(image_bytes)
        
        if not detected_faces:
            return None
        
        faces = []
        emotions = []
        
        for detected in detected_faces:
            # Try to recognize face from Qdrant
            recognition_result = await self._recognize_face(detected)
            
            # Use matching confidence if face recognized, otherwise use detection confidence
            confidence = recognition_result["confidence"] if recognition_result["face_id"] else detected.confidence
            
            faces.append(FaceRecognitionResult(
                name=recognition_result["name"],
                nim=recognition_result["nim"],
                box=detected.box,
                confidence=confidence,
                face_id=recognition_result["face_id"],
            ))
            
            # Emotion detection placeholder
            # TODO: Implement real emotion detection with FER or similar
            emotion = self._detect_emotion_placeholder(detected)
            emotions.append(EmotionRecognitionResult(
                emotion=emotion["emotion"],
                box=detected.box,
                confidence=emotion["confidence"],
            ))
        
        return VisionResult(faces=faces, emotions=emotions)
    
    async def _recognize_face(self, detected: DetectedFace) -> dict:
        """Recognize a detected face against registered faces.
        
        Args:
            detected: Detected face with embedding
            
        Returns:
            dict: Recognition result with name, nim, confidence (matching score), face_id
                  Note: confidence will be matching score if recognized, caller should use
                  detection confidence for Unknown faces
        """
        result = {
            "name": "Unknown",
            "nim": None,
            "confidence": 0.0,  # This is matching confidence, not detection confidence
            "face_id": None,
        }
        
        # Check if we have face store and embedding
        if not self.face_store or not detected.embedding:
            return result
        
        # Search for matching face in Qdrant
        try:
            search_result = await self.face_store.search_face(
                detected.embedding,
                threshold=self.similarity_threshold,
            )
            
            if search_result:
                face_data, similarity_score = search_result
                result["name"] = face_data.name
                result["nim"] = face_data.nim
                result["confidence"] = similarity_score
                result["face_id"] = face_data.id
                
        except Exception as e:
            print(f"Error searching face: {e}")
        
        return result
    
    def _detect_emotion_placeholder(self, detected: DetectedFace) -> dict:
        """Placeholder for emotion detection.
        
        TODO: Implement with FER, DeepFace, or similar library.
        
        Args:
            detected: Detected face
            
        Returns:
            dict: Emotion detection result
        """
        return {
            "emotion": "neutral",
            "confidence": 0.5,
        }
    
    def _placeholder_response(self) -> VisionResult:
        """Return placeholder response when adapters not initialized."""
        return VisionResult(
            faces=[
                FaceRecognitionResult(
                    name="Unknown",
                    nim=None,
                    box=BoundingBox(x=10, y=10, width=50, height=50),
                    confidence=0.0,
                ),
            ],
            emotions=[
                EmotionRecognitionResult(
                    emotion="neutral",
                    box=BoundingBox(x=10, y=10, width=50, height=50),
                    confidence=0.5,
                ),
            ],
        )
    
    async def get_face_embedding(self, frame_base64: str) -> Optional[List[float]]:
        """Get face embedding from a frame.
        
        Args:
            frame_base64: Base64-encoded JPEG frame
            
        Returns:
            Optional[List[float]]: Face embedding vector or None
        """
        if not self.face_recognition:
            return None
        
        try:
            image_bytes = base64.b64decode(frame_base64)
            return self.face_recognition.get_embedding(image_bytes)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    async def detect_faces(self, frame_base64: str) -> List[DetectedFace]:
        """Detect all faces in a frame.
        
        Args:
            frame_base64: Base64-encoded JPEG frame
            
        Returns:
            List[DetectedFace]: List of detected faces with embeddings
        """
        if not self.face_recognition:
            return []
        
        try:
            image_bytes = base64.b64decode(frame_base64)
            return self.face_recognition.detect_faces(image_bytes)
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def is_same_person(
        self,
        embedding1: List[float],
        embedding2: List[float],
        threshold: float = None,
    ) -> Tuple[bool, float]:
        """Check if two embeddings belong to the same person.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            Tuple[bool, float]: (is_same_person, similarity_score)
        """
        if not self.face_recognition:
            return False, 0.0
        
        threshold = threshold or self.similarity_threshold
        similarity = self.face_recognition.compute_similarity(embedding1, embedding2)
        
        return similarity >= threshold, similarity
