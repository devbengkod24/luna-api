"""Vision streaming use case."""
from typing import Optional, List
import base64
import asyncio

from src.ports.vision import (
    IFaceRecognitionAdapter,
    IFaceVectorStore,
    IEmotionDetectionAdapter,
    ITrackerAdapter
)
from src.ports.cache import ICacheAdapter
from src.domain.entities.vision import VisionResult, Confidence, BoundingBox


class VisionStreamUseCase:
    """Use case for real-time vision analysis."""
    
    def __init__(
        self,
        face_recognition: IFaceRecognitionAdapter,
        face_store: Optional[IFaceVectorStore] = None,
        tracker: Optional[ITrackerAdapter] = None,
        emotion_detection: Optional[IEmotionDetectionAdapter] = None,
        cache_adapter: Optional[ICacheAdapter] = None,
        similarity_threshold: float = 0.6,
        emotion_detection_interval: int = 3,  # Detect emotion every N frames
    ):
        """Initialize use case with dependencies.
        
        Args:
            face_recognition: Face recognition adapter
            face_store: Face vector store
            tracker: Face tracker adapter
            emotion_detection: Emotion detection adapter
            cache_adapter: Cache adapter for caching results
            similarity_threshold: Similarity threshold for face matching
        """
        self.face_recognition = face_recognition
        self.face_store = face_store
        self.tracker = tracker
        self.emotion_detection = emotion_detection
        self.cache_adapter = cache_adapter
        self.similarity_threshold = similarity_threshold
        self.emotion_detection_interval = emotion_detection_interval
        self._last_emotion_frame = {}  # track_id -> frame_id
    
    async def _process_single_face(
        self, 
        detected, 
        embedding: List[float],
        bbox: BoundingBox,
        track_id: Optional[int] = None,
        frame_id: int = 0
    ) -> VisionResult:
        """Process a single face (recognition + emotion) in parallel."""
        name = "Unknown"
        nim = None
        face_id = None
        face_confidence = detected.confidence
        
        # Parallel: face recognition search
        face_search_task = None
        if self.face_store and embedding:
            face_search_task = self.face_store.search_face(
                embedding,
                threshold=self.similarity_threshold
            )
        
        # Parallel: emotion detection
        emotion_task = None
        if self.emotion_detection and detected.crop_bytes:
            # Run emotion detection in thread pool (CPU-bound)
            emotion_task = asyncio.to_thread(
                self.emotion_detection.detect_emotion,
                detected.crop_bytes
            )
        
        # Wait for both tasks
        if face_search_task:
            match_result = await face_search_task
            if match_result:
                face_data, similarity_score = match_result
                name = face_data.name
                nim = face_data.nim
                face_id = face_data.id
                face_confidence = similarity_score
        
        emotion = "neutral"
        emotion_confidence = 0.0
        if emotion_task:
            detected_emotion = await emotion_task
            emotion = detected_emotion.emotion
            emotion_confidence = detected_emotion.confidence
        
        return VisionResult(
            box=bbox,
            confidence=Confidence(
                face=face_confidence,
                emotion=emotion_confidence
            ),
            name=name,
            nim=nim,
            emotion=emotion,
            face_id=face_id,
            track_id=track_id
        )
    
    async def execute(self, frame_base64: str, frame_id: int = 0) -> List[VisionResult]:
        """Analyze a video frame with face detection, tracking, recognition, and emotion detection."""
        image_bytes = base64.b64decode(frame_base64)
        
        # Step 1: Detect faces
        detected_faces = self.face_recognition.detect_faces(
            image_bytes,
            include_crops=bool(self.emotion_detection)
        )
        
        if not detected_faces:
            return []
        
        # Step 2: Track faces
        tracked_objects = []
        if self.tracker:
            tracked_objects = self.tracker.track(detected_faces, frame_id)
        
        # Step 3 & 4: Process all faces in parallel
        if tracked_objects:
            tasks = []
            for tracked in tracked_objects:
                detection_idx = tracked.get("detection_index")
                if detection_idx is None or detection_idx >= len(detected_faces):
                    continue
                
                detected = detected_faces[detection_idx]
                embedding = tracked.get("embedding") or detected.embedding
                bbox = tracked["bbox"]
                
                tasks.append(self._process_single_face(
                    detected, embedding, bbox, tracked["track_id"], frame_id
                ))
            
            results = await asyncio.gather(*tasks)
        else:
            tasks = []
            for detected in detected_faces:
                tasks.append(self._process_single_face(
                    detected, detected.embedding, detected.bbox, None, frame_id
                ))
            
            results = await asyncio.gather(*tasks)
        
        return list(results)

