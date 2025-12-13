"""Vision streaming use case with track-based processing."""
from typing import Optional, List, Dict
import base64
import asyncio

import numpy as np

from src.ports.vision import (
    IFaceRecognitionAdapter,
    IFaceVectorStore,
    IEmotionDetectionAdapter,
    ITrackerAdapter,
)
from src.domain.entities.vision import (
    VisionResult,
    Confidence,
    BoundingBox,
    TrackState,
    TrackedFaceData,
    Identity,
    Landmark,
)


class TrackStateManager:
    """Manages track states and their lifecycle.
    
    Responsibilities:
    - Create/delete track states based on tracker output
    - Cache identity bindings (track_id → identity_id)
    - Control recognition retry logic
    - Track emotion detection timing
    """
    
    def __init__(
        self,
        recognition_retry_interval: float = 2.0,
        max_recognition_attempts: int = 5,
        quality_improvement_threshold: float = 0.1,
        emotion_detection_interval: int = 5,
    ):
        """Initialize track state manager.
        
        Args:
            recognition_retry_interval: Minimum seconds between recognition retries
            max_recognition_attempts: Maximum number of recognition attempts per track
            quality_improvement_threshold: Quality improvement needed to trigger retry
            emotion_detection_interval: Frames between emotion detection
        """
        self._states: Dict[int, TrackState] = {}
        self.recognition_retry_interval = recognition_retry_interval
        self.max_recognition_attempts = max_recognition_attempts
        self.quality_improvement_threshold = quality_improvement_threshold
        self.emotion_detection_interval = emotion_detection_interval
    
    def get_or_create(self, track_id: int, frame_id: int) -> TrackState:
        """Get existing track state or create new one.
        
        Args:
            track_id: Track ID from tracker
            frame_id: Current frame number
            
        Returns:
            TrackState for this track
        """
        if track_id not in self._states:
            self._states[track_id] = TrackState(
                track_id=track_id,
                first_seen_frame=frame_id,
                last_seen_frame=frame_id,
            )
        else:
            self._states[track_id].last_seen_frame = frame_id
        
        return self._states[track_id]
    
    def get(self, track_id: int) -> Optional[TrackState]:
        """Get track state by ID."""
        return self._states.get(track_id)
    
    def delete(self, track_id: int):
        """Delete track state (called when track dies)."""
        if track_id in self._states:
            del self._states[track_id]
    
    def cleanup_dead_tracks(self, dead_track_ids: List[int]):
        """Remove states for dead tracks.
        
        Args:
            dead_track_ids: List of track IDs that are no longer active
        """
        for track_id in dead_track_ids:
            self.delete(track_id)
    
    def should_recognize(self, track_id: int, face_quality: float) -> bool:
        """Check if recognition should be performed for this track.
        
        Args:
            track_id: Track ID
            face_quality: Current face quality score (e.g., confidence)
            
        Returns:
            True if recognition should be attempted
        """
        state = self.get(track_id)
        if state is None:
            return True
        
        return state.should_retry_recognition(
            current_quality=face_quality,
            min_retry_interval=self.recognition_retry_interval,
            max_attempts=self.max_recognition_attempts,
            quality_improvement_threshold=self.quality_improvement_threshold,
        )
    
    def should_detect_emotion(self, track_id: int, frame_id: int) -> bool:
        """Check if emotion detection should be performed.
        
        Args:
            track_id: Track ID
            frame_id: Current frame number
            
        Returns:
            True if emotion should be detected
        """
        state = self.get(track_id)
        if state is None:
            return True
        
        return state.should_detect_emotion(
            current_frame=frame_id,
            interval=self.emotion_detection_interval,
        )
    
    def get_all_active_states(self) -> Dict[int, TrackState]:
        """Get all active track states."""
        return self._states.copy()


class VisionUseCase:
    """Use case for real-time vision analysis with track-based processing.
    
    Pipeline:
    1. Frame → Face Detection (bbox + conf + landmark)
    2. ByteTrack (track_id + tracked_bbox)
    3. For each track:
       - Crop + Align (from tracked_bbox + landmark)
       - Face Recognition (embedding) - only for new tracks or retries
       - Similarity Search (identity) - only when needed
       - Identity Binding (track_id → identity_id)
       - Emotion Detection (parallel, every N frames)
    """
    
    def __init__(
        self,
        face_recognition: IFaceRecognitionAdapter,
        face_store: Optional[IFaceVectorStore] = None,
        tracker: Optional[ITrackerAdapter] = None,
        emotion_detection: Optional[IEmotionDetectionAdapter] = None,
        similarity_threshold: float = 0.6,
        emotion_detection_interval: int = 5,
        recognition_retry_interval: float = 2.0,
        max_recognition_attempts: int = 5,
    ):
        """Initialize use case with dependencies.
        
        Args:
            face_recognition: Face recognition adapter
            face_store: Face vector store for identity search
            tracker: Face tracker adapter
            emotion_detection: Emotion detection adapter
            similarity_threshold: Similarity threshold for face matching
            emotion_detection_interval: Frames between emotion detection
            recognition_retry_interval: Seconds between recognition retries
            max_recognition_attempts: Max recognition attempts per track
        """
        self.face_recognition = face_recognition
        self.face_store = face_store
        self.tracker = tracker
        self.emotion_detection = emotion_detection
        self.similarity_threshold = similarity_threshold
        
        # Initialize track state manager
        self.track_manager = TrackStateManager(
            recognition_retry_interval=recognition_retry_interval,
            max_recognition_attempts=max_recognition_attempts,
            emotion_detection_interval=emotion_detection_interval,
        )
        
        # Cache decoded frame for reuse within same frame
        self._current_frame: Optional[np.ndarray] = None
    
    async def _recognize_and_bind_identity(
        self,
        state: TrackState,
        aligned_face: np.ndarray,
        face_quality: float,
    ) -> Optional[Identity]:
        """Perform face recognition and bind identity to track.
        
        Args:
            state: Track state
            aligned_face: Aligned face image for recognition
            face_quality: Face quality score
            
        Returns:
            Identity if found, None otherwise
        """
        # Record attempt
        state.record_recognition_attempt(face_quality)
        
        # Extract embedding
        embedding = self.face_recognition.extract_embedding(aligned_face)
        if not embedding:
            return None
        
        # Store embedding in state
        state.last_embedding = embedding
        
        # Search for identity in database
        if self.face_store:
            match = await self.face_store.search_face(
                embedding,
                threshold=self.similarity_threshold,
            )
            
            if match:
                face_data, similarity = match
                identity = Identity(
                    face_id=face_data.id,
                    name=face_data.name,
                    nim=face_data.nim,
                    similarity_score=similarity,
                )
                state.bind_identity(identity)
                return identity
        
        return None
    
    async def _process_single_track(
        self,
        tracked: TrackedFaceData,
        frame: np.ndarray,
        frame_id: int,
    ) -> VisionResult:
        """Process a single tracked face.
        
        Performs:
        - Crop + Align
        - Recognition (if needed)
        - Emotion detection (if needed)
        
        Args:
            tracked: Tracked face data from ByteTrack
            frame: Full frame image
            frame_id: Current frame number
            
        Returns:
            VisionResult for this track
        """
        track_id = tracked.track_id
        bbox = tracked.bbox
        landmark = tracked.landmark
        
        # Get or create track state
        state = self.track_manager.get_or_create(track_id, frame_id)
        
        # Update landmark in state
        if landmark:
            state.last_landmark = landmark
        
        # Default values
        name = "Unknown"
        nim = None
        face_id = None
        face_confidence = tracked.confidence
        emotion = state.last_emotion
        emotion_confidence = state.last_emotion_confidence
        
        # Use cached identity if available
        if state.identity:
            name = state.identity.name
            nim = state.identity.nim
            face_id = state.identity.face_id
            face_confidence = state.identity.similarity_score
        
        # Prepare aligned face for recognition/emotion
        aligned_face = None
        if landmark:
            aligned_face = self.face_recognition.align_face(frame, landmark)
        
        # Build parallel tasks
        tasks = []
        task_types = []
        
        # Task 1: Recognition (if needed)
        should_recognize = (
            self.track_manager.should_recognize(track_id, tracked.confidence) and
            aligned_face is not None and
            self.face_store is not None
        )
        
        if should_recognize:
            tasks.append(
                self._recognize_and_bind_identity(state, aligned_face, tracked.confidence)
            )
            task_types.append("recognition")
        
        # Task 2: Emotion detection (if needed)
        should_detect_emotion = (
            self.track_manager.should_detect_emotion(track_id, frame_id) and
            aligned_face is not None and
            self.emotion_detection is not None
        )
        
        if should_detect_emotion:
            # Run emotion detection in thread pool (CPU-bound)
            tasks.append(
                asyncio.to_thread(
                    self.emotion_detection.detect_emotion_from_array,
                    aligned_face,
                )
            )
            task_types.append("emotion")
        
        # Execute tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks)
            
            for task_type, result in zip(task_types, results):
                if task_type == "recognition" and result:
                    # Update from newly recognized identity
                    name = result.name
                    nim = result.nim
                    face_id = result.face_id
                    face_confidence = result.similarity_score
                
                elif task_type == "emotion" and result:
                    # Update emotion
                    state.update_emotion(result.emotion, result.confidence, frame_id)
                    emotion = result.emotion
                    emotion_confidence = result.confidence
        
        return VisionResult(
            bbox=bbox,
            confidence=Confidence(
                face=face_confidence,
                emotion=emotion_confidence,
            ),
            name=name,
            nim=nim,
            emotion=emotion,
            face_id=face_id,
            track_id=track_id,
        )
    
    async def execute(self, frame_base64: str, frame_id: int = 0) -> List[VisionResult]:
        """Analyze a video frame with face detection, tracking, recognition, and emotion.
        
        Pipeline:
        1. Decode frame
        2. Detect faces (bbox + conf + landmark)
        3. Track faces (track_id + tracked_bbox)
        4. For each track: crop, align, recognize (if needed), emotion (if needed)
        5. Return results
        
        Args:
            frame_base64: Base64 encoded frame image
            frame_id: Current frame number
            
        Returns:
            List[VisionResult]: Results for each tracked face
        """
        # Step 1: Decode frame
        image_bytes = base64.b64decode(frame_base64)
        frame = self.face_recognition.decode_from_bytes(image_bytes)
        if frame is None:
            return []
        
        self._current_frame = frame
        
        # Step 2: Detect faces (bbox + conf + landmark only, no embedding)
        detections = self.face_recognition.detect_faces_only_from_array(frame)
        
        if not detections:
            # No faces detected - all tracks are dead
            if self.tracker:
                _, dead_track_ids = self.tracker.track([], frame_id)
                self.track_manager.cleanup_dead_tracks(dead_track_ids)
            return []
        
        # Step 3: Track faces
        if self.tracker:
            tracked_faces, dead_track_ids = self.tracker.track(detections, frame_id)
            
            # Cleanup dead tracks
            self.track_manager.cleanup_dead_tracks(dead_track_ids)
        else:
            # No tracker - create pseudo-tracks from detections
            tracked_faces = [
                TrackedFaceData(
                    track_id=det.detection_index,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    detection_index=det.detection_index,
                    landmark=det.landmark,
                )
                for det in detections
            ]
        
        if not tracked_faces:
            return []
        
        # Step 4: Process all tracks in parallel
        tasks = [
            self._process_single_track(tracked, frame, frame_id)
            for tracked in tracked_faces
        ]
        
        results = await asyncio.gather(*tasks)
        
        return list(results)
    
    def reset(self):
        """Reset all states (e.g., for new video stream)."""
        if self.tracker:
            self.tracker.reset()
        
        # Clear all track states
        for track_id in list(self.track_manager._states.keys()):
            self.track_manager.delete(track_id)
        
        self._current_frame = None
