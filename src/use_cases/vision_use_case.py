"""Vision streaming use case."""

import asyncio
import logging
from typing import Dict, List, Optional

from src.config import settings
from src.domain.entities.vision import (
    Confidence,
    Identity,
    TrackState,
    TrackedFaceData,
    VisionResult,
)
from src.ports.services import IFaceService
from src.ports.vision import IEmotionDetectionService, IFaceRecognitionService, ITrackerService

logger = logging.getLogger(__name__)


class VisionUseCase:
    """Orchestrates face detection, tracking, recognition, and emotion detection per frame."""

    def __init__(
        self,
        face_recognition: IFaceRecognitionService,
        face_store: Optional[IFaceService] = None,
        tracker: Optional[ITrackerService] = None,
        emotion_detection: Optional[IEmotionDetectionService] = None,
    ):
        self.face_recognition = face_recognition
        self.face_store = face_store
        self.tracker = tracker
        self.emotion_detection = emotion_detection

        self.similarity_threshold = settings.face_similarity_threshold
        self.emotion_interval = settings.vision_emotion_interval
        self.retry_interval = settings.vision_recognition_retry_interval
        self.max_attempts = settings.vision_max_recognition_attempts

        self._states: Dict[int, TrackState] = {}

    # --- Track state helpers ---

    def _get_state(self, track_id: int, frame_id: int) -> TrackState:
        if track_id not in self._states:
            self._states[track_id] = TrackState(
                track_id=track_id,
                first_seen_frame=frame_id,
                last_seen_frame=frame_id,
            )
        else:
            self._states[track_id].last_seen_frame = frame_id
        return self._states[track_id]

    def _cleanup_dead(self, dead_ids: List[int]) -> None:
        for tid in dead_ids:
            self._states.pop(tid, None)

    # --- Recognition ---

    async def _recognize(
        self, state: TrackState, aligned_face, quality: float
    ) -> Optional[Identity]:
        state.record_recognition_attempt(quality)

        embedding = await asyncio.to_thread(self.face_recognition.extract_embedding, aligned_face)
        if not embedding:
            return None
        state.last_embedding = embedding

        if not self.face_store:
            return None

        match = await self.face_store.search_face(embedding, threshold=self.similarity_threshold)
        if not match:
            return None

        face, score = match
        identity = Identity(face_id=face.id, name=face.name, nim=face.nim, similarity_score=score)
        state.bind_identity(identity)
        return identity

    # --- Per-track processing ---

    async def _process_track(
        self, tracked: TrackedFaceData, frame, frame_id: int
    ) -> VisionResult:
        state = self._get_state(tracked.track_id, frame_id)
        if tracked.landmark:
            state.last_landmark = tracked.landmark

        # Defaults — use cached values if already recognized
        name, nim, face_id = "Unknown", None, None
        face_conf = tracked.confidence
        emotion, emotion_conf = state.last_emotion, state.last_emotion_confidence

        if state.identity:
            name = state.identity.name
            nim = state.identity.nim
            face_id = state.identity.face_id
            face_conf = state.identity.similarity_score

        aligned = None
        if tracked.landmark:
            aligned = await asyncio.to_thread(self.face_recognition.align_face, frame, tracked.landmark)

        # Build parallel tasks
        tasks, labels = [], []

        need_recognition = (
            aligned is not None
            and self.face_store is not None
            and state.should_retry_recognition(
                tracked.confidence, self.retry_interval, self.max_attempts
            )
        )
        if need_recognition:
            tasks.append(self._recognize(state, aligned, tracked.confidence))
            labels.append("recognition")

        need_emotion = (
            aligned is not None
            and self.emotion_detection is not None
            and state.should_detect_emotion(frame_id, self.emotion_interval)
        )
        if need_emotion:
            tasks.append(asyncio.to_thread(
                self.emotion_detection.detect_emotion_with_context,
                aligned,
                frame,
                tracked.bbox,
            ))
            labels.append("emotion")

        for label, result in zip(labels, await asyncio.gather(*tasks)):
            if label == "recognition" and result:
                name, nim, face_id = result.name, result.nim, result.face_id
                face_conf = result.similarity_score
            elif label == "emotion" and result:
                state.update_emotion(result.emotion, result.confidence, frame_id)
                emotion, emotion_conf = result.emotion, result.confidence

        return VisionResult(
            bbox=tracked.bbox,
            confidence=Confidence(face=face_conf, emotion=emotion_conf),
            name=name,
            nim=nim,
            emotion=emotion,
            face_id=face_id,
            track_id=tracked.track_id,
        )

    # --- Main entry point ---

    async def execute(self, frame_bytes: bytes, frame_id: int = 0) -> List[VisionResult]:
        frame = await asyncio.to_thread(self.face_recognition.decode_from_bytes, frame_bytes)
        if frame is None:
            return []

        detections = await asyncio.to_thread(self.face_recognition.detect_faces_only_from_array, frame)
        if not detections:
            if self.tracker:
                _, dead = self.tracker.track([], frame_id)
                self._cleanup_dead(dead)
            return []

        if self.tracker:
            tracked_faces, dead = self.tracker.track(detections, frame_id)
            self._cleanup_dead(dead)
        else:
            tracked_faces = [
                TrackedFaceData(
                    track_id=d.detection_index,
                    bbox=d.bbox,
                    confidence=d.confidence,
                    detection_index=d.detection_index,
                    landmark=d.landmark,
                )
                for d in detections
            ]

        if not tracked_faces:
            return []

        return list(await asyncio.gather(*[
            self._process_track(t, frame, frame_id) for t in tracked_faces
        ]))

    def reset(self) -> None:
        if self.tracker:
            self.tracker.reset()
        self._states.clear()
