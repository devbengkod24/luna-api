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
from src.domain.image import Image
from src.ports.services import IFaceService
from src.ports.vision import (
    IEmotionDetectionService,
    IFaceAligner,
    IFaceCropper,
    IFaceDetector,
    IFaceEmbedder,
    IImageDecoder,
    ITrackerService,
)

logger = logging.getLogger(__name__)


class VisionUseCase:
    """Orchestrates face detection, tracking, recognition, and emotion detection per frame."""

    def __init__(
        self,
        image_decoder: IImageDecoder,
        face_detector: IFaceDetector,
        face_aligner: IFaceAligner,
        face_embedder: IFaceEmbedder,
        face_store: Optional[IFaceService] = None,
        tracker: Optional[ITrackerService] = None,
        emotion_detection: Optional[IEmotionDetectionService] = None,
        face_cropper: Optional[IFaceCropper] = None,
    ):
        self.image_decoder = image_decoder
        self.face_detector = face_detector
        self.face_aligner = face_aligner
        self.face_embedder = face_embedder
        self.face_store = face_store
        self.tracker = tracker
        self.emotion_detection = emotion_detection
        self.face_cropper = face_cropper

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
        self, state: TrackState, aligned_face: Image, quality: float
    ) -> Optional[Identity]:
        state.record_recognition_attempt(quality)

        embedding = await self.face_embedder.embed(aligned_face)
        if not embedding:
            return None
        state.last_embedding = embedding

        if not self.face_store:
            return None

        match = await self.face_store.search_face(
            embedding, threshold=self.similarity_threshold
        )
        if not match:
            return None

        face, score = match
        identity = Identity(
            face_id=face.id, name=face.name, nim=face.nim, similarity_score=score
        )
        state.bind_identity(identity)
        return identity

    # --- Per-track processing ---

    async def _process_track(
        self, tracked: TrackedFaceData, frame: Image, frame_id: int
    ) -> VisionResult:
        state = self._get_state(tracked.track_id, frame_id)
        if tracked.landmark:
            state.last_landmark = tracked.landmark

        name, nim, face_id = "Unknown", None, None
        face_conf = tracked.confidence
        emotion, emotion_conf = state.last_emotion, state.last_emotion_confidence

        if state.identity:
            name = state.identity.name
            nim = state.identity.nim
            face_id = state.identity.face_id
            face_conf = state.identity.similarity_score

        # Decide what this frame needs BEFORE running the (expensive) align.
        # Align is only worthwhile if at least one downstream step will consume
        # it; in steady state most frames need neither (identity already bound,
        # emotion sampled at interval), so skipping the align here cuts a real
        # chunk of CPU/GPU time at high FPS.
        need_recognition = (
            tracked.landmark is not None
            and self.face_store is not None
            and state.should_retry_recognition(
                tracked.confidence, self.retry_interval, self.max_attempts
            )
        )
        need_emotion = (
            tracked.landmark is not None
            and self.emotion_detection is not None
            and state.should_detect_emotion(frame_id, self.emotion_interval)
        )

        aligned: Optional[Image] = None
        if tracked.landmark and (need_recognition or need_emotion):
            aligned = await self.face_aligner.align(frame, tracked.landmark)

        tasks, labels = [], []
        if need_recognition and aligned is not None:
            tasks.append(self._recognize(state, aligned, tracked.confidence))
            labels.append("recognition")
        if need_emotion and aligned is not None:
            tasks.append(
                self.emotion_detection.detect_emotion_with_context(aligned, frame, tracked.bbox)
            )
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

    async def _decode_and_track(
        self, frame_bytes: bytes, frame_id: int
    ) -> tuple[Optional[Image], List[TrackedFaceData]]:
        """Decode a frame and produce tracked faces (shared detect+track path)."""
        frame = await self.image_decoder.decode(frame_bytes)
        if frame is None:
            return None, []

        detections = await self.face_detector.detect(frame)
        if not detections:
            if self.tracker:
                _, dead = await self.tracker.track([], frame_id)
                self._cleanup_dead(dead)
            return frame, []

        if self.tracker:
            tracked_faces, dead = await self.tracker.track(detections, frame_id)
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

        return frame, tracked_faces

    async def execute(self, frame_bytes: bytes, frame_id: int = 0) -> List[VisionResult]:
        frame, tracked_faces = await self._decode_and_track(frame_bytes, frame_id)
        if frame is None or not tracked_faces:
            return []

        return list(await asyncio.gather(*[
            self._process_track(t, frame, frame_id) for t in tracked_faces
        ]))

    async def execute_with_crops(
        self, frame_bytes: bytes, frame_id: int = 0
    ) -> tuple[List[VisionResult], Dict[int, Image]]:
        """Like `execute`, but also return a natural bbox crop per track_id.

        Used by the session router to buffer face crops for active-speaker
        detection. Crops are the padded bbox region (not the canonical aligned
        face) — LR-ASD needs natural jaw/lip motion. Returns ({}, {}) shapes
        when nothing is detected.
        """
        frame, tracked_faces = await self._decode_and_track(frame_bytes, frame_id)
        if frame is None or not tracked_faces:
            return [], {}

        results = await asyncio.gather(*[
            self._process_track(t, frame, frame_id) for t in tracked_faces
        ])

        crops: Dict[int, Image] = {}
        if self.face_cropper is not None:
            cropped = await asyncio.gather(*[
                self.face_cropper.crop(frame, t.bbox, padding=0.4) for t in tracked_faces
            ])
            for t, c in zip(tracked_faces, cropped):
                if c is not None:
                    crops[t.track_id] = c

        return list(results), crops

    async def reset(self) -> None:
        if self.tracker:
            await self.tracker.reset()
        self._states.clear()
