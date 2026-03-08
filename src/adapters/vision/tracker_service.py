"""ByteTrack adapter for multi-face tracking."""

import logging
from typing import List, Set, Tuple

import numpy as np
from ioutrack import ByteTrack

from src.config import settings
from src.domain.entities.vision import BoundingBox, FaceDetection, Landmark, TrackedFaceData
from src.ports.vision import ITrackerService

logger = logging.getLogger(__name__)


class TrackerService(ITrackerService):
    """ByteTrack tracker — associates detections across frames."""

    def __init__(
        self,
        max_age: int = None,
        min_hits: int = None,
        min_score: float = None,
    ):
        self.tracker = ByteTrack(
            max_age=max_age if max_age is not None else settings.tracker_max_age,
            min_hits=min_hits if min_hits is not None else settings.tracker_min_hits,
            init_tracker_min_score=min_score if min_score is not None else settings.tracker_min_score,
        )
        self._active: Set[int] = set()
        self._prev: Set[int] = set()

    def track(
        self, detections: List[FaceDetection], frame_id: int
    ) -> Tuple[List[TrackedFaceData], List[int]]:
        self._prev = self._active.copy()

        if not detections:
            self.tracker.update(np.empty((0, 5)), return_indices=True)
            dead = list(self._prev)
            self._active.clear()
            return [], dead

        det_array = np.array([d.bbox.to_xyxy() + [d.confidence] for d in detections])
        tracks = self.tracker.update(det_array, return_indices=True)

        results = []
        current = set()
        for track in tracks:
            tid = int(track[4])
            current.add(tid)
            det_idx = int(track[5]) if len(track) > 5 and track[5] >= 0 else None

            tracked_bbox = BoundingBox.from_xyxy(int(track[0]), int(track[1]), int(track[2]), int(track[3]))
            confidence, landmark = 0.0, None

            if det_idx is not None and det_idx < len(detections):
                det = detections[det_idx]
                confidence = det.confidence
                if det.landmark is not None:
                    bbox_changed = any(
                        abs(getattr(det.bbox, a) - getattr(tracked_bbox, a)) > 2
                        for a in ("x", "y", "width", "height")
                    )
                    landmark = det.landmark.transform_to_bbox(det.bbox, tracked_bbox) if bbox_changed else det.landmark

            results.append(TrackedFaceData(
                track_id=tid, bbox=tracked_bbox, confidence=confidence,
                detection_index=det_idx, landmark=landmark,
            ))

        self._active = current
        return results, list(self._prev - current)

    def reset(self) -> None:
        self.tracker.reset()
        self._active.clear()
        self._prev.clear()

    def get_active_track_ids(self) -> List[int]:
        return list(self._active)
