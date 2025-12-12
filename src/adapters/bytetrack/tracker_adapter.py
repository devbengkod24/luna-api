from ioutrack import ByteTrack
from typing import List, Dict
import numpy as np

from src.domain.entities.vision import DetectedFace, TrackedFace, BoundingBox
from src.ports.vision import ITrackerAdapter


class TrackerAdapter(ITrackerAdapter):
    """Adapter for ByteTrack tracker."""
    
    def __init__(
        self, 
        max_age: int = 30,
        min_hits: int = 3,
        init_tracker_min_score: float = 0.4
    ):
        """Initialize ByteTrack tracker.
        
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum detections before a track is confirmed
            init_tracker_min_score: Minimum confidence score to initialize a track
        """
        self.tracker = ByteTrack(
            max_age=max_age,
            min_hits=min_hits,
            init_tracker_min_score=init_tracker_min_score
        )
    
    def track(self, detections: List[DetectedFace], frame_id: int) -> List[Dict]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detected faces in current frame
            frame_id: Current frame number
            
        Returns:
            List[Dict]: Tracked objects with track_id and detection_index
        """
        if not detections:
            self.tracker.update(np.empty((0, 5)), return_indices=True)
            return []
        
        det_array = np.array([
            det.bbox.to_xyxy() + [det.confidence]
            for det in detections
        ])
        
        tracks = self.tracker.update(det_array, return_indices=True)
        
        results = []
        for track in tracks:
            track_id = int(track[4])
            detection_idx = int(track[5]) if len(track) > 5 and track[5] >= 0 else None
            
            bbox = BoundingBox.from_xyxy(
                int(track[0]),
                int(track[1]),
                int(track[2]),
                int(track[3])
            )
            
            embedding = None
            if detection_idx is not None and detection_idx < len(detections):
                embedding = detections[detection_idx].embedding
            
            results.append({
                "track_id": track_id,
                "bbox": bbox,
                "confidence": detections[detection_idx].confidence if detection_idx is not None else 0.0,
                "detection_index": detection_idx,
                "embedding": embedding
            })
        
        return results
    
    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()