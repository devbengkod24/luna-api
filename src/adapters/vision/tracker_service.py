"""ByteTrack adapter for multi-object tracking."""
from ioutrack import ByteTrack
from typing import List, Tuple, Set
import numpy as np

from src.domain.entities.vision import (
    FaceDetection,
    TrackedFaceData,
    BoundingBox,
    Landmark,
)
from src.ports.vision import ITrackerAdapter


class TrackerAdapter(ITrackerAdapter):
    """Adapter for ByteTrack tracker.
    
    Manages track lifecycle and associates detection data (landmarks) to tracks.
    """
    
    def __init__(
        self, 
        max_age: int = 30,
        min_hits: int = 3,
        init_tracker_min_score: float = 0.4,
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
            init_tracker_min_score=init_tracker_min_score,
        )
        self.max_age = max_age
        
        # Track lifecycle management
        self._active_track_ids: Set[int] = set()
        self._previous_track_ids: Set[int] = set()
    
    def track(
        self, 
        detections: List[FaceDetection], 
        frame_id: int,
    ) -> Tuple[List[TrackedFaceData], List[int]]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detected faces in current frame
            frame_id: Current frame number
            
        Returns:
            Tuple containing:
            - List[TrackedFaceData]: Active tracked faces with associated detection data
            - List[int]: Dead track IDs (tracks that have been lost)
        """
        # Store previous active tracks to detect dead tracks
        self._previous_track_ids = self._active_track_ids.copy()
        
        if not detections:
            # Update tracker with empty detections
            self.tracker.update(np.empty((0, 5)), return_indices=True)
            
            # All previous tracks are now dead
            dead_track_ids = list(self._previous_track_ids)
            self._active_track_ids.clear()
            
            return [], dead_track_ids
        
        # Prepare detection array for ByteTrack: [x1, y1, x2, y2, confidence]
        det_array = np.array([
            det.bbox.to_xyxy() + [det.confidence]
            for det in detections
        ])
        
        # Update tracker
        tracks = self.tracker.update(det_array, return_indices=True)
        
        # Build tracked face data with associated detection info
        results = []
        current_track_ids = set()
        
        for track in tracks:
            track_id = int(track[4])
            current_track_ids.add(track_id)
            
            # Get detection index from ByteTrack
            detection_idx = int(track[5]) if len(track) > 5 and track[5] >= 0 else None
            
            # Build tracked bounding box (stabilized by tracker)
            tracked_bbox = BoundingBox.from_xyxy(
                int(track[0]),
                int(track[1]),
                int(track[2]),
                int(track[3]),
            )
            
            # Get confidence and landmark from matched detection
            confidence = 0.0
            landmark = None
            
            if detection_idx is not None and detection_idx < len(detections):
                matched_detection = detections[detection_idx]
                confidence = matched_detection.confidence
                
                # Associate landmark to this track
                # Transform landmark from detection bbox to tracked bbox if they differ
                if matched_detection.landmark is not None:
                    detection_bbox = matched_detection.bbox
                    
                    # Check if bbox differs significantly
                    if (abs(detection_bbox.x - tracked_bbox.x) > 2 or
                        abs(detection_bbox.y - tracked_bbox.y) > 2 or
                        abs(detection_bbox.width - tracked_bbox.width) > 2 or
                        abs(detection_bbox.height - tracked_bbox.height) > 2):
                        # Transform landmark to tracked bbox coordinates
                        landmark = matched_detection.landmark.transform_to_bbox(
                            detection_bbox, tracked_bbox
                        )
                    else:
                        # Use original landmark
                        landmark = matched_detection.landmark
            
            results.append(TrackedFaceData(
                track_id=track_id,
                bbox=tracked_bbox,
                confidence=confidence,
                detection_index=detection_idx,
                landmark=landmark,
            ))
        
        # Update active track IDs
        self._active_track_ids = current_track_ids
        
        # Find dead tracks (were active before, not active now)
        dead_track_ids = list(self._previous_track_ids - current_track_ids)
        
        return results, dead_track_ids
    
    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()
        self._active_track_ids.clear()
        self._previous_track_ids.clear()
    
    def get_active_track_ids(self) -> List[int]:
        """Get list of currently active track IDs."""
        return list(self._active_track_ids)
