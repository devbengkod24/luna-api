"""Vision adapters (face recognition, vector store).

Tracking, emotion and active-speaker detection used to live here. They moved to
luna-edge: every one of them needs the video frames themselves, and frames no
longer cross the network. What stays is the part that needs the face *index* —
enrolment and matching — which is cloud-side because the index is.
"""

from .face_recognition_service import FaceRecognitionService
from .face_store import FaceStore

__all__ = [
    "FaceRecognitionService",
    "FaceStore",
]
