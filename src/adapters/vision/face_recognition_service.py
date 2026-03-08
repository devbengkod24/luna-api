"""InsightFace adapter — SCRFD detection + ArcFace recognition."""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from skimage import transform as trans

from src.config import settings
from src.domain.entities.vision import BoundingBox, DetectedFace, FaceDetection, Landmark
from src.ports.vision import IFaceRecognitionService

logger = logging.getLogger(__name__)

# ArcFace alignment template (112×112)
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _estimate_norm(landmark: np.ndarray, size: int = 112) -> np.ndarray:
    dst = _ARCFACE_DST.copy() * (size / 112.0) if size != 112 else _ARCFACE_DST.copy()
    tform = trans.SimilarityTransform()
    tform.estimate(landmark, dst)
    return tform.params[0:2, :]


def _parse_face(face, idx: int) -> Optional[FaceDetection]:
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if x2 <= x1 or y2 <= y1:
        return None
    box = BoundingBox.from_xyxy(x1, y1, x2, y2)
    confidence = float(face.det_score) if hasattr(face, "det_score") else 0.0
    landmark = Landmark.from_array(face.kps) if getattr(face, "kps", None) is not None else None
    return FaceDetection(bbox=box, confidence=confidence, landmark=landmark, detection_index=idx)


class FaceRecognitionService(IFaceRecognitionService):
    """InsightFace face recognition: SCRFD detection + ArcFace embedding."""

    def __init__(self, model_name: str = None, ctx_id: int = None, det_size: int = None):
        model_name = model_name or settings.face_recognition_model
        ctx_id = ctx_id if ctx_id is not None else settings.face_recognition_ctx_id
        det_size = det_size or settings.face_detection_size

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

        self._rec_model = next(
            (m for m in self.app.models.values() if hasattr(m, "get_feat")), None
        )
        logger.info("InsightFace initialized: model=%s ctx_id=%d det_size=%d", model_name, ctx_id, det_size)

    # --- Used by face CRUD (registration) ---

    def detect_faces(self, image_bytes: bytes) -> List[DetectedFace]:
        img = self.decode_from_bytes(image_bytes)
        if img is None:
            return []
        results = []
        for idx, face in enumerate(self.app.get(img)):
            det = _parse_face(face, idx)
            if det is None:
                continue
            embedding = face.embedding.tolist() if face.embedding is not None else []
            results.append(DetectedFace(
                bbox=det.bbox,
                embedding=embedding,
                confidence=det.confidence,
                detection_index=idx,
                landmark=det.landmark,
            ))
        return results

    # --- Used by vision streaming pipeline ---

    def detect_faces_only(self, image_bytes: bytes) -> List[FaceDetection]:
        img = self.decode_from_bytes(image_bytes)
        return self.detect_faces_only_from_array(img) if img is not None else []

    def detect_faces_only_from_array(self, img: np.ndarray) -> List[FaceDetection]:
        det_model = self.app.models.get("detection")
        if det_model is not None:
            bboxes, kpss = det_model.detect(img, max_num=0, metric="default")
            raw_faces = []
            for i in range(len(bboxes)):
                face = type("Face", (), {})()
                face.bbox = bboxes[i, :4]
                face.det_score = bboxes[i, 4]
                face.kps = kpss[i] if kpss is not None else None
                raw_faces.append(face)
        else:
            raw_faces = self.app.get(img)

        return [d for d in (_parse_face(f, i) for i, f in enumerate(raw_faces)) if d is not None]

    def align_face(
        self, image: np.ndarray, landmark: Landmark, output_size: Tuple[int, int] = (112, 112)
    ) -> Optional[np.ndarray]:
        try:
            M = _estimate_norm(landmark.points, output_size[0])
            return cv2.warpAffine(image, M, output_size, borderValue=0.0)
        except Exception as e:
            logger.warning("align_face failed: %s", e)
            return None

    def extract_embedding(self, aligned_face: np.ndarray) -> Optional[List[float]]:
        if self._rec_model is None:
            return None
        try:
            if aligned_face.shape[:2] != (112, 112):
                aligned_face = cv2.resize(aligned_face, (112, 112))
            emb = self._rec_model.get_feat(aligned_face).flatten()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb.tolist()
        except Exception as e:
            logger.warning("extract_embedding failed: %s", e)
            return None

    def crop_face(
        self, image: np.ndarray, bbox: BoundingBox, padding: float = 0.1
    ) -> Optional[np.ndarray]:
        try:
            h, w = image.shape[:2]
            pad_x, pad_y = int(bbox.width * padding), int(bbox.height * padding)
            x1, y1 = max(0, bbox.x - pad_x), max(0, bbox.y - pad_y)
            x2, y2 = min(w, bbox.x + bbox.width + pad_x), min(h, bbox.y + bbox.height + pad_y)
            cropped = image[y1:y2, x1:x2]
            return cropped if cropped.size > 0 else None
        except Exception as e:
            logger.warning("crop_face failed: %s", e)
            return None

    def decode_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning("decode_from_bytes failed: %s", e)
            return None
