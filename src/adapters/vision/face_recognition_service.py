"""Face enrolment: YuNet finds the face, SFace turns it into 128 numbers.

Both halves are OpenCV's own, and both are permissively licensed — YuNet
BSD-3, SFace Apache-2.0. They replace InsightFace, whose `buffalo_sc` pack is
research-only as a whole, detection weights included, which is not a licence a
kiosk standing in a university lobby can rely on.

The pairing is not a coincidence. YuNet emits exactly the row SFace's
`alignCrop` consumes — a box followed by five keypoints — so nothing has to be
translated between them.

Measured against the SCRFD it replaces, on thirty enrolment photos at the same
pixel budget: both found every face, YuNet ran 6.1 ms against 3.9 ms median,
and agreement with the kiosk's own MediaPipe pipeline went from 0.978 to 0.965
median cosine. Both numbers sit far above the 0.35 match threshold, and a
detector that runs once per person when their photo is uploaded has no
meaningful speed budget to blow.
"""

import asyncio
import logging

import cv2
import numpy as np

from src.config import settings
from src.domain.entities.vision import BoundingBox, DetectedFace, FaceDetection, Landmark
from src.domain.image import Image
from src.ports.vision import IFaceRecognitionService

logger = logging.getLogger(__name__)

# ArcFace alignment template (112×112). Kept because `align()` is part of the
# port and still used to produce crops; the embedding no longer goes through it.
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _landmark_to_array(landmark: Landmark) -> np.ndarray:
    return np.array(landmark.points, dtype=np.float32)


def _estimate_norm(landmark: Landmark, size: int = 112) -> np.ndarray | None:
    """The similarity transform from five keypoints onto the template.

    OpenCV rather than scikit-image, and that is the difference between one
    dependency and four. `skimage.transform.SimilarityTransform` needs scipy,
    which came in free while InsightFace was installed and cost 40 MB the
    moment it left. `estimateAffinePartial2D` computes the same class of
    transform — rotation, uniform scale, translation — and OpenCV is already
    here for everything else.
    """
    dst = _ARCFACE_DST.copy() * (size / 112.0) if size != 112 else _ARCFACE_DST.copy()
    matrix, _ = cv2.estimateAffinePartial2D(
        _landmark_to_array(landmark), dst, method=cv2.LMEDS
    )
    return matrix


def _wrap_image(arr: np.ndarray | None) -> Image | None:
    if arr is None:
        return None
    h, w = arr.shape[:2]
    return Image(raw=arr, width=int(w), height=int(h))


def _unwrap(image: Image) -> np.ndarray:
    return image.raw


class FaceRecognitionService(IFaceRecognitionService):
    """YuNet detection + SFace embedding, both from OpenCV."""

    def __init__(self, det_size: int | None = None):
        self.det_size = det_size or settings.face_detection_size
        self._detector = cv2.FaceDetectorYN.create(
            settings.yunet_model_path,
            "",
            (self.det_size, self.det_size),
            settings.face_detection_threshold,
            settings.face_detection_nms,
            5000,
        )
        self._sface = None
        try:
            from src.adapters.vision.sface_embedder import SFaceEmbedder

            self._sface = SFaceEmbedder(settings.sface_model_path)
        except Exception as e:
            # Loud in the log, soft at runtime: the service still detects, and
            # every enrolment is refused rather than stored as an empty vector
            # that would look enrolled and never match.
            logger.error(
                "SFace could not be loaded (%s) — face enrolment is disabled. "
                "Expected the model at %s.", e, settings.sface_model_path,
            )
        logger.info(
            "Face pipeline initialized: YuNet det_size=%d + SFace",
            self.det_size,
        )

    # --- detection -------------------------------------------------------

    def _detect_raw(self, img: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Every face as (5×2 keypoints in ORIGINAL pixels, score).

        Detected on a downscaled copy and scaled back. Running YuNet at the
        full resolution of a phone photo took 23 ms against 6 ms here and
        found FEWER faces — the model is trained around a fixed input scale,
        and a 3000 px portrait puts the face far outside it.
        """
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
        scale = self.det_size / float(max(w, h))
        small = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        sh, sw = small.shape[:2]
        try:
            self._detector.setInputSize((sw, sh))
            _, faces = self._detector.detect(small)
        except Exception as e:
            logger.warning("YuNet detection failed: %s", e)
            return []
        if faces is None:
            return []

        out = []
        for row in faces:
            # YuNet row: x, y, w, h, then five keypoints, then score.
            kps = np.array(row[4:14], dtype=np.float32).reshape(5, 2) / scale
            out.append((kps, float(row[14])))
        return out

    @staticmethod
    def _detection_from(kps: np.ndarray, score: float, idx: int) -> FaceDetection | None:
        """The box is derived from the keypoints rather than taken from the
        detector, so the box and the points can never describe different faces."""
        xs, ys = kps[:, 0], kps[:, 1]
        pad_x = (xs.max() - xs.min()) * 0.6
        pad_y = (ys.max() - ys.min()) * 0.8
        x1, y1 = int(max(0, xs.min() - pad_x)), int(max(0, ys.min() - pad_y))
        x2, y2 = int(xs.max() + pad_x), int(ys.max() + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return FaceDetection(
            bbox=BoundingBox.from_xyxy(x1, y1, x2, y2),
            confidence=score,
            landmark=Landmark.from_xy_pairs(kps),
            detection_index=idx,
        )

    # --- sync workers ----------------------------------------------------

    def _decode_sync(self, image_bytes: bytes) -> np.ndarray | None:
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning("decode failed: %s", e)
            return None

    def _detect_from_array_sync(self, img: np.ndarray) -> list[FaceDetection]:
        found = []
        for idx, (kps, score) in enumerate(self._detect_raw(img)):
            det = self._detection_from(kps, score, idx)
            if det is not None:
                found.append(det)
        return found

    def _detect_from_bytes_sync(self, image_bytes: bytes) -> list[FaceDetection]:
        img = self._decode_sync(image_bytes)
        return self._detect_from_array_sync(img) if img is not None else []

    def _detect_with_embedding_sync(self, image_bytes: bytes) -> list[DetectedFace]:
        img = self._decode_sync(image_bytes)
        if img is None:
            return []
        results = []
        for idx, (kps, score) in enumerate(self._detect_raw(img)):
            det = self._detection_from(kps, score, idx)
            if det is None:
                continue
            embedding = self._sface.embed(img, kps) if self._sface else None
            if not embedding:
                logger.warning(
                    "Face %d detected but not embedded — skipping rather than "
                    "storing an empty vector nothing can ever match.", idx,
                )
                continue
            results.append(DetectedFace(
                bbox=det.bbox,
                embedding=embedding,
                confidence=det.confidence,
                detection_index=idx,
                landmark=det.landmark,
            ))
        return results

    def _align_sync(
        self,
        image: np.ndarray,
        landmark: Landmark,
        output_size: tuple[int, int] = (112, 112),
    ) -> np.ndarray | None:
        try:
            M = _estimate_norm(landmark, output_size[0])
            if M is None:
                return None
            return cv2.warpAffine(image, M, output_size, borderValue=0.0)
        except Exception as e:
            logger.warning("align failed: %s", e)
            return None

    def _crop_sync(
        self, image: np.ndarray, bbox: BoundingBox, padding: float = 0.1
    ) -> np.ndarray | None:
        try:
            h, w = image.shape[:2]
            pad_x, pad_y = int(bbox.width * padding), int(bbox.height * padding)
            x1 = max(0, bbox.x - pad_x)
            y1 = max(0, bbox.y - pad_y)
            x2 = min(w, bbox.x + bbox.width + pad_x)
            y2 = min(h, bbox.y + bbox.height + pad_y)
            if x2 <= x1 or y2 <= y1:
                return None
            return image[y1:y2, x1:x2].copy()
        except Exception as e:
            logger.warning("crop failed: %s", e)
            return None

    # --- port ------------------------------------------------------------

    async def decode(self, image_bytes: bytes) -> Image | None:
        return _wrap_image(await asyncio.to_thread(self._decode_sync, image_bytes))

    async def detect_with_embedding(self, image_bytes: bytes) -> list[DetectedFace]:
        return await asyncio.to_thread(self._detect_with_embedding_sync, image_bytes)

    async def detect_from_bytes(self, image_bytes: bytes) -> list[FaceDetection]:
        return await asyncio.to_thread(self._detect_from_bytes_sync, image_bytes)

    async def detect(self, image: Image) -> list[FaceDetection]:
        return await asyncio.to_thread(self._detect_from_array_sync, _unwrap(image))

    async def align(
        self,
        image: Image,
        landmark: Landmark,
        output_size: tuple[int, int] = (112, 112),
    ) -> Image | None:
        arr = await asyncio.to_thread(self._align_sync, _unwrap(image), landmark, output_size)
        return _wrap_image(arr)

    async def embed(self, aligned_face: Image) -> list[float] | None:
        """Not the enrolment path.

        Enrolment embeds from the ORIGINAL frame plus keypoints, because SFace
        does its own alignment and feeding it a pre-aligned crop aligns twice.
        This exists only to satisfy the port.
        """
        logger.debug("embed(aligned_face) is not used by the SFace path")
        return None

    async def crop(
        self, image: Image, bbox: BoundingBox, padding: float = 0.1
    ) -> Image | None:
        arr = await asyncio.to_thread(self._crop_sync, _unwrap(image), bbox, padding)
        return _wrap_image(arr)
