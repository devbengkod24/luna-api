"""SFace embeddings, so enrolment and the kiosk speak the same language.

The kiosk embeds with OpenCV's SFace and gets 128 numbers. Enrolment used to
embed with InsightFace's ArcFace and get 512. Nothing matched, ever, and it
failed silently: a vector of the wrong length is not a poor score, it is a
question the index cannot be asked. This module is the enrolment half of
fixing that.

Licence is the other half of the reason. SFace ships under Apache-2.0 and the
InsightFace weights it replaces are non-commercial, which is a problem for a
kiosk that stands in a university lobby.

Detection is YuNet, the OpenCV detector SFace was designed alongside: it emits
the box and five keypoints this needs without anything in between.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SFaceEmbedder:
    """Five keypoints and a frame in, 128 numbers out."""

    def __init__(self, model_path: str):
        import cv2

        self.model_path = model_path
        self._recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        logger.info("SFace embedder initialized: %s", model_path)

    @staticmethod
    def canonical(keypoints) -> list[tuple[float, float]] | None:
        """The five points in the order the aligner expects, decided by
        POSITION and never by the detector's own left/right labels.

        Those labels are relative to the subject, not the camera, and every
        library disagrees about which it means. Getting it wrong mirrors the
        alignment, which shifts the whole embedding and raises no error at all
        — enrolment and the kiosk would each produce a stable vector and the
        two would simply never match. Sorting by x is the same rule the edge
        applies to its MediaPipe landmarks, which is what makes the two halves
        comparable.
        """
        points = [(float(x), float(y)) for (x, y) in keypoints]
        if len(points) != 5:
            return None
        eyes = sorted(points[:2], key=lambda p: p[0])
        nose = points[2]
        mouth = sorted(points[3:], key=lambda p: p[0])
        return [eyes[0], eyes[1], nose, mouth[0], mouth[1]]

    def embed(self, image: np.ndarray, keypoints) -> list[float] | None:
        """Align with the five points, then embed.

        `alignCrop` wants OpenCV's detection row: a box followed by the five
        points, in pixels. Only the points affect the similarity transform, so
        the box is derived from them rather than passed separately — one source
        of truth for where the face is, and no chance of the two disagreeing.
        """
        ordered = self.canonical(keypoints)
        if ordered is None or image is None:
            return None
        try:
            xs = [p[0] for p in ordered]
            ys = [p[1] for p in ordered]
            row = np.array(
                [[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                 + [c for point in ordered for c in point]],
                dtype=np.float32,
            )
            aligned = self._recognizer.alignCrop(image, row)
            feature = self._recognizer.feature(aligned)
            return np.asarray(feature).flatten().astype(float).tolist()
        except Exception as e:
            logger.warning("SFace embedding failed: %s", e)
            return None
