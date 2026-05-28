"""Emotic Two-Stream ONNX model wrapper (26 contextual emotion categories + VAD)."""

import numpy as np
import onnxruntime as ort
import cv2

EMOTIONS = [
    "Peace", "Affection", "Esteem", "Anticipation", "Engagement",
    "Confidence", "Pleasure", "Excitement", "Surprise", "Sympathy",
    "Doubt/Confusion", "Disconnect", "Suffering", "Pain", "Sadness",
    "Disquietment", "Shame", "Fear", "Anger", "Aversion",
    "Contempt", "Humiliation", "Fatigue", "Boredom", "Sensory Pleasure", "Hunger",
]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EmoticModel:
    """Two-stream Emotic ONNX model: full-image context + body crop → 26 categories + VAD."""

    INPUT_SIZE = 224

    def __init__(self, model_path: str):
        available = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        self.session = ort.InferenceSession(model_path, providers=providers)

        input_names  = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]
        self.context_name = "context" if "context" in input_names else input_names[0]
        self.body_name    = "body"    if "body"    in input_names else input_names[1]
        self.cat_name     = "cats"    if "cats"    in output_names else output_names[0]
        self.vad_name     = "vad"     if "vad"     in output_names else output_names[1]

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        return normalized.transpose(2, 0, 1)[np.newaxis, ...]

    def predict(
        self,
        bgr_full_img: np.ndarray,
        bgr_body_crop: np.ndarray,
        threshold: float = 0.5,
    ) -> dict | None:
        """Run inference. Returns None if either input is invalid."""
        if bgr_full_img is None or bgr_body_crop is None:
            return None
        if bgr_full_img.size == 0 or bgr_body_crop.size == 0:
            return None

        ctx_tensor  = self._preprocess(bgr_full_img)
        body_tensor = self._preprocess(bgr_body_crop)

        outputs = self.session.run(
            [self.cat_name, self.vad_name],
            {self.context_name: ctx_tensor, self.body_name: body_tensor},
        )

        cat_logits = outputs[0][0]
        vad_values = outputs[1][0]

        probs = 1 / (1 + np.exp(-cat_logits))
        emotions = sorted(
            [{"label": EMOTIONS[i], "confidence": float(p)} for i, p in enumerate(probs) if p > threshold],
            key=lambda x: x["confidence"],
            reverse=True,
        )

        return {
            "emotions":  emotions,
            "valence":   float(np.clip(vad_values[0], -1.0, 1.0)),
            "arousal":   float(np.clip(vad_values[1], -1.0, 1.0)),
            "dominance": float(np.clip(vad_values[2], -1.0, 1.0)),
        }
