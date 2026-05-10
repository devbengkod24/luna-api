"""VAD multi-task ONNX model wrapper (FER: 7 classes + Valence/Arousal/Dominance)."""

import numpy as np
import onnxruntime as ort
import cv2

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EMOTIC_26_VAD = {
    "Peace": (0.75, -0.25, 0.35),
    "Affection": (0.85, 0.25, 0.30),
    "Esteem": (0.70, 0.35, 0.75),
    "Anticipation": (0.55, 0.30, 0.35),
    "Engagement": (0.50, 0.45, 0.45),
    "Confidence": (0.80, 0.40, 0.85),
    "Pleasure": (0.90, 0.35, 0.45),
    "Excitement": (0.80, 0.85, 0.55),
    "Surprise": (0.45, 0.75, -0.15),
    "Sympathy": (0.65, -0.15, 0.15),
    "Doubt/Confusion": (-0.15, 0.35, -0.45),
    "Disconnect": (-0.25, -0.35, -0.15),
    "Suffering": (-0.85, 0.45, -0.65),
    "Pain": (-0.95, 0.65, -0.55),
    "Sadness": (-0.85, -0.35, -0.65),
    "Disquietment": (-0.45, 0.35, -0.25),
    "Shame": (-0.65, 0.15, -0.75),
    "Fear": (-0.80, 0.75, -0.85),
    "Anger": (-0.85, 0.75, 0.45),
    "Aversion": (-0.75, 0.45, -0.15),
    "Contempt": (-0.55, 0.25, 0.45),
    "Humiliation": (-0.85, 0.25, -0.85),
    "Fatigue": (-0.35, -0.75, -0.25),
    "Boredom": (-0.45, -0.65, -0.35),
    "Sensory Pleasure": (0.85, 0.25, 0.40),
    "Hunger": (-0.15, 0.25, -0.15),
}


class VADModel:
    """Multi-task VAD ONNX model: 7-class FER + Valence/Arousal/Dominance output."""

    INPUT_SIZE = 224

    def __init__(self, model_path: str):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        input_names = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]
        assert "face_crop" in input_names, f"Unexpected input names: {input_names}"
        assert "cls_logits" in output_names, f"Unexpected output names: {output_names}"
        assert "vad" in output_names, f"Unexpected output names: {output_names}"

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        return normalized.transpose(2, 0, 1)[np.newaxis, ...]

    def predict(self, bgr_face_crop: np.ndarray) -> dict | None:
        """Run inference on a BGR face crop. Returns None if input is invalid."""
        if bgr_face_crop is None or bgr_face_crop.size == 0:
            return None

        tensor = self._preprocess(bgr_face_crop)
        outputs = self.session.run(["cls_logits", "vad"], {"face_crop": tensor})

        cls_logits = outputs[0][0]
        vad_values = outputs[1][0]

        exp_logits = np.exp(cls_logits - cls_logits.max())
        probs = exp_logits / exp_logits.sum()

        class_idx = int(np.argmax(probs))
        emotion = CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx])

        valence   = float(np.clip(vad_values[0], -1.0, 1.0))
        arousal   = float(np.clip(vad_values[1], -1.0, 1.0))
        dominance = float(np.clip(vad_values[2], -1.0, 1.0))

        # Valence gating: reduce false "happy" and surface suppressed "sad"
        if emotion == "happy" and valence < 0.3:
            emotion = "neutral"
            confidence = float(probs[CLASS_NAMES.index("neutral")])
        elif emotion == "neutral" and valence < -0.25:
            emotion = "sad"
            confidence = float(probs[CLASS_NAMES.index("sad")])

        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "dominance": round(dominance, 3),
        }
