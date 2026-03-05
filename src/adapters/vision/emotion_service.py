from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import numpy as np
import io

from src.ports.vision import IEmotionDetectionAdapter
from src.domain.entities.vision import DetectedEmotion


class EmotionDetectionAdapter(IEmotionDetectionAdapter):
    """Emotion detection adapter using HuggingFace ViT model."""

    def __init__(self, model_name: str = 'dima806/facial_emotions_image_detection', device: str = None):
        """Initialize emotion detection model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        
        print(f"Emotion detection initialized: model={model_name}, device={self.device}")

    def _id_to_label(self, class_id: int) -> str:
        """Convert model class id to label string."""
        return self.id2label[class_id]
    
    def _predict(self, image: Image.Image) -> DetectedEmotion:
        """Internal prediction method.
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            DetectedEmotion: Detected emotion with confidence
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            
            predicted_class = int(torch.argmax(probs).item())
            confidence = float(probs[predicted_class].item())
            label = self._id_to_label(predicted_class)
            
            return DetectedEmotion(
                emotion=label,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return DetectedEmotion(emotion="neutral", confidence=0.0)

    def detect_emotion(self, face_crop_bytes: bytes) -> DetectedEmotion:
        """Detect emotion from face crop (bytes input).
        
        Args:
            face_crop_bytes: Cropped face image bytes (JPEG/PNG)
            
        Returns:
            DetectedEmotion: Detected emotion with confidence
        """
        try:
            image = Image.open(io.BytesIO(face_crop_bytes)).convert('RGB')
            return self._predict(image)
            
        except Exception as e:
            print(f"Error detecting emotion from bytes: {e}")
            return DetectedEmotion(emotion="neutral", confidence=0.0)
    
    def detect_emotion_from_array(self, face_image: np.ndarray) -> DetectedEmotion:
        """Detect emotion from face region (numpy array input).
        
        Args:
            face_image: Face image as BGR numpy array
            
        Returns:
            DetectedEmotion: Detected emotion with confidence
        """
        try:
            # Convert BGR to RGB
            rgb_image = face_image[:, :, ::-1]
            
            # Convert to PIL Image
            image = Image.fromarray(rgb_image)
            
            return self._predict(image)
            
        except Exception as e:
            print(f"Error detecting emotion from array: {e}")
            return DetectedEmotion(emotion="neutral", confidence=0.0)