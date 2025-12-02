"""InsightFace adapter for face detection and recognition.

Uses SCRFD for detection (fast) and ArcFace for recognition (accurate).
Default model: buffalo_sc (SCRFD + ArcFace, optimized for speed).
"""
import numpy as np
from typing import List, Optional, Tuple

import cv2
from insightface.app import FaceAnalysis

from src.config import settings
from src.ports.vision import IFaceRecognitionAdapter
from src.domain.entities.vision import DetectedFace, BoundingBox


class InsightFaceAdapter(IFaceRecognitionAdapter):
    """Face recognition adapter using InsightFace with SCRFD + ArcFace."""
    
    def __init__(
        self,
        model_name: str = None,
        ctx_id: int = None,
        det_size: int = None,
    ):
        """Initialize InsightFace model.
        
        Args:
            model_name: Model pack name. Options:
                - buffalo_sc: SCRFD + ArcFace (fast, recommended)
                - buffalo_s: Small model (faster, less accurate)
                - buffalo_l: Large model (slower, most accurate)
            ctx_id: Execution context (-1 for CPU, 0+ for GPU)
            det_size: Detection input size (320 faster, 640 more accurate)
        """
        # Use settings defaults if not provided
        self.model_name = model_name or settings.face_recognition_model
        self.ctx_id = ctx_id if ctx_id is not None else settings.face_recognition_ctx_id
        self.det_size = det_size or settings.face_detection_size
        self.similarity_threshold = settings.face_similarity_threshold
        
        # Select execution providers based on context
        if self.ctx_id >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        # Initialize FaceAnalysis with specified model
        self.app = FaceAnalysis(
            name=self.model_name,
            providers=providers,
            allowed_modules=["detection", "recognition"],  # Only load needed modules
        )
        
        # Prepare model with detection size
        self.app.prepare(
            ctx_id=self.ctx_id,
            det_size=(self.det_size, self.det_size),
        )
        
        print(f"InsightFace initialized: model={self.model_name}, "
              f"ctx_id={self.ctx_id}, det_size={self.det_size}")
    
    def _decode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Decode image bytes to numpy array (BGR format).
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            
        Returns:
            Optional[np.ndarray]: Image in BGR format or None if failed
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    def detect_faces(self, image_bytes: bytes) -> List[DetectedFace]:
        """Detect all faces in an image and extract embeddings.
        
        Uses SCRFD for fast detection and ArcFace for embedding extraction.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            
        Returns:
            List[DetectedFace]: List of detected faces with embeddings
        """
        img = self._decode_image(image_bytes)
        if img is None:
            return []
        
        # Detect faces using SCRFD
        faces = self.app.get(img)
        
        results = []
        for face in faces:
            # Get bounding box (bbox is [x1, y1, x2, y2])
            bbox = face.bbox.astype(int)
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
            
            # Get ArcFace embedding (512-dimensional vector)
            embedding = face.embedding.tolist() if face.embedding is not None else []
            
            # Get detection confidence score
            confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.0
            
            results.append(DetectedFace(
                box=BoundingBox(x=x, y=y, width=w, height=h),
                embedding=embedding,
                confidence=confidence,
            ))
        
        return results
    
    def get_embedding(self, image_bytes: bytes) -> Optional[List[float]]:
        """Get face embedding from an image with single face.
        
        Extracts ArcFace embedding (512-dimensional vector).
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            
        Returns:
            Optional[List[float]]: Face embedding vector or None if no face detected
        """
        faces = self.detect_faces(image_bytes)
        
        if not faces:
            return None
        
        # Return the largest face's embedding (most likely the main subject)
        if len(faces) > 1:
            faces.sort(
                key=lambda f: f.box.width * f.box.height,
                reverse=True,
            )
        
        return faces[0].embedding if faces[0].embedding else None
    
    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two ArcFace embeddings.
        
        Args:
            embedding1: First embedding vector (512-dim)
            embedding2: Second embedding vector (512-dim)
            
        Returns:
            float: Cosine similarity score (-1 to 1, higher is more similar)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
        
        return similarity
    
    def is_same_person(
        self,
        embedding1: List[float],
        embedding2: List[float],
        threshold: float = None,
    ) -> Tuple[bool, float]:
        """Check if two embeddings belong to the same person.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (default from settings)
            
        Returns:
            Tuple[bool, float]: (is_same_person, similarity_score)
        """
        threshold = threshold or self.similarity_threshold
        similarity = self.compute_similarity(embedding1, embedding2)
        return similarity >= threshold, similarity
    
    def get_face_area(self, image_bytes: bytes) -> Optional[bytes]:
        """Extract the face region from an image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Optional[bytes]: Cropped face image as JPEG bytes
        """
        img = self._decode_image(image_bytes)
        if img is None:
            return None
        
        faces = self.app.get(img)
        if not faces:
            return None
        
        # Get the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Add padding (10% of face size)
        padding_x = int((x2 - x1) * 0.1)
        padding_y = int((y2 - y1) * 0.1)
        h, w = img.shape[:2]
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(w, x2 + padding_x)
        y2 = min(h, y2 + padding_y)
        
        # Crop face
        face_img = img[y1:y2, x1:x2]
        
        # Encode to JPEG with high quality
        _, buffer = cv2.imencode(".jpg", face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
    
    def get_aligned_face(self, image_bytes: bytes) -> Optional[bytes]:
        """Get aligned and normalized face image.
        
        This is useful for face registration as it provides consistent face representation.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Optional[bytes]: Aligned face image (112x112) as JPEG bytes
        """
        img = self._decode_image(image_bytes)
        if img is None:
            return None
        
        faces = self.app.get(img)
        if not faces:
            return None
        
        # Get the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Get aligned face from insightface (112x112 normalized)
        if hasattr(face, 'normed_embedding'):
            # Get the aligned face image used for embedding
            aligned = face.aimg if hasattr(face, 'aimg') else None
            if aligned is not None:
                _, buffer = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return buffer.tobytes()
        
        # Fallback to regular cropping
        return self.get_face_area(image_bytes)
