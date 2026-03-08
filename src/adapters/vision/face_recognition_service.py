"""InsightFace adapter for face detection and recognition.

Uses SCRFD for detection (fast) and ArcFace for recognition (accurate).
Default model: buffalo_sc (SCRFD + ArcFace, optimized for speed).
"""
import numpy as np
from typing import List, Optional, Tuple

import cv2
from insightface.app import FaceAnalysis
from skimage import transform as trans

from src.config import settings
from src.ports import IFaceRecognitionService
from src.domain.entities.vision import (
    DetectedFace, 
    BoundingBox, 
    FaceDetection, 
    Landmark,
)

# Standard ArcFace alignment template (112x112)
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def estimate_norm(landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Estimate affine transformation matrix for face alignment.
    
    Args:
        landmark: 5-point facial landmarks (5, 2)
        image_size: Output image size
        
    Returns:
        Affine transformation matrix (2, 3)
    """
    assert landmark.shape == (5, 2)
    
    dst = ARCFACE_DST.copy()
    if image_size != 112:
        dst = dst * (image_size / 112.0)
    
    tform = trans.SimilarityTransform()
    tform.estimate(landmark, dst)
    M = tform.params[0:2, :]
    
    return M


class FaceRecognitionService(IFaceRecognitionService):
    """Face recognition adapter using InsightFace with SCRFD + ArcFace.
    
    Supports two modes:
    1. Combined detection + recognition (legacy, for registration)
    2. Separate detection and recognition (for streaming pipeline)
    """
    
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
            allowed_modules=["detection", "recognition"],
        )
        
        # Prepare model with detection size
        self.app.prepare(
            ctx_id=self.ctx_id,
            det_size=(self.det_size, self.det_size),
        )
        
        # Store recognition model reference for direct embedding extraction
        self._rec_model = None
        for model in self.app.models.values():
            if hasattr(model, 'get_feat'):
                self._rec_model = model
                break
        
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

    def _crop_face_from_image(
        self, img: np.ndarray, box: BoundingBox
    ) -> Optional[bytes]:
        """Crop face from already decoded image (internal helper)."""
        # Add padding
        padding_x = int(box.width * 0.1)
        padding_y = int(box.height * 0.1)
        
        h, w = img.shape[:2]
        x1 = max(0, box.x - padding_x)
        y1 = max(0, box.y - padding_y)
        x2 = min(w, box.x + box.width + padding_x)
        y2 = min(h, box.y + box.height + padding_y)
        
        # Crop
        face_img = img[y1:y2, x1:x2]
        
        # Encode to JPEG
        _, buffer = cv2.imencode(".jpg", face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
    
    # =========================================================================
    # Legacy method (combined detection + recognition, for registration)
    # =========================================================================
    
    def detect_faces(self, image_bytes: bytes, include_crops: bool = False) -> List[DetectedFace]:
        """Detect all faces in an image and extract embeddings.
        
        Uses SCRFD for fast detection and ArcFace for embedding extraction.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            include_crops: Whether to include cropped face images in the result
        Returns:
            List[DetectedFace]: List of detected faces with embeddings
        """
        img = self._decode_image(image_bytes)
        if img is None:
            return []
        
        # Detect faces using SCRFD + extract embeddings with ArcFace
        faces = self.app.get(img)
        
        results = []
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            box = BoundingBox.from_xyxy(x1, y1, x2, y2)
            
            embedding = face.embedding.tolist() if face.embedding is not None else []
            
            confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.0
            
            # Extract landmark
            landmark = None
            if hasattr(face, 'kps') and face.kps is not None:
                landmark = Landmark.from_array(face.kps)
            
            crop_bytes = None
            if include_crops:
                crop_bytes = self._crop_face_from_image(img, box)
            
            results.append(DetectedFace(
                bbox=box,
                embedding=embedding,
                confidence=confidence,
                crop_bytes=crop_bytes,
                detection_index=idx,
                landmark=landmark,
            ))
        
        return results
    
    # =========================================================================
    # New methods (separate detection and recognition, for streaming pipeline)
    # =========================================================================
    
    def detect_faces_only(self, image_bytes: bytes) -> List[FaceDetection]:
        """Detect faces WITHOUT extracting embeddings (for streaming pipeline).
        
        This is faster as it only runs SCRFD detection without ArcFace.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            
        Returns:
            List[FaceDetection]: Detected faces with bbox, confidence, landmark
        """
        img = self._decode_image(image_bytes)
        if img is None:
            return []
        
        return self.detect_faces_only_from_array(img)
    
    def detect_faces_only_from_array(self, img: np.ndarray) -> List[FaceDetection]:
        """Detect faces from numpy array WITHOUT extracting embeddings.
        
        Args:
            img: Image as BGR numpy array
            
        Returns:
            List[FaceDetection]: Detected faces with bbox, confidence, landmark
        """
        # Use detection model only (SCRFD)
        det_model = self.app.models.get('detection')
        if det_model is None:
            # Fallback: use full pipeline but ignore embeddings
            faces = self.app.get(img)
        else:
            # Direct detection without recognition
            bboxes, kpss = det_model.detect(img, max_num=0, metric='default')
            faces = []
            for i in range(len(bboxes)):
                face = type('Face', (), {})()
                face.bbox = bboxes[i, :4]
                face.det_score = bboxes[i, 4]
                face.kps = kpss[i] if kpss is not None else None
                faces.append(face)
        
        results = []
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Skip invalid bboxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            box = BoundingBox.from_xyxy(x1, y1, x2, y2)
            confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.0
            
            # Extract landmark
            landmark = None
            if hasattr(face, 'kps') and face.kps is not None:
                landmark = Landmark.from_array(face.kps)
            
            results.append(FaceDetection(
                bbox=box,
                confidence=confidence,
                landmark=landmark,
                detection_index=idx,
            ))
        
        return results
    
    def align_face(
        self,
        image: np.ndarray,
        landmark: Landmark,
        output_size: Tuple[int, int] = (112, 112),
    ) -> Optional[np.ndarray]:
        """Align face using landmarks (for recognition).
        
        Uses affine transformation based on 5-point landmarks.
        Standard ArcFace alignment to 112x112.
        
        Args:
            image: Full frame image (BGR numpy array)
            landmark: 5-point facial landmarks
            output_size: Output aligned face size (default 112x112 for ArcFace)
            
        Returns:
            Aligned face image or None if alignment fails
        """
        try:
            # Get affine transformation matrix
            M = estimate_norm(landmark.points, output_size[0])
            
            # Apply affine transformation
            aligned = cv2.warpAffine(
                image, 
                M, 
                output_size, 
                borderValue=0.0
            )
            
            return aligned
            
        except Exception as e:
            print(f"Error aligning face: {e}")
            return None
    
    def extract_embedding(self, aligned_face: np.ndarray) -> Optional[List[float]]:
        """Extract embedding from aligned face image.
        
        Args:
            aligned_face: Aligned face image (112x112 BGR)
            
        Returns:
            512-dimensional embedding vector or None if extraction fails
        """
        try:
            if self._rec_model is None:
                print("Recognition model not available")
                return None
            
            # Ensure correct size
            if aligned_face.shape[:2] != (112, 112):
                aligned_face = cv2.resize(aligned_face, (112, 112))
            
            # Extract embedding using ArcFace model
            embedding = self._rec_model.get_feat(aligned_face)
            
            # Normalize embedding
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.tolist()
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def crop_face(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        padding: float = 0.1,
    ) -> Optional[np.ndarray]:
        """Crop face region from image with padding.
        
        Args:
            image: Full frame image (BGR numpy array)
            bbox: Face bounding box
            padding: Padding ratio (default 10%)
            
        Returns:
            Cropped face image or None if cropping fails
        """
        try:
            h, w = image.shape[:2]
            
            # Calculate padding
            pad_x = int(bbox.width * padding)
            pad_y = int(bbox.height * padding)
            
            # Apply padding with bounds checking
            x1 = max(0, bbox.x - pad_x)
            y1 = max(0, bbox.y - pad_y)
            x2 = min(w, bbox.x + bbox.width + pad_x)
            y2 = min(h, bbox.y + bbox.height + pad_y)
            
            # Crop
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return None
            
            return cropped
            
        except Exception as e:
            print(f"Error cropping face: {e}")
            return None
    
    def encode_to_bytes(self, image: np.ndarray, quality: int = 95) -> Optional[bytes]:
        """Encode numpy array image to JPEG bytes.
        
        Args:
            image: Image as BGR numpy array
            quality: JPEG quality (0-100)
            
        Returns:
            JPEG bytes or None if encoding fails
        """
        try:
            _, buffer = cv2.imencode(
                ".jpg", 
                image, 
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            return buffer.tobytes()
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def decode_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Decode image bytes to numpy array (public method).
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            
        Returns:
            Image as BGR numpy array or None if decoding fails
        """
        return self._decode_image(image_bytes)
