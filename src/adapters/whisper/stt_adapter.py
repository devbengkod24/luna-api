"""STT adapter implementation using faster-whisper.

This adapter provides high-quality speech-to-text transcription using
the faster-whisper library with CTranslate2 backend for improved performance.
"""

import base64
import tempfile
import os
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from src.ports.speech import ISTTAdapter
from src.config import settings


class STTAdapter(ISTTAdapter):
    """STT adapter using faster-whisper."""
    
    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[int] = None,
        compute_type: Optional[str] = "int8"
    ):
        """Initialize STT adapter with faster-whisper.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v1, large-v2, large-v3)
                       Defaults to settings.whisper_model
            device: Device to run model on ('cuda', 'cpu', or 'auto')
                   Defaults to 'cuda' if gpu_device >= 0, else 'cpu'
            compute_type: Computation type for quantization (int8, float16, float32)
        """
        self.model_size = model_size
        
        # Determine device based on gpu_device setting
        if device > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.compute_type = compute_type
        
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            
            print(
                f"STT Adapter initialized: model={self.model_size}, "
                f"device={self.device}, compute_type={self.compute_type}"
            )
        except Exception as e:
            print(f"Error initializing STT adapter: {e}")
            raise RuntimeError(f"Failed to initialize STT adapter: {str(e)}")
    
    async def transcribe(self, speech_base64: str) -> str:
        """Transcribe speech audio to text.
        
        Args:
            speech_base64: Base64-encoded speech audio data
            
        Returns:
            str: Transcribed text
            
        Raises:
            ValueError: If base64 data is invalid
            RuntimeError: If transcription fails
        """
        temp_file = None
        
        try:
            # Decode base64 audio data
            try:
                audio_data = base64.b64decode(speech_base64)
            except Exception as e:
                raise ValueError(f"Invalid base64 audio data: {str(e)}")
            
            if not audio_data:
                raise ValueError("Empty audio data")
            
            # Create temporary file for audio
            # faster-whisper requires file input
            with tempfile.NamedTemporaryFile(
                mode='wb',
                suffix='.wav',
                delete=False
            ) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Transcribe audio
            try:
                segments, info = self.model.transcribe(
                    temp_file_path,
                    language="id",  # Indonesian language
                    beam_size=5,
                    vad_filter=True,  # Voice activity detection
                )
                
                # Combine all segments into full transcription
                transcript_parts = []
                for segment in segments:
                    transcript_parts.append(segment.text.strip())
                
                transcript = " ".join(transcript_parts).strip()
                
                if not transcript:
                    raise RuntimeError("Transcription resulted in empty text")
                
                print(f"Transcription completed: {len(transcript)} characters")
                return transcript
                
            except Exception as e:
                print(f"Transcription error: {e}")
                raise RuntimeError(f"Transcription failed: {str(e)}")
            
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file: {e}")
