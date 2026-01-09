"""Facebook MMS-TTS Indonesian adapter implementation."""
import base64
import io
import logging
from typing import Optional

import numpy as np
import torch
from transformers import VitsModel, AutoTokenizer
from scipy.io import wavfile

from src.ports.speech import ITTSAdapter

logger = logging.getLogger(__name__)


class TTSAdapter(ITTSAdapter):
    """Adapter for Facebook MMS-TTS Indonesian model."""
    
    MODEL_REPO = "facebook/mms-tts-ind"
    
    def __init__(self, voice_clone_dir: str = "storage/voice"):
        """Initialize the TTS adapter.
        
        Args:
            voice_clone_dir: Directory for potential future voice cloning (currently unused)
        """
        # Deteksi device (CUDA jika tersedia, fallback ke CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Facebook MMS-TTS on device: {self.device}")
        
        # Inisialisasi model dan tokenizer
        self.model = None
        self.tokenizer = None
        self.sr = None  # Sample rate
        self._load_model()
    
    def _load_model(self):
        """Load the Facebook MMS-TTS Indonesian model."""
        try:
            logger.info(f"Loading Facebook MMS-TTS model from {self.MODEL_REPO}...")
            
            # Load model dan tokenizer
            self.model = VitsModel.from_pretrained(self.MODEL_REPO)
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_REPO)
            
            # Pindahkan model ke device yang sesuai
            self.model = self.model.to(self.device)
            
            # Ambil sample rate dari konfigurasi model
            self.sr = self.model.config.sampling_rate
            
            # Set model ke evaluation mode
            self.model.eval()
            
            # Kosongkan cache GPU jika menggunakan CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info(f"Facebook MMS-TTS model loaded successfully (sample rate: {self.sr} Hz)")
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise
    
    def _audio_to_base64(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Convert audio numpy array to base64-encoded WAV string.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Base64-encoded WAV audio string
        """
        # Konversi ke 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Tulis ke buffer WAV
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        buffer.seek(0)
        
        # Encode ke base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return audio_base64
    
    async def synthesize(self, text: str, use_voice_clone: bool = True) -> str:
        """Synthesize text to speech audio.
        
        Args:
            text: Text to synthesize in Indonesian
            use_voice_clone: Unused parameter (kept for interface compatibility)
            
        Returns:
            Base64-encoded WAV audio data
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("TTS model not initialized")
        
        try:
            logger.info(f"Synthesizing text: {text[:50]}...")
            
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # Pindahkan inputs ke device yang sama dengan model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate audio waveform
            with torch.no_grad():
                output = self.model(**inputs).waveform
            
            # Konversi tensor ke numpy array
            # Output shape: [1, sequence_length], kita ambil [0]
            audio_numpy = output.squeeze().cpu().numpy()
            
            # Konversi ke base64
            audio_base64 = self._audio_to_base64(audio_numpy, self.sr)
            
            logger.info("Text synthesized successfully")
            return audio_base64
            
        except Exception as e:
            logger.error(f"Failed to synthesize text: {e}")
            raise