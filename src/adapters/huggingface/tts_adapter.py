"""Chatterbox TTS adapter implementation."""
import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy.io import wavfile

from src.ports.speech import ITTSAdapter

logger = logging.getLogger(__name__)


class TTSAdapter(ITTSAdapter):
    """Adapter for Chatterbox TTS Indonesian model."""
    
    MODEL_REPO = "grandhigh/Chatterbox-TTS-Indonesian"
    CHECKPOINT_FILENAME = "t3_cfg.safetensors"
    
    def __init__(self, voice_clone_dir: str = "storage/voice"):
        """Initialize the TTS adapter.
        
        Args:
            voice_clone_dir: Directory containing voice cloning audio files
        """
        self.voice_clone_dir = Path(voice_clone_dir)
        self.voice_clone_dir.mkdir(parents=True, exist_ok=True)
        
        # Deteksi device (CUDA jika tersedia, fallback ke CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Chatterbox TTS on device: {self.device}")
        
        # Inisialisasi model
        self.model = None
        self.sr = None  # Sample rate
        self._load_model()
    
    def _load_model(self):
        """Load the Chatterbox TTS model with Indonesian checkpoint."""
        try:
            # Inisialisasi model di perangkat yang dipilih
            logger.info("Loading Chatterbox TTS base model...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.sr = self.model.sr
            
            # Unduh checkpoint dari Hugging Face Hub (tidak perlu token untuk model publik)
            logger.info(f"Downloading checkpoint from {self.MODEL_REPO}...")
            checkpoint_path = hf_hub_download(
                repo_id=self.MODEL_REPO, 
                filename=self.CHECKPOINT_FILENAME
            )
            
            # Muat state_dict ke dalam model
            logger.info("Loading Indonesian checkpoint...")
            t3_state = load_file(checkpoint_path, device="cpu")
            self.model.t3.load_state_dict(t3_state)
            
            # Kosongkan cache GPU jika menggunakan CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Chatterbox TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise
    
    def _get_voice_clone_file(self) -> Optional[str]:
        """Get the first available voice clone audio file from the directory.
        
        Returns:
            Path to the voice clone audio file, or None if not found
        """
        if not self.voice_clone_dir.exists():
            return None
        
        # Cari file audio dengan ekstensi yang umum
        audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
        for ext in audio_extensions:
            for audio_file in self.voice_clone_dir.glob(f'*{ext}'):
                logger.info(f"Using voice clone file: {audio_file}")
                return str(audio_file)
        
        return None
    
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
            text: Text to synthesize
            use_voice_clone: Whether to use voice cloning if available
            
        Returns:
            Base64-encoded WAV audio data
        """
        if not self.model:
            raise RuntimeError("TTS model not initialized")
        
        try:
            logger.info(f"Synthesizing text: {text[:50]}...")
            
            # Cek apakah ada file voice clone jika diminta
            audio_prompt_path = None
            if use_voice_clone:
                audio_prompt_path = self._get_voice_clone_file()
            
            # Generate audio dengan atau tanpa voice cloning
            if audio_prompt_path:
                logger.info("Using voice cloning")
                wav_audio = self.model.generate(
                    text,
                    audio_prompt_path=audio_prompt_path
                )
            else:
                logger.info("Using default voice")
                wav_audio = self.model.generate(text)
            
            # Konversi tensor ke numpy array
            audio_numpy = wav_audio.cpu().numpy() if torch.is_tensor(wav_audio) else wav_audio
            
            # Konversi ke base64
            audio_base64 = self._audio_to_base64(audio_numpy, self.sr)
            
            logger.info("Text synthesized successfully")
            return audio_base64
            
        except Exception as e:
            logger.error(f"Failed to synthesize text: {e}")
            raise