"""Facebook MMS-TTS Indonesian adapter implementation."""
import base64
import io
import logging
import os
from typing import Optional

import numpy as np
import torch
from transformers import VitsModel, AutoTokenizer
from scipy.io import wavfile

from src.ports.services import ITTSService

logger = logging.getLogger(__name__)


class MMSTTSService(ITTSService):
    """Text-to-Speech service using Facebook MMS-TTS Indonesian model."""
    
    MODEL_REPO = "facebook/mms-tts-ind"
    
    def __init__(self):
        """Initialize the TTS adapter."""
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
    
    async def synthesize(self, text: str, character: int) -> str:
        """Synthesize text to speech audio.
        
        Args:
            text: Text to synthesize in Indonesian
            character: Character ID (currently unused for MMS-TTS)
            
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


class ChatterboxSTTSService(ITTSService):
    """Text-to-Speech service using Chatterbox-TTS Indonesian fine-tune.

    Model source:
    - https://huggingface.co/grandhigh/Chatterbox-TTS-Indonesian
    """

    MODEL_REPO = "grandhigh/Chatterbox-TTS-Indonesian"
    CHECKPOINT_FILENAME = "t3_cfg.safetensors"

    def __init__(
        self,
        device: Optional[str] = None,
        voice_clone_dir: str = "storage/voice",
    ) -> None:
        # Chatterbox bisa jalan di CPU, tapi CUDA sangat disarankan.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.voice_clone_dir = voice_clone_dir

        self.model = None
        self.sr: Optional[int] = None
        self._load_model()
    def _character_prompt_path(self, character: int) -> Optional[str]:
        """Resolve audio prompt path for a given character.

        Convention:
        - `storage/voice/0.wav` for Character.RUSSELL (0)
        - `storage/voice/1.wav` for Character.FREDRICKSEN (1)
        """
        try:
            filename = f"{int(character)}.wav"
        except Exception:
            return None
        path = os.path.join(self.voice_clone_dir, filename)
        return path if os.path.isfile(path) else None


    def _load_model(self) -> None:
        """Load base Chatterbox model and apply Indonesian fine-tune checkpoint."""
        try:
            logger.info("Initializing Chatterbox-TTS on device: %s", self.device)

            from chatterbox.tts import ChatterboxTTS  # type: ignore
            import perth  # type: ignore
           
            perth.PerthImplicitWatermarker = perth.DummyWatermarker

            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.sr = int(getattr(self.model, "sr", 24000))

            # Apply fine-tune weights (if deps are available).
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
                from safetensors.torch import load_file  # type: ignore

                checkpoint_path = hf_hub_download(
                    repo_id=self.MODEL_REPO,
                    filename=self.CHECKPOINT_FILENAME,
                )
                state = load_file(checkpoint_path, device="cpu")

                # The HF card loads into `model.t3`.
                if not hasattr(self.model, "t3"):
                    raise RuntimeError("ChatterboxTTS model missing attribute `t3`.")
                self.model.t3.load_state_dict(state)

                if self.device == "cuda":
                    torch.cuda.empty_cache()

                logger.info(
                    "Chatterbox-TTS Indonesian checkpoint loaded successfully (sr=%s).",
                    self.sr,
                )
            except Exception as e:
                # Keep service usable (base model) even if checkpoint can't be applied.
                logger.warning(
                    "Failed to apply Indonesian checkpoint; using base Chatterbox model. Error: %s",
                    e,
                )

        except Exception as e:
            logger.error("Failed to initialize Chatterbox-TTS: %s", e)
            raise

    def _audio_to_base64(self, audio_array: np.ndarray, sample_rate: int) -> str:
        # Normalize defensively, then encode as 16-bit PCM WAV.
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("Generated empty audio.")

        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak

        audio_int16 = (audio * 32767.0).astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    async def synthesize(self, text: str, character: int) -> str:
        if self.model is None or self.sr is None:
            raise RuntimeError("Chatterbox TTS model not initialized")

        try:
            logger.info("Chatterbox synthesizing text: %s...", (text or "")[:50])

            # `generate` usually returns a torch Tensor waveform.
            prompt = self._character_prompt_path(character)
            if prompt:
                wav = self.model.generate(text, audio_prompt_path=prompt)
            else:
                wav = self.model.generate(text)

            # Support torch Tensor / numpy.
            if hasattr(wav, "detach"):
                audio_np = wav.detach().cpu().numpy()
            else:
                audio_np = np.asarray(wav)

            audio_np = np.squeeze(audio_np)
            return self._audio_to_base64(audio_np, int(self.sr))

        except Exception as e:
            logger.error("Failed to synthesize with Chatterbox: %s", e)
            raise