"""Audio port interfaces."""
from abc import ABC, abstractmethod


class ISTTAdapter(ABC):
    """Interface for Speech-to-Text adapter."""
    
    @abstractmethod
    async def transcribe(self, audio_base64: str) -> str:
        """Transcribe audio to text.
        
        Args:
            audio_base64: Base64-encoded audio data
            
        Returns:
            str: Transcribed text
        """
        pass


class ITTSAdapter(ABC):
    """Interface for Text-to-Speech adapter."""
    
    @abstractmethod
    async def synthesize(self, text: str) -> str:
        """Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            
        Returns:
            str: Base64-encoded audio data
        """
        pass

