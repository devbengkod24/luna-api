"""Speech-to-Text adapter implementation template."""
import base64

from src.ports.audio import ISTTAdapter


class STTAdapter(ISTTAdapter):
    """STT adapter - integrate with speech-to-text service."""
    
    def __init__(self, api_key: str = None):
        """Initialize STT client.
        
        Args:
            api_key: API key for STT service (e.g., OpenAI Whisper, Google Speech-to-Text)
        """
        self.api_key = api_key
        # TODO: Initialize your STT client here
        # self.client = WhisperClient(api_key=api_key)
    
    async def transcribe(self, audio_base64: str) -> str:
        """Transcribe audio to text.
        
        Args:
            audio_base64: Base64-encoded audio data
            
        Returns:
            str: Transcribed text
        """
        # TODO: Implement audio transcription
        # 1. Decode base64 to audio bytes
        # audio_bytes = base64.b64decode(audio_base64)
        
        # 2. Call STT service
        # response = await self.client.transcribe(audio_bytes)
        
        # 3. Return transcription
        # return response.text
        
        # Placeholder
        return "Ini adalah transkrip placeholder dari audio."

