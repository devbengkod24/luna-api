"""Chat speech use case."""
from typing import AsyncGenerator, Optional

from src.ports.speech import ISTTAdapter, ITTSAdapter
from src.domain.entities.chat import Character, UserContext
from src.use_cases.base_chat import BaseChatUseCase


class ChatSpeechUseCase(BaseChatUseCase):
    """Use case for speech-based chat with streaming response."""
    
    def __init__(
        self,
        chat_repository,
        agent_adapter,
        stt_adapter: ISTTAdapter,
        tts_adapter: ITTSAdapter,
    ):
        """Initialize use case with dependencies.
        
        Args:
            chat_repository: Repository for chat persistence
            agent_adapter: Agent adapter for response generation
            stt_adapter: STT adapter for speech transcription
            tts_adapter: TTS adapter for speech synthesis (optional)
        """
        super().__init__(chat_repository, agent_adapter)
        self.stt_adapter = stt_adapter
        self.tts_adapter = tts_adapter
    
    async def execute(
        self,
        session_uuid: Optional[str],
        speech_base64: str,
        character: Character,
        context: UserContext | None,
        enable_tts: bool = True,
    ) -> AsyncGenerator[dict, None]:
        """Execute chat speech use case with streaming.
        
        Args:
            session_uuid: Optional session UUID (if None, create new session)
            speech_base64: Base64-encoded speech audio data
            character: Character
            context: User context (emotion, name, nim) - can be None if all fields are null
            enable_tts: Whether to synthesize response to speech (default: True)
            
        Yields:
            dict: Streaming events (log, transcript, token, audio, done)
        """
        # Step 1: Transcribe speech
        yield {"type": "log", "content": "Mentranskripsi speech..."}
        transcript = await self.stt_adapter.transcribe(speech_base64)
        yield {"type": "transcript", "content": transcript}
        
        # Step 2: Execute common chat logic with transcribed text
        # Collect full response for TTS
        full_response = ""
        async for event in self._execute_chat(session_uuid, character, transcript, context):
            yield event
            # Collect tokens for TTS synthesis
            if event.get("type") == "token":
                full_response += event.get("content", "")
        
        # Step 3: Synthesize response to speech (if enabled and adapter available)
        if enable_tts and self.tts_adapter and full_response:
            try:
                yield {"type": "log", "content": "Mensintesis speech..."}
                audio_base64 = await self.tts_adapter.synthesize(full_response)
                yield {"type": "audio", "content": audio_base64}
            except Exception as e:
                yield {"type": "log", "content": f"Warning: Failed to synthesize speech: {e}"}