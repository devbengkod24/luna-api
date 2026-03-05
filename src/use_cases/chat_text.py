"""Chat text use case."""

from typing import AsyncGenerator, Optional

from src.domain.entities.chat import Character, UserContext
from src.use_cases.base_chat import BaseChatUseCase

from src.ports.services import ITTSAdapter

class ChatTextUseCase(BaseChatUseCase):
    """Use case for text-based chat with streaming response."""

    def __init__(
        self,
        chat_repository,
        agent_adapter,
        tts_adapter: ITTSAdapter,
    ):
        super().__init__(chat_repository, agent_adapter)
        self.tts_adapter = tts_adapter

    async def execute(
        self,
        session_uuid: Optional[str],
        prompt: str,
        character: Character,
        context: UserContext | None,
        enable_tts: bool = True,
    ) -> AsyncGenerator[dict, None]:
        """Execute chat text use case with streaming.

        Args:
            session_uuid: Optional session UUID (if None, create new session)
            prompt: User prompt
            character: Character
            context: User context (emotion, name, nim) - can be None if all fields are null
            enable_tts: Whether to synthesize response to speech (default: True)
        Yields:
            dict: Streaming events (log, token, done)
        """
        full_response = ""
        async for event in self._execute_chat(session_uuid, character, prompt, context):
            yield event

            if event.get("type") == "token":
                full_response += event.get("content", "")

        if enable_tts and self.tts_adapter and full_response:
            try:
                yield {"type": "log", "content": "Mensintesis speech..."}
                audio_base64 = await self.tts_adapter.synthesize(full_response)
                yield {"type": "audio", "content": audio_base64}
            except Exception as e:
                yield {"type": "log", "content": f"Warning: Failed to synthesize speech: {e}"}
