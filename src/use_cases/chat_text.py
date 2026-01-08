"""Chat text use case."""

from typing import AsyncGenerator, Optional

from src.domain.entities.chat import Character, UserContext
from src.use_cases.base_chat import BaseChatUseCase


class ChatTextUseCase(BaseChatUseCase):
    """Use case for text-based chat with streaming response."""

    async def execute(
        self,
        session_uuid: Optional[str],
        prompt: str,
        character: Character,
        context: UserContext | None,
    ) -> AsyncGenerator[dict, None]:
        """Execute chat text use case with streaming.

        Args:
            session_uuid: Optional session UUID (if None, create new session)
            prompt: User prompt
            character: Character
            context: User context (emotion, name, nim) - can be None if all fields are null

        Yields:
            dict: Streaming events (log, token, done)
        """
        async for event in self._execute_chat(session_uuid, character, prompt, context):
            yield event
