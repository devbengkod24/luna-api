"""Prompt repository port — abstracts where system prompts come from."""

from abc import ABC, abstractmethod

from src.domain.entities import UserContext
from src.shared.constants import Character


class IPromptRepository(ABC):
    """Provides rendered system prompts to use cases."""

    @abstractmethod
    def get_system_prompt(
        self, character: Character, context: UserContext | None = None
    ) -> str:
        """Return a rendered system prompt for the given character + context."""
        ...

    @abstractmethod
    def get_tts_system_prompt(self) -> str:
        """Return the system prompt used to preprocess text for TTS."""
        ...

    @abstractmethod
    def build_title_prompt(self, user_input: str) -> str:
        """Return the prompt used to generate a conversation title."""
        ...
