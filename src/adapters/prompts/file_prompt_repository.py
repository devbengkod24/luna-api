"""Filesystem-backed prompt repository.

Loads templates once at startup and renders them with the per-request context.
"""

import logging
from functools import lru_cache
from pathlib import Path
from src.config import settings
from src.domain.entities import UserContext
from src.ports.prompts import IPromptRepository
from src.shared.constants import Character

logger = logging.getLogger(__name__)

_DEFAULT_TTS_FALLBACK = (
    "Rewrite the text into concise, natural Indonesian that is easy to read "
    "aloud by a TTS model. Preserve meaning and output only the speakable text."
)

class FilePromptRepository(IPromptRepository):

    def __init__(self, base_path: str = "src/prompts"):
        self.base_path = base_path

    def get_system_prompt(
        self, character: Character, context: UserContext | None = None
    ) -> str:
        try:
            templates = _load_prompt_templates(self.base_path)
            char_content = templates.get(character.slug, "")

            name = getattr(context, "name", None) or "Unknown"
            nim = getattr(context, "nim", None) or "Unknown"
            emotion = getattr(context, "emotion", None) or "neutral"

            # The tool policy has to match the tools actually registered on the
            # agent — naming rag_search in the prompt while the tool is absent
            # leaves the model with no instruction that it lacks Udinus
            # knowledge, and it answers campus questions from (fabricated) memory.
            policy_key = "tool-policy-rag" if settings.enable_rag_tool else "tool-policy-web"

            return templates["base"].format(
                character=character.name.capitalize(),
                name=name,
                nim=nim,
                emotion=emotion,
                TOOL_POLICY=templates[policy_key],
                CHARACTER=char_content,
            ).strip()
        except Exception:
            logger.exception("Failed to load system prompt templates")
            return "You are a helpful AI assistant."

    def get_tts_system_prompt(self) -> str:
        path = Path(self.base_path) / "text-to-speech.txt"
        try:
            prompt = path.read_text(encoding="utf-8").strip()
            if prompt:
                return prompt
        except Exception:
            logger.exception("Failed to load TTS system prompt")
        return _DEFAULT_TTS_FALLBACK


@lru_cache(maxsize=None)
def _load_prompt_templates(path: str) -> dict[str, str]:
    base_dir = Path(path)
    templates: dict[str, str] = {
        "base": (base_dir / "base.txt").read_text(encoding="utf-8"),
        "tool-policy-rag": (base_dir / "tool-policy-rag.txt").read_text(encoding="utf-8").strip(),
        "tool-policy-web": (base_dir / "tool-policy-web.txt").read_text(encoding="utf-8").strip(),
    }
    for character in Character:
        char_path = base_dir / f"{character.slug}.txt"
        templates[character.slug] = (
            char_path.read_text(encoding="utf-8") if char_path.exists() else ""
        )
    return templates
