"""Filesystem-backed prompt repository.

Loads templates once at startup and renders them with the per-request context.
"""

import logging
from functools import lru_cache
from pathlib import Path
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

            return templates["base"].format(
                character=character.name.capitalize(),
                name=name,
                nim=nim,
                emotion=emotion,
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

    def build_title_prompt(self, user_input: str) -> str:
        return (
            "Buatkan judul singkat (maksimal 5 kata) dalam Bahasa Indonesia untuk "
            f"percakapan yang dimulai dengan pesan berikut:\n\n    \"{user_input}\"\n\n"
            "    Jawab hanya dengan judul saja, tanpa tanda kutip atau penjelasan tambahan."
        )


@lru_cache(maxsize=None)
def _load_prompt_templates(path: str) -> dict[str, str]:
    base_dir = Path(path)
    templates: dict[str, str] = {
        "base": (base_dir / "base.txt").read_text(encoding="utf-8"),
    }
    for character in Character:
        char_path = base_dir / f"{character.slug}.txt"
        templates[character.slug] = (
            char_path.read_text(encoding="utf-8") if char_path.exists() else ""
        )
    return templates
