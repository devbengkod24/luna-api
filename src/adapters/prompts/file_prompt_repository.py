"""Filesystem-backed prompt repository.

Loads templates once at startup and renders them with the per-request context.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path

from src.adapters.agent.input_guard import is_harmful, is_jailbreak
from src.config import settings
from src.domain.entities import UserContext
from src.ports.prompts import IPromptRepository
from src.shared.constants import Character

logger = logging.getLogger(__name__)

# Longest a context value may be once it is in the prompt. The REST schema
# caps them too, but the kiosk path does not go through that schema.
_CONTEXT_LIMITS = {"name": 100, "nim": 20, "emotion": 20}


def _context_value(value: str | None, field: str) -> str | None:
    """A context field fit to sit inside the system prompt as data.

    The name, NIM and emotion come from outside — the kiosk, or any REST caller —
    and are placed in the system prompt, above the policy the model follows. A
    value with line breaks can start a new "section" of that prompt, and the
    input guard only ever looked at the question. So: one line, no control
    characters, bounded, and dropped entirely if it reads as an instruction.
    """
    if not value:
        return None
    single_line = re.sub(r"\s+", " ", "".join(ch for ch in str(value) if ch.isprintable() or ch.isspace()))
    cleaned = single_line.strip()[: _CONTEXT_LIMITS[field]]
    if not cleaned or is_jailbreak(cleaned) or is_harmful(cleaned):
        if cleaned:
            logger.warning("Dropped a %s from the user context that read as an instruction", field)
        return None
    return cleaned


class FilePromptRepository(IPromptRepository):

    def __init__(self, base_path: str = "src/prompts"):
        self.base_path = base_path

    def get_system_prompt(
        self, character: Character, context: UserContext | None = None
    ) -> str:
        try:
            templates = _load_prompt_templates(self.base_path)
            char_content = templates.get(character.slug, "")

            name = _context_value(getattr(context, "name", None), "name") or "Unknown"
            nim = _context_value(getattr(context, "nim", None), "nim") or "Unknown"
            emotion = _context_value(getattr(context, "emotion", None), "emotion") or "neutral"

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
