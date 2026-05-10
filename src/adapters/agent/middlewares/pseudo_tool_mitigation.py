"""Mitigasi pseudo-tool Ollama: nama tool fiktif + JSON tool-ish di teks jawaban."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from langchain.agents.middleware import ModelResponse, wrap_model_call
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# Ollama / model kecil kadang memancarkan ini alih-alih jawaban biasa atau tool resmi.
PSEUDO_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "answer directly",
        "answer_directly",
        "direct_answer",
        "no_tool",
        "no tool",
        "none",
        "empathy",
        "respond_with_empathy",
        "respond with empathy",
        "casual",
        "casual_response",
        "direct",
        "skip_tool",
    }
)


def is_pseudo_tool_json_text(content: str) -> bool:
    """True jika `content` adalah JSON dengan field name yang termasuk pseudo-tool."""
    stripped = (content or "").strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped)
        name = (data.get("name") or "").strip().lower()
        return name in PSEUDO_TOOL_NAMES
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


def _text_content(msg: AIMessage) -> str:
    c = msg.content
    return c if isinstance(c, str) else ""


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, Mapping):
        return (tc.get("name") or "").strip().lower()
    return (getattr(tc, "name", None) or "").strip().lower()


def _filter_tool_calls(tool_calls: list[Any] | None) -> list[Any]:
    if not tool_calls:
        return []
    return [tc for tc in tool_calls if _tool_call_name(tc) not in PSEUDO_TOOL_NAMES]


@wrap_model_call
async def strip_pseudo_tools_middleware(request, handler):
    """Filter tool_calls pseudo-Ollama; fallback sekali tanpa tools jika isi JSON sampah."""
    response = await handler(request)
    if not response.result:
        return response

    first = response.result[0]
    if not isinstance(first, AIMessage):
        return response

    rest = list(response.result[1:])
    text = _text_content(first).strip()
    pseudo_body = is_pseudo_tool_json_text(text)
    filtered = _filter_tool_calls(first.tool_calls)

    need_fallback = False
    if not filtered:
        if pseudo_body or not text:
            need_fallback = True

    if need_fallback and request.tools:
        logger.debug("Pseudo-tool mitigation: retrying model call without tools")
        return await handler(request.override(tools=[]))

    if need_fallback and not request.tools:
        cleaned = first.model_copy(update={"tool_calls": [], "content": ""})
        return ModelResponse(
            result=[cleaned, *rest],
            structured_response=response.structured_response,
        )

    original_tcs = list(first.tool_calls or [])
    if filtered == original_tcs and not pseudo_body:
        return response

    updated = first.model_copy(update={"tool_calls": filtered})
    return ModelResponse(
        result=[updated, *rest],
        structured_response=response.structured_response,
    )
