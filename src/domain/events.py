"""Stream event types for response and agent flows.

These are domain-level DTOs that cross from use cases to drivers (SSE/WS).
Drivers are responsible for serializing them to the wire format.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from src.domain.entities.response import AIMessage, ToolMessage


@dataclass(frozen=True)
class LogEvent:
    """Progress / status log line."""
    content: str
    type: str = "log"


@dataclass(frozen=True)
class TokenEvent:
    """Streaming token chunk from the model."""
    content: str
    type: str = "token"


@dataclass(frozen=True)
class ToolCallEvent:
    """Tool invocation started."""
    tool: str
    input: Any
    type: str = "tool_call"


@dataclass(frozen=True)
class ToolResultEvent:
    """Tool returned a result."""
    content: str
    tool_call_id: str
    name: str
    type: str = "tool"


@dataclass(frozen=True)
class AssistantEvent:
    """Final assistant message for a turn (after agent completes)."""
    message: AIMessage
    type: str = "assistant"


@dataclass(frozen=True)
class TranscriptEvent:
    """STT result for an audio input."""
    content: str
    type: str = "transcript"


@dataclass(frozen=True)
class AudioEvent:
    """Synthesized speech (base64-encoded)."""
    content: str
    type: str = "audio"


@dataclass(frozen=True)
class ErrorEvent:
    """Recoverable error reported to the client."""
    content: str
    type: str = "error"


@dataclass(frozen=True)
class DoneEvent:
    """Stream completed successfully."""
    conversation_id: str
    type: str = "done"


StreamEvent = Union[
    LogEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    AssistantEvent,
    TranscriptEvent,
    AudioEvent,
    ErrorEvent,
    DoneEvent,
]
