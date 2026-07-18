"""Presenter for stream events (SSE / WebSocket payloads)."""

from typing import Any

from src.domain.events import (
    AssistantEvent,
    AudioChunkEvent,
    AudioEvent,
    DoneEvent,
    ErrorEvent,
    LogEvent,
    SourcesEvent,
    StreamEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    TranscriptEvent,
)


def stream_event_to_dict(event: StreamEvent) -> dict[str, Any]:
    """Serialize a typed stream event to a dict for SSE/WS transport."""
    if isinstance(event, LogEvent):
        return {"type": "log", "content": event.content}
    if isinstance(event, TokenEvent):
        return {"type": "token", "content": event.content}
    if isinstance(event, ToolCallEvent):
        return {"type": "tool_call", "tool": event.tool, "input": event.input}
    if isinstance(event, ToolResultEvent):
        return {
            "type": "tool",
            "content": event.content,
            "tool_call_id": event.tool_call_id,
            "name": event.name,
        }
    if isinstance(event, SourcesEvent):
        return {"type": "sources", "items": event.sources}
    if isinstance(event, AssistantEvent):
        msg = event.message
        return {
            "type": "assistant",
            "content": msg.content,
            "additional_kwargs": {
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.additional_kwargs or [])
                ]
            },
            "response_metadata": {
                "model": msg.response_metadata.model,
                "done_reason": msg.response_metadata.done_reason,
            },
            "usage_metadata": {
                "input_tokens": msg.usage_metadata.input_tokens,
                "output_tokens": msg.usage_metadata.output_tokens,
                "total_tokens": msg.usage_metadata.total_tokens,
            },
        }
    if isinstance(event, TranscriptEvent):
        return {"type": "transcript", "content": event.content}
    if isinstance(event, AudioEvent):
        return {"type": "audio", "content": event.content}
    if isinstance(event, AudioChunkEvent):
        return {
            "type": "audio_chunk",
            "content": event.content,
            "sample_rate": event.sample_rate,
            "encoding": event.encoding,
        }
    if isinstance(event, ErrorEvent):
        return {"type": "error", "content": event.content}
    if isinstance(event, DoneEvent):
        return {"type": "done", "conversation_id": event.conversation_id}
    raise TypeError(f"Unknown stream event type: {type(event).__name__}")
