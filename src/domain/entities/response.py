"""Response domain entities — used by the AI response (text/speech) flow."""

from dataclasses import dataclass

from src.shared.constants import MessageType


@dataclass
class ResponseMetadata:
    model: str
    done_reason: str


@dataclass
class UsageMetadata:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class Function:
    name: str
    arguments: str


@dataclass
class ToolsCall:
    id: str
    type: str
    function: Function


@dataclass
class UserMessage:
    content: str
    type: MessageType = MessageType.USER


@dataclass
class ToolMessage:
    content: str
    tool_call_id: str
    name: str
    type: MessageType = MessageType.TOOL


@dataclass
class AIMessage:
    content: str
    additional_kwargs: list[ToolsCall]
    response_metadata: ResponseMetadata
    usage_metadata: UsageMetadata
    type: MessageType = MessageType.ASSISTANT


@dataclass
class UserContext:
    emotion: str | None = None
    name: str | None = None
    nim: str | None = None
