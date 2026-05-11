"""Response use cases — text and speech AI response with streaming."""

import asyncio
import json
import logging
from functools import lru_cache
from typing import AsyncGenerator, Optional

from src.shared.constants import Character, MessageType
from src.domain.entities import (
    UserContext,
    UserMessage,
    Conversation,
    Message,
    AIMessage,
    ToolMessage,
    ToolsCall,
    Function,
    ResponseMetadata,
    UsageMetadata,
)

from src.ports.services import (
    ITTSService,
    IAgentService,
    ISTTService,
)
from src.ports.repositories import (
    IConversationRepository,
    IMessageRepository,
)
from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _load_prompt_templates(path: str) -> dict[str, str]:
    from pathlib import Path

    base_dir = Path(path)
    base_content = (base_dir / "base.txt").read_text(encoding="utf-8")

    char_templates: dict[str, str] = {"base": base_content, "vendra": "", "shazia": ""}
    vendra_path = base_dir / "vendra.txt"
    if vendra_path.exists():
        char_templates["vendra"] = vendra_path.read_text(encoding="utf-8")

    shazia_path = base_dir / "shazia.txt"
    if shazia_path.exists():
        char_templates["shazia"] = shazia_path.read_text(encoding="utf-8")

    return char_templates


@lru_cache(maxsize=None)
def load_tts_system_prompt(path: str = "src/prompts/text-to-speech.txt") -> str:
    """Load the TTS preprocessing system prompt."""
    try:
        from pathlib import Path

        prompt = Path(path).read_text(encoding="utf-8").strip()
        if prompt:
            return prompt
    except Exception:
        logger.exception("Failed to load TTS system prompt")

    return (
        "Rewrite the text into concise, natural Indonesian that is easy to read "
        "aloud by a TTS model. Preserve meaning and output only the speakable text."
    )


def load_system_prompt(
    character: Character,
    context: "UserContext | None" = None,
    path: str = "src/prompts",
) -> str:
    """Load dan render system prompt dengan context injection."""
    try:
        templates = _load_prompt_templates(path)

        if character == Character.VENDRA:
            char_content = templates.get("vendra", "")
        elif character == Character.SHAZIA:
            char_content = templates.get("shazia", "")
        else:
            char_content = ""

        name = getattr(context, "name", None) or "Unknown"
        nim = getattr(context, "nim", None) or "Unknown"
        emotion = getattr(context, "emotion", None) or "neutral"

        _char_name_map = {
            Character.VENDRA: "Vendra",
            Character.SHAZIA: "Shazia",
        }
        char_label = _char_name_map.get(character, character.name.capitalize())

        combined = templates["base"].format(
            character=char_label,
            name=name,
            nim=nim,
            emotion=emotion,
            CHARACTER=char_content,
        )
        return combined.strip()
    except Exception:
        logger.exception("Failed to load system prompt templates")
        return "You are a helpful AI assistant."


def build_title_prompt(user_input: str) -> str:
    return f"""Buatkan judul singkat (maksimal 5 kata) dalam Bahasa Indonesia untuk percakapan yang dimulai dengan pesan berikut:

    "{user_input}"

    Jawab hanya dengan judul saja, tanpa tanda kutip atau penjelasan tambahan."""


def _message_content_to_agent_dict(content: UserMessage | ToolMessage | AIMessage) -> dict:
    """Convert domain message content to dict format expected by agent."""
    if isinstance(content, UserMessage):
        return {"type": MessageType.USER.value, "content": content.content}
    if isinstance(content, ToolMessage):
        return {
            "type": MessageType.TOOL.value,
            "content": content.content,
            "tool_call_id": content.tool_call_id,
            "name": content.name,
        }
    if isinstance(content, AIMessage):
        tool_calls = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (content.additional_kwargs or [])
        ]
        return {
            "type": MessageType.ASSISTANT.value,
            "content": content.content,
            "additional_kwargs": {"tool_calls": tool_calls},
        }
    return {"type": MessageType.USER.value, "content": ""}


async def _yield_heartbeats_until_done(
    task: asyncio.Task,
    message: str,
) -> AsyncGenerator[dict, None]:
    """Yield periodic log events while a long-running task is still active."""
    heartbeat_interval = max(1.0, min(10.0, settings.sse_idle_timeout_seconds / 4))
    elapsed = 0.0
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=heartbeat_interval)
        if not done:
            elapsed += heartbeat_interval
            yield {"type": "log", "content": f"{message} ({int(elapsed)}s)"}


async def _stream_conversation(
    conversation_repo: IConversationRepository,
    message_repo: IMessageRepository,
    agent_service: IAgentService,
    tts_service: ITTSService,
    conversation_id: Optional[str],
    user_input: str,
    character: Character,
    context: UserContext | None,
    enable_tts: bool,
) -> AsyncGenerator[dict, None]:
    """Core streaming logic shared by text and speech use cases."""
    if conversation_id:
        conversation = await conversation_repo.get_by_id(conversation_id)
        if not conversation:
            yield {"type": "error", "content": "Conversation not found"}
            return
    else:
        yield {"type": "log", "content": "Membuat sesi baru..."}
        conversation = await conversation_repo.create(Conversation(title="Obrolan Baru"))
        yield {"type": "log", "content": "Sesi dibuat: Obrolan Baru"}

        async def _generate_and_update_title() -> None:
            try:
                title_prompt = build_title_prompt(user_input)
                title = await agent_service.generate_title(title_prompt)
                if title and title.strip():
                    await conversation_repo.update_title(conversation.id, title.strip())
            except Exception:
                logger.exception("Failed to generate/update conversation title (conversation_id=%s)", conversation.id)

        asyncio.create_task(_generate_and_update_title())

    system_prompt = load_system_prompt(character, context)

    yield {"type": "log", "content": "Menganalisis konteks..."}
    session_messages = await message_repo.get_recent_by_conversation_id(
        conversation.id, settings.chat_history_limit
    )
    messages_for_agent = [_message_content_to_agent_dict(m.body) for m in session_messages]

    # Persist current user message AFTER loading history, to avoid duplicating
    # user_input in the LLM context (agent already receives user_input separately).
    await message_repo.create(
        Message(
            conversation_id=conversation.id,
            character=character,
            body=UserMessage(content=user_input),
            context=context if context else None,
        )
    )

    full_response = ""
    try:
        gen = agent_service.generate_stream(
            user_input=user_input,
            messages=messages_for_agent,
            context=context,
            system_prompt=system_prompt,
        )
        while True:
            try:
                async with asyncio.timeout(settings.agent_idle_timeout_seconds):
                    event = await anext(gen)
            except StopAsyncIteration:
                break
            except TimeoutError:
                yield {"type": "error", "content": "Agent timeout"}
                return

            if event.get("type") == "token":
                full_response += event.get("content", "")
            yield event

            if event.get("type") == "assistant":
                rm = event.get("response_metadata") or {}
                um = event.get("usage_metadata") or {}
                tool_calls_raw = (event.get("additional_kwargs") or {}).get("tool_calls", [])
                tool_calls = []
                for tool in tool_calls_raw:
                    fn = tool.get("function") or {}
                    args = fn.get("arguments", "")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls.append(
                        ToolsCall(
                            id=tool.get("id", ""),
                            type=tool.get("type", ""),
                            function=Function(
                                name=fn.get("name", ""),
                                arguments=args if isinstance(args, str) else json.dumps(args),
                            ),
                        )
                    )
                await message_repo.create(
                    Message(
                        conversation_id=conversation.id,
                        character=character,
                        body=AIMessage(
                            content=event.get("content", ""),
                            additional_kwargs=tool_calls,
                            response_metadata=ResponseMetadata(
                                model=rm.get("model", ""),
                                done_reason=rm.get("done_reason", ""),
                            ),
                            usage_metadata=UsageMetadata(
                                input_tokens=um.get("input_tokens", 0),
                                output_tokens=um.get("output_tokens", 0),
                                total_tokens=um.get("total_tokens", 0),
                            ),
                        ),
                        context=context if context else None,
                    )
                )
            elif event.get("type") == "tool":
                await message_repo.create(
                    Message(
                        conversation_id=conversation.id,
                        character=character,
                        body=ToolMessage(
                            content=event.get("content", ""),
                            tool_call_id=event.get("tool_call_id", ""),
                            name=event.get("name", ""),
                        ),
                        context=context if context else None,
                    )
                )
    except Exception as e:
        logger.exception("Response streaming failed (conversation_id=%s)", conversation.id)
        yield {"type": "error", "content": "Internal server error"}
        return

    if enable_tts and tts_service and full_response:
        try:
            yield {"type": "log", "content": "Mensintesis speech..."}
            tts_system_prompt = load_tts_system_prompt()
            preprocess_task = asyncio.create_task(
                agent_service.preprocess_tts(full_response, tts_system_prompt)
            )
            async for heartbeat in _yield_heartbeats_until_done(
                preprocess_task, "Menyiapkan teks untuk speech..."
            ):
                yield heartbeat
            preprocessed_response = preprocess_task.result()

            synth_task = asyncio.create_task(
                tts_service.synthesize(preprocessed_response, character)
            )
            async for heartbeat in _yield_heartbeats_until_done(
                synth_task, "Mensintesis speech..."
            ):
                yield heartbeat
            audio_base64 = synth_task.result()
            yield {"type": "audio", "content": audio_base64}
        except Exception as e:
            logger.warning("TTS synth failed (conversation_id=%s): %s", conversation.id, e)
            yield {"type": "log", "content": "Warning: Failed to synthesize speech"}

    yield {"type": "done", "conversation_id": conversation.id}


class ResponseTextUseCase:
    """Use case for text-based AI response with streaming."""

    def __init__(
        self,
        conversation_repo: IConversationRepository,
        message_repo: IMessageRepository,
        agent_service: IAgentService,
        tts_service: ITTSService,
    ):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.agent_service = agent_service
        self.tts_service = tts_service

    async def execute(
        self,
        conversation_id: Optional[str],
        user_input: str,
        character: Character,
        context: UserContext | None,
        enable_tts: bool = True,
    ) -> AsyncGenerator[dict, None]:
        async for event in _stream_conversation(
            self.conversation_repo,
            self.message_repo,
            self.agent_service,
            self.tts_service,
            conversation_id,
            user_input,
            character,
            context,
            enable_tts,
        ):
            yield event


class ResponseSpeechUseCase:
    """Use case for speech-based AI response with streaming."""

    def __init__(
        self,
        conversation_repo: IConversationRepository,
        message_repo: IMessageRepository,
        agent_service: IAgentService,
        tts_service: ITTSService,
        stt_service: ISTTService,
    ):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.agent_service = agent_service
        self.tts_service = tts_service
        self.stt_service = stt_service

    async def execute(
        self,
        conversation_id: Optional[str],
        speech_base64: str,
        character: Character,
        context: UserContext | None,
        enable_tts: bool = True,
    ) -> AsyncGenerator[dict, None]:
        yield {"type": "log", "content": "Mentranskripsi speech..."}
        transcript = await self.stt_service.transcribe(speech_base64)
        yield {"type": "transcript", "content": transcript}

        async for event in _stream_conversation(
            self.conversation_repo,
            self.message_repo,
            self.agent_service,
            self.tts_service,
            conversation_id,
            transcript,
            character,
            context,
            enable_tts,
        ):
            yield event
