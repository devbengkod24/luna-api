"""Response use cases — text and speech AI response with streaming."""

import json
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


def load_system_prompt(
    character: Character,
    context: "UserContext | None" = None,
    path: str = "src/prompts",
) -> str:
    """Load dan render system prompt dengan context injection.

    base.txt menggunakan template placeholders:
      {character}  — nama karakter (mis. "Russell", "Fredricksen")
      {name}       — nama user dari UserContext
      {nim}        — NIM user dari UserContext
      {emotion}    — emosi user dari UserContext
      {CHARACTER}  — isi file character (russell.txt / fredricksen.txt)
    """
    from pathlib import Path

    try:
        base_dir = Path(path)
        base_path = base_dir / "base.txt"

        # Load character-specific file
        if character == Character.RUSSELL:
            char_path = base_dir / "russell.txt"
        elif character == Character.FREDRICKSEN:
            char_path = base_dir / "fredricksen.txt"
        else:
            char_path = None

        base_content = base_path.read_text(encoding="utf-8")

        char_content = ""
        if char_path and char_path.exists():
            char_content = char_path.read_text(encoding="utf-8")

        # Resolve UserContext values (gunakan fallback jika context None)
        name = getattr(context, "name", None) or "Unknown"
        nim = getattr(context, "nim", None) or "Unknown"
        emotion = getattr(context, "emotion", None) or "neutral"

        # Map Character enum ke nama string untuk {character} placeholder
        # Character.RUSSELL (0) → "Russell", Character.FREDRICKSEN (1) → "Fredricksen"
        _char_name_map = {
            Character.RUSSELL: "Russell",
            Character.FREDRICKSEN: "Fredricksen",
        }
        char_label = _char_name_map.get(character, character.name.capitalize())

        # Inject placeholders — gunakan .format() dengan explicit mapping
        combined = base_content.format(
            character=char_label,
            name=name,
            nim=nim,
            emotion=emotion,
            CHARACTER=char_content,
        )
        return combined.strip()

    except Exception:
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
        if conversation_id:
            conversation = await self.conversation_repo.get_by_id(conversation_id)
            if not conversation:
                yield {"type": "error", "content": "Conversation not found"}
                return
        else:
            yield {"type": "log", "content": "Membuat sesi baru..."}
            title_prompt = build_title_prompt(user_input)
            title = await self.agent_service.generate_title(title_prompt)
            conversation = await self.conversation_repo.create(Conversation(title=title))
            yield {"type": "log", "content": f"Sesi dibuat: {title}"}

        system_prompt = load_system_prompt(character, context)

        await self.message_repo.create(
            Message(
                conversation_id=conversation.id,
                character=character,
                body=UserMessage(content=user_input),
                context=context if context else None,
            )
        )

        yield {"type": "log", "content": "Menganalisis konteks..."}
        session_messages = await self.message_repo.get_by_conversation_id(conversation.id)
        messages_for_agent = [_message_content_to_agent_dict(m.body) for m in session_messages]
        full_response = ""
        try:
            async for event in self.agent_service.generate_stream(
                user_input=user_input,
                messages=messages_for_agent,
                context=context,
                system_prompt=system_prompt,
            ):
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
                    await self.message_repo.create(
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
                    await self.message_repo.create(
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
            yield {"type": "error", "content": str(e)}

        if enable_tts and self.tts_service and full_response:
            try:
                yield {"type": "log", "content": "Mensintesis speech..."}
                audio_base64 = await self.tts_service.synthesize(full_response, character)
                yield {"type": "audio", "content": audio_base64}
            except Exception as e:
                yield {"type": "log", "content": f"Warning: Failed to synthesize speech: {e}"}

        yield {"type": "done", "conversation_id": conversation.id}


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

        if conversation_id:
            conversation = await self.conversation_repo.get_by_id(conversation_id)
            if not conversation:
                yield {"type": "error", "content": "Conversation not found"}
                return
        else:
            yield {"type": "log", "content": "Membuat sesi baru..."}
            title_prompt = build_title_prompt(transcript)
            title = await self.agent_service.generate_title(title_prompt)
            conversation = await self.conversation_repo.create(Conversation(title=title))
            yield {"type": "log", "content": f"Sesi dibuat: {title}"}

        system_prompt = load_system_prompt(character, context)

        await self.message_repo.create(
            Message(
                conversation_id=conversation.id,
                character=character,
                body=UserMessage(content=transcript),
                context=context if context else None,
            )
        )

        session_messages = await self.message_repo.get_by_conversation_id(conversation.id)
        messages_for_agent = [_message_content_to_agent_dict(m.body) for m in session_messages]
        full_response = ""
        try:
            async for event in self.agent_service.generate_stream(
                user_input=transcript,
                messages=messages_for_agent,
                context=context,
                system_prompt=system_prompt,
            ):
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
                    await self.message_repo.create(
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
                    await self.message_repo.create(
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
            yield {"type": "error", "content": str(e)}

        if enable_tts and self.tts_service and full_response:
            try:
                yield {"type": "log", "content": "Mensintesis speech..."}
                audio_base64 = await self.tts_service.synthesize(full_response, character)
                yield {"type": "audio", "content": audio_base64}
            except Exception as e:
                yield {"type": "log", "content": f"Warning: Failed to synthesize speech: {e}"}

        yield {"type": "done", "conversation_id": conversation.id}
