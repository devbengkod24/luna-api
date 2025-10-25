"""Chat audio use case."""
import uuid
from typing import AsyncGenerator, Optional

from src.ports.repository import IChatRepository
from src.ports.llm import ILLMAdapter
from src.ports.audio import ISTTAdapter
from src.domain.entities.context import UserContext


class ChatAudioUseCase:
    """Use case for audio-based chat with streaming response."""
    
    def __init__(
        self,
        chat_repository: IChatRepository,
        llm_adapter: ILLMAdapter,
        stt_adapter: ISTTAdapter,
    ):
        """Initialize use case with dependencies.
        
        Args:
            chat_repository: Repository for chat persistence
            llm_adapter: LLM adapter for response generation
            stt_adapter: STT adapter for audio transcription
        """
        self.chat_repository = chat_repository
        self.llm_adapter = llm_adapter
        self.stt_adapter = stt_adapter
    
    async def execute(
        self,
        session_uuid: Optional[str],
        audio_base64: str,
        context: UserContext,
    ) -> AsyncGenerator[dict, None]:
        """Execute chat audio use case with streaming.
        
        Args:
            session_uuid: Optional session UUID (if None, create new session)
            audio_base64: Base64-encoded audio data
            context: User context (emotion, name)
            
        Yields:
            dict: Streaming events (log, token, done)
        """
        # Step 1: Transcribe audio
        yield {"type": "log", "content": "Mentranskripsi audio..."}
        transcript = await self.stt_adapter.transcribe(audio_base64)
        yield {"type": "log", "content": f"Transkrip: '{transcript}'"}
        
        # Step 2: Get or create session
        if session_uuid:
            session = await self.chat_repository.get_session_by_uuid(session_uuid)
            if not session:
                yield {"type": "error", "content": "Session not found"}
                return
        else:
            # Generate new session UUID
            session_uuid = str(uuid.uuid4())
            
            # Generate title from transcript
            title = await self.llm_adapter.generate_title(transcript)
            
            session = await self.chat_repository.create_session(
                uuid=session_uuid,
                title=title,
            )
        
        # Step 3: Add user message to session
        await self.chat_repository.add_message(
            session_id=session.id,
            role="user",
            content=transcript,
            context=context.to_dict(),
        )
        
        # Step 4: Build message history
        messages = []
        for msg in session.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        messages.append({"role": "user", "content": transcript})
        
        # Step 5: Generate streaming response
        full_response = ""
        async for token in self.llm_adapter.generate_stream(messages, context.to_dict()):
            full_response += token
            yield {"type": "token", "content": token}
        
        # Step 6: Save assistant response
        await self.chat_repository.add_message(
            session_id=session.id,
            role="assistant",
            content=full_response,
        )
        
        # Step 7: Send completion event
        yield {"type": "done", "session_uuid": session_uuid}

