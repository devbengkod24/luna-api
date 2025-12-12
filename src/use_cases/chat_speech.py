"""Chat speech use case."""
import uuid
from typing import AsyncGenerator, Optional, Dict, Any

from src.ports.chat_repository import IChatRepository
from src.ports.llm import ILLMAdapter
from src.ports.speech import ISTTAdapter
from src.domain.entities.chat import UserContext


class ChatSpeechUseCase:
    """Use case for speech-based chat with streaming response."""
    
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
            stt_adapter: STT adapter for speech transcription
        """
        self.chat_repository = chat_repository
        self.llm_adapter = llm_adapter
        self.stt_adapter = stt_adapter

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with context.

        Args:
            context: User context

        Returns:
            str: System prompt
        """
        emotion = context.get("emotion", "neutral")
        name = context.get("name", "Pengguna")

        emotion_instructions = {
            "happy": "Pengguna sedang senang, berikan respons yang antusias dan positif.",
            "sad": "Pengguna sedang sedih, berikan respons yang empatik dan mendukung.",
            "angry": "Pengguna sedang marah, tetap tenang dan bantu meredakan emosi.",
            "surprised": "Pengguna sedang terkejut, berikan penjelasan yang jelas.",
            "fearful": "Pengguna sedang takut, berikan ketenangan dan dukungan.",
            "disgusted": "Pengguna sedang tidak senang, tanyakan apa yang bisa dibantu.",
            "neutral": "Berikan respons yang informatif dan membantu.",
        }

        emotion_instruction = emotion_instructions.get(
            emotion, emotion_instructions["neutral"]
        )

        return f"""Kamu adalah asisten AI bernama Sahabat AI yang ramah, sopan, dan membantu.
        Kamu berbicara dalam Bahasa Indonesia yang baik dan benar.

        Konteks percakapan:
        - Nama pengguna: {name}
        - Emosi saat ini: {emotion}

        Instruksi berdasarkan emosi:
        {emotion_instruction}

        Panduan umum:
        - Jawab dengan ringkas tapi informatif
        - Gunakan bahasa yang sesuai dengan konteks
        - Jika tidak tahu jawaban, katakan dengan jujur
        - Jangan berikan informasi yang salah atau berbahaya"""

    def _build_title_prompt(self, first_message: str) -> str:
        """Build prompt for generating conversation title.

        Args:
            first_message: First user message in the conversation

        Returns:
            str: Complete prompt for title generation
        """
        return f"""Buatkan judul singkat (maksimal 5 kata) dalam Bahasa Indonesia untuk percakapan yang dimulai dengan pesan berikut:

"{first_message}"

Jawab hanya dengan judul saja, tanpa tanda kutip atau penjelasan tambahan."""
    
    async def execute(
        self,
        session_uuid: Optional[str],
        speech_base64: str,
        context: UserContext,
    ) -> AsyncGenerator[dict, None]:
        """Execute chat speech use case with streaming.
        
        Args:
            session_uuid: Optional session UUID (if None, create new session)
            speech_base64: Base64-encoded speech audio data
            context: User context (emotion, name)
            
        Yields:
            dict: Streaming events (log, token, done)
        """
        # Step 1: Transcribe speech
        yield {"type": "log", "content": "Mentranskripsi speech..."}
        transcript = await self.stt_adapter.transcribe(speech_base64)
        yield {"type": "transcript", "content": transcript}
        
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
            title_prompt = self._build_title_prompt(transcript)
            title = await self.llm_adapter.generate_title(title_prompt)
            
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
        
        # Build system prompt and add as first message
        system_prompt = self._build_system_prompt(context.to_dict())
        messages_with_system = [
            {"role": "system", "content": system_prompt},
            *messages
        ]
        
        # Step 5: Generate streaming response
        full_response = ""
        async for token in self.llm_adapter.generate_stream(messages_with_system):
            full_response += token
            yield {"type": "token", "content": token}
        
        # Step 6: Save assistant response
        await self.chat_repository.add_message(
            session_id=session.id,
            role="assistant",
            content=full_response,
        )
        
        # Step 7: Send completion event
        yield {"type": "done", "finish_reason": "stop", "session_uuid": session_uuid}

