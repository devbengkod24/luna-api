"""Chat text use case."""
import uuid
from typing import AsyncGenerator, Optional

from src.ports.chat_repository import IChatRepository
from src.ports.agent import IAgentAdapter
from src.domain.entities.chat import UserContext
from typing import Dict, Any

class ChatTextUseCase:
    """Use case for text-based chat with streaming response."""
    
    def __init__(
        self,
        chat_repository: IChatRepository,
        agent_adapter: IAgentAdapter,
    ):
        """Initialize use case with dependencies.
        
        Args:
            chat_repository: Repository for chat persistence
            agent_adapter: Agent adapter for response generation
        """
        self.chat_repository = chat_repository
        self.agent_adapter = agent_adapter
    

    def _system_prompt(self, context: Dict[str, Any]) -> str:
        """System prompt with context."""
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
        - Jangan berikan informasi yang salah atau berbahaya
        
        Kamu memiliki akses ke knowledge base yang dapat kamu gunakan untuk mencari informasi.
        Gunakan tool search_knowledge_base ketika kamu perlu mencari informasi dari dokumen.

        Instruksi tool:
        - Gunakan tool jika pertanyaan memerlukan informasi spesifik dari knowledge base
        - Jika informasi tidak ditemukan, jawab berdasarkan pengetahuan umum kamu
        """

    def _title_prompt(self, first_message: str) -> str:
        """Prompt for generating conversation title."""
        return f"""Buatkan judul singkat (maksimal 5 kata) dalam Bahasa Indonesia untuk percakapan yang dimulai dengan pesan berikut:

        "{first_message}"

        Jawab hanya dengan judul saja, tanpa tanda kutip atau penjelasan tambahan."""


    async def execute(
        self,
        session_uuid: Optional[str],
        prompt: str,
        context: UserContext,
    ) -> AsyncGenerator[dict, None]:
        """Execute chat text use case with streaming.
        
        Args:
            session_uuid: Optional session UUID (if None, create new session)
            prompt: User prompt
            context: User context (emotion, name)
            
        Yields:
            dict: Streaming events (log, token, done)
        """
       
        if session_uuid:
            session = await self.chat_repository.get_session_by_uuid(session_uuid)
            if not session:
                yield {"type": "error", "content": "Session not found"}
                return
        else:
            session_uuid = str(uuid.uuid4())
            
            yield {"type": "log", "content": "Membuat sesi baru..."}
            title_prompt = self._title_prompt(prompt)
            title = await self.agent_adapter.generate_title(title_prompt)
            
            session = await self.chat_repository.create_session(
                uuid=session_uuid,
                title=title,
            )
            
            yield {"type": "log", "content": f"Sesi dibuat: {title}"}
        
        await self.chat_repository.add_message(
            session_id=session.id,
            role="user",
            content=prompt,
            context=context.to_dict(),
        )

        system_prompt = self._system_prompt(context.to_dict())

        yield {"type": "log", "content": "Menganalisis konteks..."}
        messages = []
        for msg in session.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        messages.append({"role": "user", "content": prompt})
        
        try:
            full_response = ""
            
            async for event in self.agent_adapter.generate_stream(
                messages=messages,
                context=context,
                system_prompt=system_prompt,
            ):
                if event.get("type") == "token":
                    token = event.get("content", "")
                    full_response += token

                yield event

            await self.chat_repository.add_message(
                session_id=session.id,
                role="assistant",
                content=full_response,
            )
                
            yield {"type": "done", "finish_reason": "stop", "session_uuid": session_uuid}
        except Exception as e:
            error_message = str(e)
            yield {"type": "error", "content": error_message}
            yield {"type": "done", "finish_reason": "error", "session_uuid": session_uuid}
            return
