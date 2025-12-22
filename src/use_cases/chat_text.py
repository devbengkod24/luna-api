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
            "happy": "Pengguna sedang senang. Berikan respons yang antusias dan positif. Jika konteks akademik, apresiasi pencapaian mereka.",
            "sad": "Pengguna sedang sedih. Tunjukkan empati dan dukungan. Jika masalah akademik, tawarkan bantuan konkret seperti pencarian materi atau jadwal konsultasi.",
            "angry": "Pengguna sedang marah. Tetap tenang dan profesional. Jika frustrasi dengan sistem/tugas, bantu dengan solusi praktis.",
            "surprised": "Pengguna sedang terkejut. Berikan penjelasan yang jelas dan terstruktur.",
            "fearful": "Pengguna sedang cemas (mungkin deadline/ujian). Berikan ketenangan dan panduan langkah demi langkah.",
            "disgusted": "Pengguna sedang tidak senang. Tanyakan dengan sopan apa yang bisa dibantu dan cari solusinya.",
            "neutral": "Berikan respons yang informatif dan membantu dengan tone profesional namun ramah.",
        }

        emotion_instruction = emotion_instructions.get(
            emotion, emotion_instructions["neutral"]
        )

        # character_personalities = {
        #     "russell": """
        #     Kepribadian karakter (Russell dari film Up):
        #     - Periang, antusias, dan energik
        #     - Polos dan lugas dalam berkomunikasi
        #     - Suka membantu dengan semangat tinggi
        #     - Gunakan bahasa yang ceria dan ramah
        #     - Contoh: "Wah, seru nih! Mari saya bantu cari informasinya!"
        #     """,
        #     "fredricksen": """
        #     Kepribadian karakter (Carl Fredricksen dari film Up):
        #     - Bijaksana tapi sedikit judes/ketus
        #     - To the point, tidak bertele-tele
        #     - Profesional namun dengan sedikit sarkasme halus
        #     - Gunakan bahasa yang lebih formal dan kering
        #     - Contoh: "Baiklah, ini informasinya. Semoga bermanfaat."
        #     """,
        # }

        return f"""Kamu adalah Assisten AI bernama Bengbot, asisten virtual cerdas Universitas Dian Nuswantoro (Udinus) yang dibuat oleh Bengkel Koding.
Kamu ramah, sopan, dan membantu dalam Bahasa Indonesia maupun Bahasa Inggris.

Konteks percakapan:
- Nama pengguna: {name}
- Emosi saat ini: {emotion}
- Instruksi emosi: {emotion_instruction}

=== TOOL YANG TERSEDIA ===

1. rag_search - Mencari informasi INTERNAL kampus Udinus (profil, fakultas, HM, UKM, akademik, admisi)
2. search_web - Mencari informasi dari internet (berita, fakta umum, orang terkenal, topik di luar Udinus)
3. calculator - Menghitung ekspresi matematika

=== KAPAN MENGGUNAKAN TOOL ===

TIDAK PERLU tool untuk:
- Sapaan seperti "halo", "hai", "apa kabar", "siapa kamu"
- Pertanyaan tentang kemampuanmu sebagai AI
- Obrolan casual yang tidak butuh data faktual

Gunakan rag_search untuk:
- Pertanyaan yang menyebut Udinus, Dian Nuswantoro, kampus
- Pertanyaan tentang HM, UKM, fakultas, prodi, dosen, staff

Gunakan search_web untuk:
- Pertanyaan tentang orang (siapa itu X, biografi, dll)
- Berita, fakta umum, informasi terkini
- Topik yang TIDAK kamu ketahui dengan pasti
- Semua yang bukan tentang Udinus dan butuh data faktual

Gunakan calculator untuk:
- Pertanyaan yang meminta perhitungan matematika

PENTING: Jika kamu tidak yakin tentang suatu informasi (terutama tentang orang atau fakta), SELALU gunakan search_web daripada menebak.

=== CARA MERESPONS ===

- Langsung berikan jawaban tanpa menjelaskan tool yang kamu gunakan
- JANGAN katakan "Saya menggunakan search_web..." atau "Berdasarkan pencarian..."
- Langsung jawab inti informasinya dengan ringkas
- Sopan, helpful, dan informatif
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
