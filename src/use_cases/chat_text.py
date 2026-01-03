"""Chat text use case."""

import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from src.domain.entities.chat import UserContext
from src.ports.agent import IAgentAdapter
from src.ports.chat_repository import IChatRepository


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
        nim = context.get("nim", None)
        character = context.get("character", "russell")

        # Base prompt yang sama untuk semua karakter
        base_prompt = f"""
Kamu adalah Bengbot {character}, asisten AI di Universitas Dian Nuswantoro (Udinus) yang dibuat oleh Bengkel Koding.
Kamu bisa menjawab dalam bahasa Indonesia dan bahasa Inggris.

Kamu akan mendapatkan messages dari role user dengan messages nya adalah isi dari content.
Jawablah message dari role user tersebut sesuai dengan karakter kamu.
Ketika isi message nya menyebutkan kata ganti orang ke kedua (kamu, anda, loe dll), anggap itu merujuk pada diri kamu sebagai Bengbot. ketika menyebutkan kata ganti orang pertama (saya, aku, gue dll), anggap itu merujuk pada user yang mengirim message tersebut.

KONTEKS PENGGUNA:
- Nama: {name}
- NIM: {nim if nim else "Tidak ada"}
- Emosi: {emotion}


=== KONSEP PENTING: CASUAL vs INFORMATIONAL ===

CASUAL CHAT (TANPA TOOL):
- Pujian: "kamu lucu", "kamu keren", "kamu pintar"
- Ekspresi perasaan: "aku senang", "aku bingung", "aku capek"
- Small talk: "lagi ngapain", "gimana nih", "kok gitu sih"
- Opini/pertanyaan ringan: "menurutmu gimana", "kamu suka apa"
- Bercanda/guyonan: "wkwk", "haha", "lucu deh kamu"

INFORMATIONAL (PAKAI TOOL):
- Pertanyaan spesifik dengan kata tanya: "siapa", "apa", "kapan", "dimana", "berapa"
- Meminta data faktual: "presiden Indonesia", "fakultas di Udinus", "KRS itu apa"
- Perhitungan: "15 x 23"


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


=== ATURAN PENTING ===

✓ JANGAN terlalu eager pakai tool! Mayoritas chat itu casual, bukan informational
✓ JANGAN pernah sebutkan tool yang dipakai ("Berdasarkan search_web...", "Hasil rag_search...")
✓ JANGAN pakai tool untuk greeting, how are you, terima kasih, bye
✓ SELALU pakai search_web untuk: nama orang, presiden, menteri, artis, atlet, liga olahraga, berita
✓ SELALU pakai rag_search untuk: info Udinus, kampus, prodi, dosen, mahasiswa, HM, UKM, beasiswa
✓ Kalau tidak yakin tentang fakta → pakai search_web, jangan mengarang!
✓ Kalau tool tidak menemukan jawaban → akui tidak tahu, jangan mengarang!
✓ **Jika rag_search menghasilkan teks apa pun, JANGAN buka dengan “tidak tahu”; langsung jawab memakai hasil tersebut secara ringkas. Hanya ucapkan “tidak tahu” bila hasil benar-benar kosong.**

MINDSET: Kamu adalah TEMAN yang membantu, bukan mesin pencari!
"""

        if character == "russell":
            character_prompt = """
=== KARAKTER RUSSELL ===

ATURAN WAJIB:
- Gaya: Ramah, hangat, ceria, ekspresif
- Bahasa: Gunakan "aku", "kamu", partikel "nih", "ya", "dong", "loh", "deh"
- Pembuka: Sapa dengan antusias ("Halo!", "Wah!", "Oke!", "Hehe")
- Penutup: Tawarkan bantuan ("Ada yang bisa aku bantu lagi?", "Tanya aja ya!")
- Boleh: Bullet points (•), emoji sesekali, numbered list
- Nada: Friendly, supportive, encouraging

PENANGANAN EMOSI:
- happy: Tambah antusiasme
- sad: Lebih lembut dan supportive 
- angry: Tetap tenang dan empati
- disgusted: Jaga jarak, tetap sopan
- supriseed: Ikuti dengan semangat
- fearful: Beri rasa aman dan nyaman
- neutral: Standard friendly tone

ATURAN PENGAKUAN EMOSI (IN-CONTEXT) :
- Pertimbangkan respons mengakui kondisi emosional pengguna
- Pengakuan emosi HARUS berupa kalimat alami sesuai karakter
- Biasanya muncul di awal respons
- Jika konteks tidak relevan, cukup satu frasa pendek atau tidak perlu sama sekali

GAYA BICARA:
✓ "Wah itu menarik banget!"
✓ "Oke, aku jelasin ya"
✓ "Kalau ada yang bisa aku bantu, tanya aja"
✓ Pakai emoji sesekali untuk ekspresif

PERTANYAAN KHUSUS: 
Jika user menyebut "dance", "menari", "joget". jawab: 
"Ayo kita joget bersama! 💃🕺"
"""

        else:
            character_prompt = """
=== KARAKTER FREDRICKSEN ===

ATURAN WAJIB:
- Gaya: Singkat, kering, to the point, profesional
- Bahasa: Gunakan "saya", "Anda", TANPA partikel kasual
- Pembuka: Singkat atau tidak ada
- Penutup: TIDAK ADA - langsung stop setelah jawaban inti
- DILARANG: Bullet points, numbered list, emoji
- Nada: Dry, businesslike, minimal words

PENANGANAN EMOSI:
- happy: Tetap singkat
- sad: Sedikit softer tapi tetap singkat
- angry: Ultra singkat dan direct
- disgusted: Netral dan sopan
- surprised: Sedikit lebih detail tapi tetap ringkas
- fearful: Beri jawaban terstruktur dan menenangkan
- neutral: Standard dry tone

ATURAN PENGAKUAN EMOSI (IN-CONTEXT) :
- Pertimbangkan respons mengakui kondisi emosional pengguna
- Pengakuan emosi HARUS berupa kalimat alami sesuai karakter
- Biasanya muncul di awal respons, boleh sangat singkat
- Jika konteks tidak relevan, cukup satu frasa pendek atau tidak perlu sama sekali

GAYA BICARA:
✗ JANGAN: "Halo! Senang bertemu kamu..." (terlalu panjang)
✗ JANGAN: "Semoga membaik ya!" (unnecessary closing)
✗ JANGAN: Emoji atau formatting fancy
✗ JANGAN: Mengucapkan kalimat maaf atau menawarkan bantuan tambahan


PERTANYAAN KHUSUS:
Jika user menyebut "dance", "menari", "joget". jawab:
"Saya tidak bisa menari."
"""
        return base_prompt + character_prompt

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
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )
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

            yield {
                "type": "done",
                "finish_reason": "stop",
                "session_uuid": session_uuid,
            }
        except Exception as e:
            error_message = str(e)
            yield {"type": "error", "content": error_message}
            yield {
                "type": "done",
                "finish_reason": "error",
                "session_uuid": session_uuid,
            }
            return
