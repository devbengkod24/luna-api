"""Base chat use case with common logic."""

import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional

from src.domain.entities.chat import Character, UserContext
from src.ports.services import IAgentAdapter
from src.ports.repositories import IChatRepository


class BaseChatUseCase(ABC):
    """Base use case for chat with streaming response."""

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

    def _build_system_prompt(
        self, character: Character, context: UserContext | None
    ) -> str:
        """Build system prompt with context.

        Args:
            character: Character
            context: User context (can be None if all fields are null)

        Returns:
            str: System prompt
        """
        if context is None:
            emotion = "neutral"
            name = "Pengguna"
            nim = None
        else:
            emotion = context.emotion or "neutral"
            name = context.name or "Pengguna"
            nim = context.nim

        # Base prompt yang sama untuk semua karakter
        base_prompt = f"""
Kamu adalah Bengbot {character}, asisten AI di Universitas Dian Nuswantoro (Udinus) yang dibuat oleh Bengkel Koding.
Kamu bisa menjawab dalam bahasa Indonesia dan bahasa Inggris.

Kamu akan mendapatkan messages dari role user dengan messages nya adalah isi dari content.
Jawablah message dari role user tersebut sesuai dengan karakter kamu.
Ketika isi message nya menyebutkan kata ganti orang ke kedua (kamu, anda, loe dll), anggap itu merujuk pada diri kamu sebagai Bengbot. ketika isi message menyebutkan kata ganti orang pertama (saya, aku, gue dll), anggap itu merujuk pada user/pengguna yang mengirim message tersebut.

KONTEKS PENGGUNA ATAU USER:
- Nama: {name}
- NIM: {nim if nim else "Tidak ada"}
- Emosi: {emotion}


=== KONSEP PENTING: CASUAL vs INFORMATIONAL ===

CASUAL CHAT (TANPA TOOL):
- Sapaan: "halo", "hai", "selamat pagi", "apa kabar", "siapa kamu"
- Pertanyaan tentang pengguna: "siapa aku", "siapa namaku", "siapa saya", "siapa nama saya", "apa nim saya", "nim saya berapa", "apa emosi saya", "apa emosi aku"
- Pujian: "kamu lucu", "kamu keren", "kamu pintar"
- Ekspresi perasaan: "aku senang", "aku bingung", "aku capek"
- Small talk: "lagi ngapain", "gimana nih", "kok gitu sih"
- Opini/pertanyaan ringan: "menurutmu gimana", "kamu suka apa"
- Bercanda/guyonan: "wkwk", "haha", "lucu deh kamu"

INFORMATIONAL (PAKAI TOOL):
- Pertanyaan spesifik dengan kata tanya TENTANG ORANG/TOPIK LAIN: "siapa itu Jokowi", "siapa presiden Indonesia"
- Pertanyaan spesifik dengan kata tanya: "apa", "kapan", "dimana", "berapa"
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
- PERTANYAAN TENTANG IDENTITAS USER: "siapa aku", "siapa saya", "apa nama saya", "nim saya", "nim saya berapa", "apa emosi saya", "apa emosi aku"
- Untuk pertanyaan identitas user, LANGSUNG jawab dari KONTEKS PENGGUNA di atas
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
✓ JANGAN pakai tool untuk pertanyaan identitas diri user (siapa aku, nama saya, nim saya) - LANGSUNG jawab dari KONTEKS PENGGUNA
✓ SELALU pakai search_web untuk: nama orang, presiden, menteri, artis, atlet, liga olahraga, berita
✓ SELALU pakai rag_search untuk: info Udinus, kampus, prodi, dosen, mahasiswa, HM, UKM, beasiswa
✓ Kalau tidak yakin tentang fakta → pakai search_web, jangan mengarang!
✓ Kalau tool tidak menemukan jawaban → akui tidak tahu, jangan mengarang!
✓ **Jika rag_search menghasilkan teks apa pun, JANGAN buka dengan "tidak tahu"; langsung jawab memakai hasil tersebut secara ringkas. Hanya ucapkan "tidak tahu" bila hasil benar-benar kosong.**
✓ **PENTING: Setelah memanggil tool, WAIT dan BACA hasil tool dengan seksama sebelum merespon. JANGAN langsung bilang "tidak tahu" jika tool sudah dipanggil - tunggu hasilnya dulu!**
✓ **Jika tool mengembalikan hasil, GUNAKAN hasil tersebut untuk menjawab. Jangan abaikan hasil tool!**
✓ **Jika hasil tool berisi informasi yang relevan, jawablah berdasarkan informasi tersebut. Jangan bilang "tidak bisa menemukan" jika tool sudah mengembalikan hasil.**

MINDSET: Kamu adalah TEMAN yang membantu, bukan mesin pencari!
"""

        character_prompt = self._get_character_prompt(character.value)
        return base_prompt + character_prompt

    def _get_character_prompt(self, character: str) -> str:
        """Get character-specific prompt.

        Args:
            character: Character name (russell or fredricksen)

        Returns:
            str: Character-specific prompt
        """
        if character == "russell":
            return """
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
        elif character == "fredricksen":
            return """
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
        else:
            # Default to russell
            return self._get_character_prompt("russell")

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

    async def _get_or_create_session(
        self,
        session_uuid: Optional[str],
        first_message: str,
    ) -> tuple[Any, str]:
        """Get existing session or create new one.

        Args:
            session_uuid: Optional session UUID
            first_message: First message for title generation if creating new session

        Returns:
            tuple: (session object, session_uuid)
        """
        if session_uuid:
            session = await self.chat_repository.get_session_by_uuid(session_uuid)
            if not session:
                return None, session_uuid
            return session, session_uuid
        else:
            # Generate new session UUID
            session_uuid = str(uuid.uuid4())

            # Generate title from first message
            title_prompt = self._build_title_prompt(first_message)
            title = await self.agent_adapter.generate_title(title_prompt)

            session = await self.chat_repository.create_session(
                uuid=session_uuid,
                title=title,
            )
            return session, session_uuid

    async def _build_message_history(
        self,
        session: Any,
        current_message: str,
    ) -> list[dict]:
        """Build message history from session.

        Args:
            session: Session object
            current_message: Current user message

        Returns:
            list: List of message dictionaries
        """
        messages = []
        for msg in session.messages:
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )
        messages.append({"role": "user", "content": current_message})
        return messages

    async def _execute_chat(
        self,
        session_uuid: Optional[str],
        character: Character,
        prompt: str,
        context: UserContext | None,
    ) -> AsyncGenerator[dict, None]:
        """Execute common chat logic with streaming.

        Args:
            session_uuid: Optional session UUID
            character: Character
            prompt: User prompt
            context: User context

        Yields:
            dict: Streaming events (log, token, done, error)
        """
        # Step 1: Get or create session
        session, session_uuid = await self._get_or_create_session(session_uuid, prompt)

        if not session:
            yield {"type": "error", "content": "Session not found"}
            return

        # If it's a new session, yield log
        if not session_uuid:
            yield {"type": "log", "content": "Membuat sesi baru..."}

        # Step 2: Add user message to session
        await self.chat_repository.add_message(
            session_id=session.id,
            role="user",
            content=prompt,
            context=context.to_dict() if context else None,
        )

        # Step 3: Build message history
        messages = await self._build_message_history(session, prompt)

        # Step 4: Build system prompt
        system_prompt = self._build_system_prompt(character, context)

        # Step 5: Generate streaming response
        yield {"type": "log", "content": "Menganalisis konteks..."}

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

            # Step 6: Save assistant response
            await self.chat_repository.add_message(
                session_id=session.id,
                role="assistant",
                content=full_response,
            )

            # Step 7: Send completion event
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
