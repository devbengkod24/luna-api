"""LLM adapter implementation using LangChain v1 with Ollama.

This adapter provides streaming chat completions using local Ollama models.
"""

from typing import Any, AsyncGenerator, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from src.config import settings
from src.ports.llm import ILLMAdapter


class LLMAdapter(ILLMAdapter):
    """LLM adapter using LangChain v1 with Ollama."""

    def __init__(self, api_key: str = None, model: str = None):
        """Initialize LLM client with Ollama.

        Args:
            api_key: Not used for Ollama (local model)
            model: Model name (defaults to settings.llm_model)
        """
        self.model = model or settings.llm_model

        # Initialize ChatOllama from langchain-ollama v1
        self.llm = ChatOllama(
            model=self.model,
            base_url=f"http://{settings.ollama_host}:{settings.ollama_port}",
            temperature=0.7,
        )

        print(
            f"LLM Adapter initialized: model={self.model}, "
            f"ollama_url=http://{settings.ollama_host}:{settings.ollama_port}"
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            context: User context (emotion, name, etc.)

        Yields:
            str: Token chunks from the LLM
        """
        # Build system prompt with context
        system_prompt = self._build_system_prompt(context)

        # Convert messages to LangChain format
        langchain_messages = [SystemMessage(content=system_prompt)]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))

        # Stream response from Ollama
        try:
            async for chunk in self.llm.astream(langchain_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"LLM streaming error: {e}")
            yield f"Error: {str(e)}"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        context: Dict[str, Any] = None,
    ) -> str:
        """Generate non-streaming response from Ollama LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            context: User context (emotion, name, etc.)

        Returns:
            str: Complete response from LLM
        """
        context = context or {}

        # Collect all chunks
        response = ""
        async for chunk in self.generate_stream(messages, context):
            response += chunk

        return response

    async def generate_title(self, first_message: str) -> str:
        """Generate a short title for a conversation.

        Args:
            first_message: First user message in the conversation

        Returns:
            str: Generated title (max 5 words)
        """
        prompt = f"""Buatkan judul singkat (maksimal 5 kata) dalam Bahasa Indonesia untuk percakapan yang dimulai dengan pesan berikut:

"{first_message}"

Jawab hanya dengan judul saja, tanpa tanda kutip atau penjelasan tambahan."""

        try:
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)

            # Clean up the title
            title = response.content.strip()
            # Remove quotes if present
            title = title.strip("\"'")
            # Limit to reasonable length
            if len(title) > 50:
                title = title[:47] + "..."

            return title or "Obrolan Baru"

        except Exception as e:
            print(f"Error generating title: {e}")
            return "Obrolan Baru"

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
