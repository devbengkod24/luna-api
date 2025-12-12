"""LLM adapter implementation using LangChain v1 with Ollama.

This adapter provides streaming chat completions using local Ollama models.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import AsyncGenerator, Dict, List

from src.config import settings
from src.ports.llm import ILLMAdapter

class LLMAdapter(ILLMAdapter):
    """LLM adapter using LangChain v1 with Ollama."""
    
    def __init__(self, model: str = None):
        """Initialize LLM client with Ollama.

        Args:
            model: Model name
        """
        self.model = model
        
        self.llm = ChatOllama(
            model=self.model,
            base_url=f"http://{settings.ollama_host}:{settings.ollama_port}",
            temperature=settings.llm_temperature,
        )

        print(
            f"LLM Adapter initialized: model={self.model}, "
            f"ollama_url=http://{settings.ollama_host}:{settings.ollama_port}"
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                     Can include system message with role 'system'.

        Yields:
            str: Token chunks from the LLM
        """
        # Convert messages to LangChain format
        langchain_messages = []

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
            raise RuntimeError(f"LLM error: {str(e)}")

    async def generate(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Generate non-streaming response from Ollama LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                     Can include system message with role 'system'.

        Returns:
            str: Complete response from LLM
        """
        # Collect all chunks
        response = ""
        async for chunk in self.generate_stream(messages):
            response += chunk

        return response

    async def generate_title(self, title_prompt: str) -> str:
        """Generate a short title for a conversation.

        Args:
            title_prompt: Complete prompt for title generation (including instructions)

        Returns:
            str: Generated title
        """
        try:
            messages = [HumanMessage(content=title_prompt)]
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