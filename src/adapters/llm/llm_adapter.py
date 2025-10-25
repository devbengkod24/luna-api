"""LLM adapter implementation template."""
from typing import AsyncGenerator, List, Dict, Any

from src.ports.llm import ILLMAdapter


class LLMAdapter(ILLMAdapter):
    """LLM adapter - integrate with your LLM provider (OpenAI, Anthropic, local models, etc.)."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """Initialize LLM client.
        
        Args:
            api_key: API key for LLM provider
            model: Model name
        """
        self.api_key = api_key
        self.model = model
        # TODO: Initialize your LLM client here
        # self.client = OpenAI(api_key=api_key)
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            context: User context (emotion, name, etc.)
            
        Yields:
            str: Token chunks from the LLM
        """
        # TODO: Implement streaming generation with your LLM provider
        # Example with OpenAI:
        # system_prompt = self._build_system_prompt(context)
        # full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        # response = await self.client.chat.completions.create(
        #     model=self.model,
        #     messages=full_messages,
        #     stream=True,
        # )
        
        # async for chunk in response:
        #     if chunk.choices[0].delta.content:
        #         yield chunk.choices[0].delta.content
        
        # Placeholder response
        yield "Ini adalah respons placeholder dari LLM. "
        yield "Silakan implementasikan adapter LLM sesuai dengan provider Anda."
    
    async def generate_title(self, first_message: str) -> str:
        """Generate a short title for a conversation.
        
        Args:
            first_message: First user message in the conversation
            
        Returns:
            str: Generated title
        """
        # TODO: Implement title generation
        # prompt = f"Generate a short title (max 5 words) for a conversation starting with: {first_message}"
        # response = await self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        # )
        # return response.choices[0].message.content.strip()
        
        # Placeholder
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
        
        return f"""Anda adalah asisten AI yang ramah dan membantu.
Konteks percakapan:
- Nama pengguna: {name}
- Emosi saat ini: {emotion}

Sesuaikan respons Anda dengan emosi pengguna."""

