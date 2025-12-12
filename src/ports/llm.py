"""LLM port interfaces."""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict


class ILLMAdapter(ABC):
    """Interface for Large Language Model adapter."""
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
                     Can include system message with role 'system'.
            
        Yields:
            str: Token chunks from the LLM
        """
        pass
    
    @abstractmethod
    async def generate_title(self, title_prompt: str) -> str:
        """Generate a short title for a conversation.
        
        Args:
            title_prompt: Complete prompt for title generation (including instructions)
            
        Returns:
            str: Generated title
        """
        pass

