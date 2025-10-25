"""LLM port interfaces."""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any


class ILLMAdapter(ABC):
    """Interface for Large Language Model adapter."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def generate_title(self, first_message: str) -> str:
        """Generate a short title for a conversation.
        
        Args:
            first_message: First user message in the conversation
            
        Returns:
            str: Generated title
        """
        pass

