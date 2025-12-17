"""Agent port interfaces."""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Optional, Any

from src.domain.entities.chat import UserContext


class IAgentAdapter(ABC):
    """Interface for AI Agent adapter (with tools).
    
    Agent is an orchestration layer that can use tools (e.g., RAG) 
    to answer questions. This interface abstracts away the specific
    framework implementation (LangChain, AutoGPT, etc.).
    """
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        context: Optional[UserContext] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response from agent with tools.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
                     Can include system message with role 'system'.
            context: Optional user context (emotion, name, etc.)
            
        Yields:
            dict: Event dictionaries with the following types:
                - {"type": "log", "content": "..."} - Log/info message
                - {"type": "tool_call", "tool": "...", "input": "..."} - Tool being called
                - {"type": "tool_result", "tool": "...", "result": "..."} - Tool result
                - {"type": "token", "content": "..."} - Token chunk from LLM
                - {"type": "done", "finish_reason": "stop|error", ...} - Stream complete
                - {"type": "error", "content": "..."} - Error occurred
        """
        ...