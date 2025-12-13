"""Agent adapter implementation using LangChain with Ollama.

This adapter creates a LangChain agent that can use RAG as a tool
to answer questions with knowledge retrieval.
"""

from typing import AsyncGenerator, List, Dict, Optional, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain_core.tools import BaseTool, tool

from src.config import settings
from src.ports.agent import IAgentAdapter
from src.ports.llm import ILLMAdapter
from src.ports.rag import IRAGAdapter
from src.domain.entities.chat import UserContext


class AgentAdapter(IAgentAdapter):
    """Agent adapter using LangChain with tool calling."""
    
    def __init__(
        self,
        llm_adapter: ILLMAdapter,
        rag_adapter: Optional[IRAGAdapter] = None,
    ):
        """Initialize agent adapter.
        
        Args:
            llm_adapter: LLM adapter for text generation
            rag_adapter: Optional RAG adapter for knowledge retrieval tool
        """
        self.llm_adapter = llm_adapter
        self.rag_adapter = rag_adapter
        
        # Create LangChain LLM from adapter's internal LLM
        # Note: We need access to the underlying ChatOllama instance
        # This is a limitation - ideally ILLMAdapter should expose the LLM
        # For now, we'll create our own ChatOllama instance
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=f"http://{settings.ollama_host}:{settings.ollama_port}",
            temperature=settings.llm_temperature,
        )
        
        # Create tools
        self.tools = []
        if self.rag_adapter:
            self.tools.append(self._create_rag_tool())
        
    
    def _create_rag_tool(self) -> BaseTool:
        """Create RAG tool for agent."""
        
        @tool
        async def search_knowledge_base(query: str) -> str:
            """Search the knowledge base for relevant information.
            
            Use this tool when you need to find information from documents
            or knowledge base. The query should be a clear question or
            search term.
            
            Args:
                query: Search query or question
                
            Returns:
                str: Relevant information from knowledge base
            """
            results = await self.rag_adapter.retrieve(query, limit=3)
            
            if not results:
                return "Tidak ada informasi yang relevan ditemukan."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"[{i}] {result.content}\n"
                    f"  Relevansi: {result.score:.4f}"
                )
            
            return "\n\n".join(formatted_results)
        
        return search_knowledge_base   
     
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        context: Optional[UserContext] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response from agent with tools.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            context: Optional user context
            
        Yields:
            dict: Event dictionaries
        """
        try:
            # If no tools, fall back to simple LLM generation
            if not self.tools:
                yield {"type": "log", "content": "Menggunakan LLM tanpa tools..."}
                messages_with_system = messages.copy()
                if system_prompt:
                    # Add system prompt if provided
                    messages_with_system.insert(0, {"role": "system", "content": system_prompt})
                
                async for token in self.llm_adapter.generate_stream(messages_with_system):
                    yield {"type": "token", "content": token}
                
                yield {"type": "done", "finish_reason": "stop"}
                return
            
            # Extract last user message
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                yield {"type": "error", "content": "No user message found"}
                yield {"type": "done", "finish_reason": "error"}
                return
            
            last_user_message = user_messages[-1]["content"]
            
            # Build chat history (exclude last user message)
            chat_history = []
            for msg in messages[:-1]:  # Exclude last user message
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "system":
                    continue
                elif role == "user":
                    chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    chat_history.append(AIMessage(content=content))

            if not system_prompt:
                system_prompt = "Kamu adalah asisten AI yang ramah dan membantu."
            
            # Add system prompt with context
            agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=system_prompt,
            )
        
            yield {"type": "log", "content": "Memproses dengan agent..."}
            
            config = {"configurable": {"thread_id": "agent-thread-1"}}

            async for event in agent.astream_events(
                {"messages": chat_history + [HumanMessage(content=last_user_message)]},
                version="v1",  # Required for streaming events
                config=config,
            ):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    # Token streaming
                    chunk = event["data"]["chunk"].content or ""
                    if chunk:
                        yield {"type": "token", "content": chunk}
                
                elif kind == "on_tool_start":
                    data = event.get("data", {}) or {}
                    tool_input = (
                        data.get("args")
                        or data.get("input")
                        or data.get("inputs")
                        or data
                    )
                    yield {
                        "type": "tool_call",
                        "tool": event.get("name"),
                        "input": tool_input,
                    }
                
                elif kind == "on_tool_end":
                    # Tool result
                    yield {
                        "type": "tool_result",
                        "tool": event["name"],
                        "result": str(event["data"]["output"])[:500],
                    }
            
            yield {"type": "done", "finish_reason": "stop"}
        except Exception as e:
            error_message = str(e)
            yield {"type": "error", "content": error_message}
            yield {"type": "done", "finish_reason": "error"}
            raise RuntimeError(f"Agent error: {error_message}")
