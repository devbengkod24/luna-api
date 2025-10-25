"""Chat text use case."""
import uuid
from typing import AsyncGenerator, Optional

from src.ports.repository import IChatRepository
from src.ports.llm import ILLMAdapter
from src.ports.rag import IRAGAdapter
from src.domain.entities.context import UserContext


class ChatTextUseCase:
    """Use case for text-based chat with streaming response."""
    
    def __init__(
        self,
        chat_repository: IChatRepository,
        llm_adapter: ILLMAdapter,
        rag_adapter: Optional[IRAGAdapter] = None,
    ):
        """Initialize use case with dependencies.
        
        Args:
            chat_repository: Repository for chat persistence
            llm_adapter: LLM adapter for response generation
            rag_adapter: Optional RAG adapter for knowledge retrieval
        """
        self.chat_repository = chat_repository
        self.llm_adapter = llm_adapter
        self.rag_adapter = rag_adapter
    
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
        # Step 1: Get or create session
        if session_uuid:
            session = await self.chat_repository.get_session_by_uuid(session_uuid)
            if not session:
                yield {"type": "error", "content": "Session not found"}
                return
        else:
            # Generate new session UUID
            session_uuid = str(uuid.uuid4())
            
            # Generate title from first message
            yield {"type": "log", "content": "Membuat sesi baru..."}
            title = await self.llm_adapter.generate_title(prompt)
            
            session = await self.chat_repository.create_session(
                uuid=session_uuid,
                title=title,
            )
            
            yield {"type": "log", "content": f"Sesi dibuat: {title}"}
        
        # Step 2: Add user message to session
        await self.chat_repository.add_message(
            session_id=session.id,
            role="user",
            content=prompt,
            context=context.to_dict(),
        )
        
        # Step 3: Retrieve relevant knowledge (if RAG enabled)
        if self.rag_adapter:
            yield {"type": "log", "content": "Mencari pengetahuan relevan..."}
            knowledge = await self.rag_adapter.search(prompt, limit=3)
            # Add knowledge to context (simplified)
            # In real implementation, you would format this better
        
        # Step 4: Build message history
        yield {"type": "log", "content": "Menganalisis konteks..."}
        messages = []
        for msg in session.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        messages.append({"role": "user", "content": prompt})
        
        # Step 5: Generate streaming response
        full_response = ""
        async for token in self.llm_adapter.generate_stream(messages, context.to_dict()):
            full_response += token
            yield {"type": "token", "content": token}
        
        # Step 6: Save assistant response
        await self.chat_repository.add_message(
            session_id=session.id,
            role="assistant",
            content=full_response,
        )
        
        # Step 7: Send completion event
        yield {"type": "done", "session_uuid": session_uuid}

