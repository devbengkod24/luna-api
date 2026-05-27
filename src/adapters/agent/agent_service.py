import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage as LCAIMessage, HumanMessage, SystemMessage, ToolMessage as LCToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from nemoguardrails import LLMRails, RailsConfig
from sympy import SympifyError, sympify

from src.adapters.agent.middlewares.pseudo_tool_mitigation import strip_pseudo_tools_middleware
from src.config import settings
from src.domain.entities import (
    AIMessage,
    Function,
    Message,
    ResponseMetadata,
    ToolMessage,
    ToolsCall,
    UsageMetadata,
    UserContext,
    UserMessage,
)
from src.domain.events import (
    AssistantEvent,
    ErrorEvent,
    LogEvent,
    StreamEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from src.ports.services import IAgentService, IRAGService

import logging
logger = logging.getLogger(__name__)

_GUARDRAIL_PASS = "GUARDRAIL_PASS"
_GUARDRAILS_PATH = Path(__file__).parent / "guardrails"


class LangchainAgentService(IAgentService):
    """Agent adapter using LangChain with tool calling."""

    def __init__(
        self,
        rag_service: IRAGService,
    ):
        self.rag_service = rag_service

        self.llm = ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.ollama_temperature,
            top_p=settings.ollama_top_p,
            top_k=settings.ollama_top_k,
            min_p=settings.ollama_min_p,

            num_ctx=settings.ollama_num_ctx,
            num_predict=settings.ollama_max_tokens,
            num_batch=settings.ollama_num_batch,

            repeat_penalty=settings.ollama_repeat_penalty,
            repeat_last_n=settings.ollama_repeat_last_n,
        )

        self.tools = self._create_tools()
        try:
            self.agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt="",
                middleware=[strip_pseudo_tools_middleware],
            )
        except Exception as e:
            logger.error("Agent build failed at startup: %s", e)
            raise

        try:
            self.rails = LLMRails(RailsConfig.from_path(str(_GUARDRAILS_PATH)))
        except Exception as e:
            logger.warning("Guardrails init failed (running without guardrails): %s", e)
            self.rails = None

    def _create_tools(self):
        @tool
        def search_web(query: str) -> str:
            """Search for GENERAL information from the internet (current news, world facts, real-time info).

            Use this tool for:
            - Current news and information (president, weather, prices, etc)
            - General facts NOT related to Universitas Dian Nuswantoro
            - Information requiring real-time data from the internet

            DO NOT use for questions about Udinus campus.
            """
            try:
                search_tool = DuckDuckGoSearchRun()
                return search_tool.run(query)
            except Exception as e:
                return f"Web search error: {str(e)}"

        @tool
        def calculator(expression: str) -> str:
            """Calculate safe mathematical expressions (+, -, *, /, **, sin, cos, etc)."""
            try:
                expr = sympify(expression, evaluate=True)
                return f"{float(expr):.4f}"
            except SympifyError:
                return "Ekspresi tidak valid. Gunakan: 2+2, sin(30), sqrt(16)"
            except Exception as e:
                return f"Error: {str(e)}"

        @tool
        async def rag_search(query: str) -> str:
            """Search for INTERNAL information from Universitas Dian Nuswantoro (Udinus) knowledge base.

            Use this tool for:
            - University profile, faculties, study programs of Udinus
            - Student associations (HM), student organizations, campus organizations
            - Academic procedures, admissions, new student registration
            - Lecturer information, staff, campus organizational structure
            - Academic calendar, campus event schedules
            - Campus facilities, locations, infrastructure
            - Curriculum, courses, study program syllabi
            - Scholarships, financial aid, student support programs

            DO NOT use for general information outside Udinus context.
            """
            try:
                async with asyncio.timeout(settings.agent_tool_timeout_seconds):
                    results = await self.rag_service.retrieve(query, limit=3)

                if not results:
                    return "No relevant information found in Udinus knowledge base."

                # Format results (RetrievalResultEntity)
                formatted_results = []
                for i, result in enumerate(results, 1):
                    content = result.content if hasattr(result, "content") else result.get("content", "")
                    score = result.score if hasattr(result, "score") else result.get("score", 0.0)
                    formatted_results.append(
                        f"[{i}] {content}\n  Relevance: {score:.4f}"
                    )

                return "\n\n".join(formatted_results)
            except Exception as e:
                return f"RAG search error: {str(e)}"

        return [search_web, calculator, rag_search]

    async def generate_stream(
        self,
        user_input: str,
        messages: List[Message],
        context: Optional[UserContext] = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        try:
            chat_history = [self._to_lc_message(m) for m in messages]
            chat_history = [m for m in chat_history if m is not None]

            if self.rails:
                try:
                    check = await self.rails.generate_async(
                        messages=[{"role": "user", "content": user_input}]
                    )
                    content = check.get("content", "")
                    if content and content != _GUARDRAIL_PASS:
                        yield TokenEvent(content=content)
                        return
                except Exception as e:
                    logger.warning("Guardrails check error (skipping): %s", e)

            yield LogEvent(content="Memproses dengan agent...")

            config = {"configurable": {"thread_id": "agent-thread-1"}}

            messages_to_send: list = []
            if system_prompt:
                messages_to_send.append(SystemMessage(content=system_prompt))
            messages_to_send.extend(chat_history)
            messages_to_send.append(HumanMessage(content=user_input))

            async for event in self.agent.astream_events(
                {"messages": messages_to_send},
                config=config,
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content or ""
                    if chunk:
                        yield TokenEvent(content=chunk)

                elif kind == "on_tool_start":
                    yield ToolCallEvent(tool=event["name"], input=event["data"]["input"])

                elif kind == "on_tool_end":
                    tool_msg = event["data"]["output"]
                    yield ToolResultEvent(
                        content=tool_msg.content,
                        tool_call_id=tool_msg.tool_call_id,
                        name=tool_msg.name,
                    )

                elif kind == "on_chat_model_end":
                    ai_msg = event["data"]["output"]
                    yield AssistantEvent(message=self._to_domain_ai_message(ai_msg))
        except Exception:
            logger.exception("Agent stream failed")
            yield ErrorEvent(content="Agent error")
            return

    @staticmethod
    def _to_lc_message(message: Message):
        """Convert a domain Message to a LangChain message, or None if unsupported."""
        body = message.body
        if isinstance(body, UserMessage):
            return HumanMessage(content=body.content)
        if isinstance(body, AIMessage):
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (body.additional_kwargs or [])
            ]
            return LCAIMessage(
                content=body.content,
                additional_kwargs={"tool_calls": tool_calls},
            )
        if isinstance(body, ToolMessage):
            return LCToolMessage(
                content=body.content,
                tool_call_id=body.tool_call_id,
                name=body.name,
            )
        return None

    @staticmethod
    def _to_domain_ai_message(ai_msg: Any) -> AIMessage:
        """Convert a LangChain AIMessage into the domain AIMessage entity."""
        rm = ai_msg.response_metadata or {}
        um = getattr(ai_msg, "usage_metadata", {}) or {}
        tool_calls = [
            ToolsCall(
                id=tc["id"],
                type="function",
                function=Function(name=tc["name"], arguments=tc["args"]),
            )
            for tc in (ai_msg.tool_calls or [])
        ]
        return AIMessage(
            content=ai_msg.content,
            additional_kwargs=tool_calls,
            response_metadata=ResponseMetadata(
                model=rm.get("model", ""),
                done_reason=rm.get("done_reason", ""),
            ),
            usage_metadata=UsageMetadata(
                input_tokens=um.get("input_tokens", 0),
                output_tokens=um.get("output_tokens", 0),
                total_tokens=um.get("total_tokens", 0),
            ),
        )

    async def generate_title(self, title_prompt: str) -> str:
        """Generate a short title for a conversation."""
        try:
            messages = [HumanMessage(content=title_prompt)]
            response = await self.llm.ainvoke(messages)

            title = response.content.strip()
            title = title.strip("\"'")
            if len(title) > 50:
                title = title[:47] + "..."

            return title or "Obrolan Baru"

        except Exception as e:
            logger.error("Error generating title: %s", e)
            return "Obrolan Baru"

    async def preprocess_tts(self, text: str, system_prompt: str) -> str:
        """Preprocess text for TTS."""
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]
            response = await self.llm.ainvoke(messages)

            return response.content.strip()

        except Exception as e:
            logger.error("Error preprocess text for TTS: %s", e)
            return text
