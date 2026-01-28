"""Agent adapter implementation using LangChain with Ollama.

This adapter creates a LangChain agent that can use RAG as a tool
to answer questions with knowledge retrieval.
"""

import math
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from sympy import SympifyError, sympify

from src.config import settings
from src.domain.entities.chat import UserContext
from src.ports.agent import IAgentAdapter
from src.ports.rag import IRAGAdapter

import logging
logger = logging.getLogger(__name__)


class AgentAdapter(IAgentAdapter):
    """Agent adapter using LangChain with tool calling."""

    def __init__(
        self,
        model: str = None,
        rag_adapter: Optional[IRAGAdapter] = None,
        web_search_enabled: bool = False,
    ):
        """Initialize agent adapter.

        Args:
            rag_adapter: Optional RAG adapter for knowledge retrieval tool
        """
        self.rag_adapter = rag_adapter

        self.llm = ChatOllama(
            model=model,
            base_url=f"http://{settings.ollama_host}:{settings.ollama_port}",
            temperature=settings.llm_temperature,
        )

        self.tools = self._create_tools()
        # self.web_search = DuckDuckGoSearchRun(max_results=3)
        self.agent = None

    def _create_tools(self):
        """Tools yang bisa dipakai agent."""

        @tool
        def search_web(query: str) -> str:
            """Cari informasi UMUM dari internet (berita terkini, fakta dunia, info real-time).

            Gunakan tool ini untuk:
            - Berita dan informasi terkini (presiden, cuaca, harga, dll)
            - Fakta umum yang TIDAK terkait dengan Universitas Dian Nuswantoro
            - Informasi yang membutuhkan data real-time dari internet

            JANGAN gunakan untuk pertanyaan tentang kampus Udinus.
            """
            return DuckDuckGoSearchRun(max_results=5).run(query)

        @tool
        def calculator(expression: str) -> str:
            """Hitung ekspresi matematika aman ( +, -, *, /, **, sin, cos, etc )."""
            try:
                expr = sympify(expression, evaluate=True)
                return f"{float(expr):.4f}"
            except SympifyError:
                return "Ekspresi tidak valid. Gunakan: 2+2, sin(30), sqrt(16)"
            except Exception as e:
                return f"Error: {str(e)}"

        @tool
        async def rag_search(query: str) -> str:
            """Cari informasi INTERNAL dari knowledge base Universitas Dian Nuswantoro (Udinus).

            Gunakan tool ini untuk:
            - Profil universitas, fakultas, program studi Udinus
            - Himpunan Mahasiswa (HM), UKM, organisasi kampus
            - Prosedur akademik, admisi, pendaftaran mahasiswa baru
            - Informasi dosen, staff, struktur organisasi kampus
            - Kalender akademik, jadwal kegiatan kampus
            - Fasilitas kampus, lokasi, sarana prasarana
            - Kurikulum, mata kuliah, silabus program studi
            - Beasiswa, bantuan keuangan, program dukungan mahasiswa

            JANGAN gunakan untuk informasi umum di luar konteks Udinus.
            """
            results = await self.rag_adapter.retrieve(query, limit=3)

            logger.info(f"rag_search tool called with query: '{query}'")
            logger.info(f"Retrieved {len(results)} results")
            for i, result in enumerate(results, 1):
                logger.info(f"  Result {i}: score={result.score:.4f}, content_preview={result.content[:200]}...")

            if not results:
                return "Tidak ada informasi yang relevan ditemukan di knowledge base Udinus."

            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"[{i}] {result.content}\n" f"  Relevansi: {result.score:.4f}"
                )

            return "\n\n".join(formatted_results)

        return [search_web, calculator, rag_search]

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
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                yield {"type": "error", "content": "No user message found"}
                yield {"type": "done", "finish_reason": "error"}
                return

            last_user_message = user_messages[-1]["content"]

            chat_history = []
            for msg in messages[:-1]:
                role = msg.get("role")
                content = msg.get("content", "")

                if role == "system":
                    chat_history.append(SystemMessage(content=content))
                elif role == "user":
                    chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    chat_history.append(AIMessage(content=content))

            self.agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=system_prompt,
            )

            yield {"type": "log", "content": "Memproses dengan agent..."}

            config = {"configurable": {"thread_id": "agent-thread-1"}}

            async for event in self.agent.astream_events(
                {"messages": chat_history + [HumanMessage(content=last_user_message)]},
                config=config,
                version="v1",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
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
                    yield {
                        "type": "tool_result",
                        "tool": event["name"],
                        "result": str(event["data"]["output"]),
                    }

            yield {"type": "done", "finish_reason": "stop"}
        except Exception as e:
            error_message = str(e)
            yield {"type": "error", "content": error_message}
            yield {"type": "done", "finish_reason": "error"}
            raise RuntimeError(f"Agent error: {error_message}")

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
            print(f"Error generating title: {e}")
            return "Obrolan Baru"
