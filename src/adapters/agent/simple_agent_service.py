"""SimpleAgentService — port dari reference/main.py untuk FastAPI backend.

Arsitektur ini menggantikan LangchainAgentService (yang bergantung pada
create_agent / LangGraph) dengan implementasi imperative loop yang lebih stabil.

Pipeline per turn:
  1. Panggil LLM dengan tools terikat (bind_tools).
  2. Jika model memilih tool → eksekusi tool, panggil LLM lagi.
  3. Jika model menjawab langsung → emit token events ke generator.

Perbedaan dari reference/main.py:
  - Tidak ada print() ke terminal, semua output via AsyncGenerator events.
  - generate_stream() emit events {type: token|tool|tool_call|assistant|log|error}.
  - generate_title() dipertahankan untuk membuat judul percakapan.
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from sympy import SympifyError, sympify

from src.config import settings
from src.domain.entities import UserContext
from src.ports.services import IAgentService, IRAGService
from src.shared.constants import MessageType

logger = logging.getLogger(__name__)


# ── Pseudo-tool names Ollama emits when it wants to answer directly ──────────
# Mirroring reference/main.py _PSEUDO_TOOLS filter.
# Ollama terkadang memancarkan '{"name": "no tool", ...}' sebagai teks biasa
# ketika seharusnya menjawab langsung — ini harus difilter, bukan dieksekusi.
_PSEUDO_TOOLS: frozenset[str] = frozenset(
    {
        "answer directly",
        "answer_directly",
        "direct_answer",
        "no_tool",
        "no tool",
        "none",
        "empathy",
        "respond_with_empathy",
        "respond with empathy",
        "casual",
        "casual_response",
        "direct",
        "skip_tool",
    }
)


def _is_pseudo_tool_content(content: str) -> bool:
    """Return True jika `content` adalah JSON pseudo-tool payload.

    Ollama terkadang emit misal {'name': 'no tool', 'parameters': {}}
    sebagai plain text. Kita detect dan discard.
    """
    stripped = (content or "").strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped)
        name = (data.get("name") or "").strip().lower()
        return name in _PSEUDO_TOOLS
    except Exception:
        return False


def _create_llm() -> ChatOllama:
    """Buat ChatOllama dengan semua sampling parameters dari reference experiment.

    Parameter di-mirror dari GPT4All defaults untuk kualitas jawaban faktual.
    """
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
        num_ctx=settings.ollama_num_ctx,          # context window
        num_predict=settings.ollama_max_tokens,   # max output tokens
        # --- Sampling parameters ---
        temperature=settings.ollama_temperature,  # 0.3 → sedikit kreatif, natural
        top_k=settings.ollama_top_k,              # 40  → batasi pool kandidat token
        top_p=settings.ollama_top_p,              # 0.6 → nucleus sampling
        min_p=settings.ollama_min_p,              # 0.0 → threshold probabilitas minimum
        repeat_penalty=settings.ollama_repeat_penalty,  # 1.18 → cegah output berulang
        repeat_last_n=settings.ollama_repeat_last_n,    # 64   → window pengecekan
        num_batch=settings.ollama_num_batch,      # 128  → token paralel per batch
    )


class SimpleAgent:
    """Minimal tool-calling agent — tidak bergantung pada LangGraph.

    Port dari reference/main.py dengan adaptasi untuk FastAPI:
    - Tidak ada print() ke terminal
    - Output via callback events dict (token, tool, assistant, error)
    """

    def __init__(self, llm: ChatOllama, tools: list, system_prompt: str) -> None:
        self.tool_map: Dict[str, object] = {t.name: t for t in tools}
        self.llm_raw = llm               # tanpa tools — untuk fallback direct answer
        self.llm = llm.bind_tools(tools) # dengan tools — untuk tool-calling
        self.system_msg = SystemMessage(content=system_prompt)

    async def ainvoke(
        self,
        messages: list,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run one agent turn, yielding events untuk streaming ke client.

        Events yang di-emit:
          {"type": "token",     "content": str}         — token streaming jawaban akhir
          {"type": "tool_call", "tool": str, "input": dict} — sebelum eksekusi tool
          {"type": "tool",      "content": str, "tool_call_id": str, "name": str}
          {"type": "assistant", "content": str, "additional_kwargs": dict,
           "response_metadata": dict, "usage_metadata": dict}
          {"type": "error",     "content": str}
        """
        all_msgs = [self.system_msg] + messages

        # ── Round 1: tanya LLM, cek apakah butuh tool ────────────────────────
        ai_response = await self.llm.ainvoke(all_msgs)
        raw_tool_calls = getattr(ai_response, "tool_calls", None) or []

        # Filter pseudo-tools (Ollama-internal signals, not real tools)
        tool_calls = [
            tc
            for tc in raw_tool_calls
            if (tc.get("name") or "").strip().lower() not in _PSEUDO_TOOLS
        ]

        # ── Jawaban langsung (tidak pakai tool) ──────────────────────────────
        if not tool_calls:
            answer = (ai_response.content or "").strip()

            # Deteksi: model output JSON pseudo-tool sebagai teks
            # Fallback: panggil LLM tanpa tools untuk dapat jawaban teks asli.
            if not answer or _is_pseudo_tool_content(answer):
                fallback = await self.llm_raw.ainvoke(all_msgs)
                answer = (getattr(fallback, "content", None) or "").strip()
                ai_response = fallback

            # Emit token events (streaming simulasi karena sudah non-stream ainvoke)
            if answer:
                yield {"type": "token", "content": answer}

            # Emit assistant event untuk disimpan ke database
            yield {
                "type": "assistant",
                "content": answer,
                "additional_kwargs": {"tool_calls": []},
                "response_metadata": getattr(ai_response, "response_metadata", {}),
                "usage_metadata": getattr(ai_response, "usage_metadata", {}),
            }
            return

        # ── Eksekusi tool calls ───────────────────────────────────────────────
        all_msgs.append(ai_response)

        # Emit assistant event (round 1 — berisi tool_calls)
        tool_calls_payload = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": tc.get("args", {}),
                },
            }
            for tc in tool_calls
        ]
        yield {
            "type": "assistant",
            "content": ai_response.content or "",
            "additional_kwargs": {"tool_calls": tool_calls_payload},
            "response_metadata": getattr(ai_response, "response_metadata", {}),
            "usage_metadata": getattr(ai_response, "usage_metadata", {}),
        }

        for tc in tool_calls:
            tc_name = tc.get("name", "?")
            tc_args = tc.get("args", {})
            tc_id = tc.get("id", "")

            # Beberapa model membungkus args dalam {"input": ...}
            if isinstance(tc_args, dict) and list(tc_args.keys()) == ["input"]:
                tc_args = tc_args["input"]

            # Emit tool_call event (sebelum eksekusi)
            yield {
                "type": "tool_call",
                "tool": tc_name,
                "input": tc_args,
            }

            tool_obj = self.tool_map.get(tc_name)
            if tool_obj is None:
                result_str = f"Unknown tool: {tc_name}"
                logger.warning("Unknown tool called: %s", tc_name)
            else:
                try:
                    result = await tool_obj.ainvoke(tc_args)
                    result_str = str(result)
                except Exception as e:
                    result_str = f"Tool error: {e}"
                    logger.exception("Error executing tool %s: %s", tc_name, e)

            # Emit tool result event (untuk disimpan ke database)
            yield {
                "type": "tool",
                "content": result_str,
                "tool_call_id": tc_id,
                "name": tc_name,
            }

            all_msgs.append(
                ToolMessage(content=result_str, tool_call_id=tc_id)
            )

        # ── Round 2: jawaban akhir setelah tool ───────────────────────────────
        buffer = ""
        # Stream token per token untuk jawaban akhir
        async for chunk in self.llm.astream(all_msgs):
            token = getattr(chunk, "content", "") or ""
            if token:
                yield {"type": "token", "content": token}
                buffer += token

        # Emit final assistant event untuk persisted answer
        yield {
            "type": "assistant",
            "content": buffer.strip(),
            "additional_kwargs": {"tool_calls": []},
            "response_metadata": {},
            "usage_metadata": {},
        }


class SimpleAgentService(IAgentService):
    """Agent service menggunakan SimpleAgent — menggantikan LangchainAgentService.

    Tidak bergantung pada LangGraph/create_agent yang menyebabkan ImportError.
    """

    def __init__(self, rag_service: IRAGService) -> None:
        self.rag_service = rag_service
        self.llm = _create_llm()

    def _create_tools(self, rag_service: IRAGService) -> list:
        @tool
        def search_web(query: str) -> str:
            """Search general web information (news, facts, real-time) via DuckDuckGo."""
            try:
                return DuckDuckGoSearchRun(max_results=5).run(query)
            except Exception as e:
                return f"Web search error: {str(e)}"

        @tool
        def calculator(expression: str) -> str:
            """Evaluate a safe mathematical expression (+, -, *, /, **, sin, cos, etc.)."""
            try:
                expr = sympify(expression, evaluate=True)
                return f"{float(expr):.4f}"
            except SympifyError:
                return "Invalid expression. Examples: 2+2, sin(30), sqrt(16)."
            except Exception as e:
                return f"Error: {str(e)}"

        @tool
        async def rag_search(query: str) -> str:
            """Search the internal Udinus knowledge base using RAG."""
            try:
                results = await rag_service.retrieve(query, limit=8)
                if not results:
                    return "No relevant information found in the Udinus knowledge base."

                formatted_results = []
                for i, result in enumerate(results, 1):
                    content = (
                        result.content
                        if hasattr(result, "content")
                        else result.get("content", "")
                    )
                    score = (
                        result.score
                        if hasattr(result, "score")
                        else result.get("score", 0.0)
                    )
                    formatted_results.append(f"[{i}] {content}\n  Relevance: {score:.4f}")

                return "\n\n".join(formatted_results)
            except Exception as e:
                return f"RAG search error: {str(e)}"

        return [search_web, calculator, rag_search]

    async def generate_stream(
        self,
        user_input: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        context: Optional[UserContext] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream agent response events ke caller.

        Events: token | tool_call | tool | assistant | log | error
        """
        try:
            # Build chat history dari messages DB
            chat_history = []
            for msg in messages:
                msg_type = msg.get("type")
                content = msg.get("content", "")

                if msg_type == MessageType.USER:
                    chat_history.append(HumanMessage(content=content))
                elif msg_type == MessageType.ASSISTANT:
                    tool_calls_raw = msg.get("additional_kwargs", {}).get("tool_calls", [])
                    chat_history.append(
                        AIMessage(
                            content=content,
                            additional_kwargs={"tool_calls": tool_calls_raw},
                        )
                    )
                elif msg_type == MessageType.TOOL:
                    chat_history.append(
                        ToolMessage(
                            content=content,
                            tool_call_id=msg.get("tool_call_id", ""),
                            name=msg.get("name", ""),
                        )
                    )

            tools = self._create_tools(self.rag_service)
            agent = SimpleAgent(
                llm=self.llm,
                tools=tools,
                system_prompt=system_prompt,
            )

            yield {"type": "log", "content": "Memproses dengan agent..."}

            # user_input SELALU ditambahkan ke ujung chat_history.
            # Tanpa ini, LLM kehilangan konteks pertanyaan terbaru ketika
            # ada existing history, dan akan mengabaikan tool results.
            send_messages = chat_history + [HumanMessage(content=user_input)]

            async for event in agent.ainvoke(send_messages):
                yield event

        except Exception as e:
            error_message = str(e)
            logger.exception("SimpleAgent error: %s", error_message)
            yield {"type": "error", "content": error_message}

    async def generate_title(self, title_prompt: str) -> str:
        """Generate a short title for a conversation."""
        try:
            messages = [HumanMessage(content=title_prompt)]
            response = await self.llm.ainvoke(messages)
            title = response.content.strip().strip("\"'")
            if len(title) > 50:
                title = title[:47] + "..."
            return title or "Obrolan Baru"
        except Exception as e:
            logger.error("Error generating title: %s", e)
            return "Obrolan Baru"
