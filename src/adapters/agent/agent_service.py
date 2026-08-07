import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator
from langchain.agents import create_agent
from langchain_core.messages import AIMessage as LCAIMessage, HumanMessage, SystemMessage, ToolMessage as LCToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from nemoguardrails import LLMRails, RailsConfig

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
    SourcesEvent,
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

            reasoning=settings.ollama_reasoning or None,
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
        web_search_fn = self._build_web_search()

        @tool(response_format="content_and_artifact")
        def search_web(query: str, scope: str = "general"):
            """Search the web for information you do not reliably know.

            scope="campus" — the question is about Universitas Dian Nuswantoro
            (Udinus): rector, leadership, lecturers, faculties, study programs,
            tuition, registration, scholarships, schedules, facilities, contacts.
            Searches ONLY official Udinus sites. Write the query in INDONESIAN.

            scope="general" — anything else: national/world news, public figures,
            prices, general knowledge, current events.

            Pass a standalone, keyword-rich query — resolve pronouns and
            follow-up context from the conversation into an explicit query.
            """
            # Returns (text_for_llm, sources) — sources ride on ToolMessage.artifact,
            # unseen by the LLM, and are surfaced as a SourcesEvent for the client.
            if web_search_fn is None:
                return "Web search is not configured.", []
            try:
                return web_search_fn(query, scope)
            except Exception as e:
                logger.warning("Web search failed (scope=%s): %s", scope, e)
                return f"Web search error: {str(e)}", []

        @tool
        async def rag_search(query: str) -> str:
            """Search the Universitas Dian Nuswantoro (Udinus) knowledge base.

            Use this tool for Udinus-specific information:
            - profile, faculties, study programs
            - tuition, registration, heregistration
            - scholarships, financial aid
            - facilities, academic rules, campus organizations

            Pass a standalone, keyword-rich query — resolve pronouns and follow-up
            context from the conversation into an explicit query.
            DO NOT use for general information outside the Udinus context.
            """
            try:
                async with asyncio.timeout(settings.agent_tool_timeout_seconds):
                    results = await self.rag_service.retrieve(query, limit=settings.rag_top_k)
                if not results:
                    return "No relevant information found in Udinus knowledge base."
                formatted = []
                for i, result in enumerate(results, 1):
                    content = result.get("content", "") if isinstance(result, dict) else getattr(result, "content", "")
                    score = result.get("score", 0.0) if isinstance(result, dict) else getattr(result, "score", 0.0)
                    formatted.append(f"[{i}] {content}\n  Relevance: {score:.4f}")
                return "\n\n".join(formatted)
            except Exception as e:
                return f"RAG search error: {str(e)}"

        tools = [search_web]
        if settings.enable_rag_tool and self.rag_service is not None:
            tools.append(rag_search)
        else:
            logger.info(
                "rag_search tool disabled (ENABLE_RAG_TOOL=%s, rag_service=%s) — "
                "the model goes straight to search_web.",
                settings.enable_rag_tool, self.rag_service is not None,
            )
        return tools

    @staticmethod
    def _build_web_search():
        """Build a callable(query, scope)->(text, sources) backed by Exa, or None
        if unavailable (no API key, or langchain_exa not installed). langchain_exa
        is imported lazily so the module still loads without the package.

        Two retrievers are built once at startup: a campus one pinned to the
        official Udinus domains and a general one with no domain filter. `scope`
        rides along on the same tool call the model already makes, so picking
        between them costs no extra model round-trip.
        """
        if not settings.exa_api_key:
            logger.info("EXA_API_KEY not set — search_web unavailable.")
            return None
        try:
            from langchain_exa import ExaSearchRetriever

            # Exa's text option must be True or a dict (not False). Fast mode =
            # highlights + bounded cached text (fewer tokens); rich mode = full text.
            text_opts = True if settings.exa_fetch_text else {"max_characters": 1000}

            def _make(domains: list[str] | None):
                return ExaSearchRetriever(
                    exa_api_key=settings.exa_api_key,
                    k=settings.web_search_count,
                    type="auto",
                    highlights=True,
                    text_contents_options=text_opts,
                    include_domains=domains or None,
                    livecrawl=settings.exa_livecrawl or None,
                )

            general = _make(settings.exa_include_domains)
            campus = _make(settings.exa_campus_domains or settings.exa_include_domains)
        except Exception as e:
            logger.warning("Exa init failed (web search disabled): %s", e)
            return None
        logger.info(
            "Web search: exa (k=%d, text=%s, campus_domains=%s, general_domains=%s)",
            settings.web_search_count, settings.exa_fetch_text,
            settings.exa_campus_domains or "any",
            settings.exa_include_domains or "any",
        )

        def _run(query: str, scope: str = "general"):
            # Be lenient about what the model puts in `scope` — anything that
            # isn't recognisably "campus" falls through to the general index.
            is_campus = str(scope or "").strip().lower().startswith("campus")
            docs = (campus if is_campus else general).invoke(query)
            logger.info(
                "search_web (scope=%s, query_len=%d) -> %d results",
                "campus" if is_campus else "general", len(query), len(docs or []),
            )
            return (
                LangchainAgentService._format_web_docs(docs),
                LangchainAgentService._extract_sources(docs),
            )

        return _run

    @staticmethod
    def _extract_sources(docs) -> list[dict]:
        """Pull {title, url} out of retriever Documents, de-duplicated by url."""
        sources: list[dict] = []
        seen: set[str] = set()
        for doc in docs or []:
            meta = getattr(doc, "metadata", {}) or {}
            url = meta.get("url") or meta.get("id") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append({"title": meta.get("title") or url, "url": url})
        return sources

    @staticmethod
    def _format_web_docs(docs) -> str:
        """Format Exa retriever Documents into a compact string for the LLM.

        Prefers Exa's embedding-selected `highlights` (clean, relevant snippets)
        over raw `page_content`, which carries page nav/boilerplate. Includes
        title + url so the model can cite the source (see SOURCE ATTRIBUTION).
        """
        if not docs:
            return "No web results found."
        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            meta = getattr(doc, "metadata", {}) or {}
            title = meta.get("title") or ""
            url = meta.get("url") or meta.get("id") or ""
            highlights = meta.get("highlights") or []
            if highlights:
                content = " … ".join(h.strip() for h in highlights if h and h.strip())
            else:
                content = (getattr(doc, "page_content", "") or "").strip()
            header = " — ".join(x for x in (title, url) if x)
            parts.append(f"[{i}] {header}\n{content}".strip())
        return "\n\n".join(parts)

    async def generate_stream(
        self,
        user_input: str,
        messages: list[Message],
        context: UserContext | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        try:
            chat_history = [self._to_lc_message(m) for m in messages]
            chat_history = self._pair_tool_messages([m for m in chat_history if m is not None])

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

            config = {"configurable": {"thread_id": str(uuid.uuid4())}}

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
                    # search_web rides structured sources on .artifact (unseen by
                    # the LLM) — surface them so the client can show references.
                    artifact = getattr(tool_msg, "artifact", None)
                    if artifact:
                        yield SourcesEvent(sources=artifact)

                elif kind == "on_chat_model_end":
                    ai_msg = event["data"]["output"]
                    yield AssistantEvent(message=self._to_domain_ai_message(ai_msg))
        except Exception:
            logger.exception("Agent stream failed")
            yield ErrorEvent(content="Agent error")
            return

    @staticmethod
    def _pair_tool_messages(history: list) -> list:
        """Drop tool-call/tool-result halves that the history window cut in two.

        `chat_history_limit` slices the stored messages at an arbitrary point, so
        a window can open on a tool result whose originating call fell outside
        it, or close on a call whose result did. That was harmless while tool
        calls never reached the model; now that they do, an unmatched half asks
        the model to reason about a call it never made.

        Matching is by id over the whole window rather than by position: message
        rows in one turn share a `created_at` (Postgres now() is the transaction
        timestamp), so the history comes back in arbitrary intra-turn order and a
        result can precede its own call. Position-based pairing would discard
        perfectly good context on that basis.
        """
        answered = {
            m.tool_call_id for m in history if isinstance(m, LCToolMessage)
        }
        called = {
            tc.get("id")
            for m in history
            if isinstance(m, LCAIMessage)
            for tc in (m.tool_calls or [])
        }
        cleaned: list = []
        for msg in history:
            if isinstance(msg, LCAIMessage) and msg.tool_calls:
                kept = [tc for tc in msg.tool_calls if tc.get("id") in answered]
                if len(kept) != len(msg.tool_calls):
                    msg = msg.model_copy(update={"tool_calls": kept})
                # An assistant turn that was nothing but unanswered tool calls
                # carries no information once those are stripped.
                if not kept and not str(msg.content or "").strip():
                    continue
            elif isinstance(msg, LCToolMessage) and msg.tool_call_id not in called:
                continue
            cleaned.append(msg)
        return cleaned

    @staticmethod
    def _tool_call_args(raw) -> dict:
        """Normalize persisted tool-call arguments to the dict LangChain wants.

        Fresh in-memory calls carry a dict; ones round-tripped through the DB
        come back as a JSON string (see mappers._dict_to_ai_message).
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _to_lc_message(message: Message):
        """Convert a domain Message to a LangChain message, or None if unsupported."""
        body = message.body
        if isinstance(body, UserMessage):
            return HumanMessage(content=body.content)
        if isinstance(body, AIMessage):
            # Tool calls MUST go on the `tool_calls` attribute: langchain-ollama
            # reads `message.tool_calls` when converting to the wire format and
            # ignores additional_kwargs entirely. Putting them in
            # additional_kwargs (as this did) silently dropped every tool call
            # from replayed history, leaving an empty assistant turn followed by
            # an orphan tool result — malformed history the model had to guess at.
            tool_calls = [
                {
                    "name": tc.function.name,
                    "args": LangchainAgentService._tool_call_args(tc.function.arguments),
                    "id": tc.id,
                    "type": "tool_call",
                }
                for tc in (body.additional_kwargs or [])
            ]
            return LCAIMessage(content=body.content, tool_calls=tool_calls)
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

    async def preprocess_tts(self, text: str, system_prompt: str) -> str:
        """Preprocess text for TTS."""
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]
            response = await self.llm.ainvoke(messages)

            return response.content.strip()

        except Exception as e:
            logger.error("Error preprocess text for TTS: %s", e)
            return text
