"""LangChain agent middleware that selects tools with local hybrid vector search.

Agents with many tools burn tokens on tool schemas the model will never use
and lose accuracy picking between lookalike options. This middleware indexes
every tool once, then, before each model call, searches the index with the
user's latest message and hands the model only the top_k most relevant tools.
The full tool set stays registered with the agent, so any selected tool can
still execute. Selection is a local vector search, not another LLM call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from langchain_dynamic_tools._embeddings import DenseEmbedder, SparseEmbedder
from langchain_dynamic_tools._index import DEFAULT_PATH, ToolVectorIndex

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

OnError = Literal["fallback_all", "raise"]
"""What to do when the vector search itself fails.

'fallback_all' keeps every tool for that step (the default, agents never
crash over a retrieval hiccup). 'raise' propagates the error.
"""


def _message_text(message: HumanMessage) -> str:
    """Extract plain text from a message with string or content-block content."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(part for part in parts if part)
    return str(content)


def _user_texts(messages: Sequence[BaseMessage]) -> list[str]:
    """Return the non-empty texts of all user messages, in order."""
    texts = []
    for message in messages:
        if isinstance(message, HumanMessage):
            text = _message_text(message).strip()
            if text:
                texts.append(text)
    return texts


#: Queries shorter than this borrow context from the previous user message.
#: Follow-ups like "do that again for Berlin" carry no retrievable content
#: on their own, so the previous turn is prepended for the search.
SHORT_QUERY_CHARS = 40

#: Upper bound on the search query; keeps embedding cost flat.
MAX_QUERY_CHARS = 2000


def _selection_query(messages: Sequence[BaseMessage]) -> str | None:
    """Build the vector search query for this step, or None if absent.

    Normally the latest user message. When it is very short, the previous
    user message is prepended so follow-ups still match the right tools.
    """
    texts = _user_texts(messages)
    if not texts:
        return None
    if len(texts[-1]) >= SHORT_QUERY_CHARS or len(texts) == 1:
        return texts[-1][:MAX_QUERY_CHARS]
    return "\n".join([texts[-2], texts[-1]])[:MAX_QUERY_CHARS]


class DynamicToolSelectorMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Selects the most relevant tools per step with local hybrid vector search.

    Point it at your tool set and drop it into an agent. Before each model
    call it embeds the search query (the user's latest message, plus the
    previous one when the latest is a very short follow-up), searches a local zvec
    collection holding dense and sparse embeddings of every tool, fuses the
    two rankings, and overrides the request so the model sees only the
    top_k winners. All tools remain registered for execution, so nothing is
    lost, the model just gets a shorter, sharper tool list.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI
        from langchain_dynamic_tools import DynamicToolSelectorMiddleware

        all_tools = [get_weather, query_sql, send_email, git_push]

        tool_router = DynamicToolSelectorMiddleware(
            tools=all_tools,
            top_k=4,
        )

        agent = create_agent(
            model=ChatOpenAI(model="gpt-4o"),
            tools=all_tools,  # full set available for execution
            middleware=[tool_router],
        )
        ```
    """

    def __init__(
        self,
        tools: Sequence[BaseTool | dict[str, Any]],
        top_k: int = 4,
        *,
        path: str = DEFAULT_PATH,
        dense_embedder: DenseEmbedder | Embeddings | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        dense_dim: int | None = None,
        always_include: list[str] | None = None,
        reranker: Any | None = None,
        on_error: OnError = "fallback_all",
    ) -> None:
        """Index the tools and prepare per-step selection.

        Args:
            tools: The agent's tools. BaseTool instances are indexed; provider
                tool dicts are kept for pass-through at call time.
            top_k: Maximum tools handed to the model per step.
            path: Filesystem path of the local zvec collection.
            dense_embedder: Optional dense embedder override. A standard
                LangChain Embeddings object (e.g. OpenAIEmbeddings) is
                accepted directly and adapted automatically.
            sparse_embedder: Optional sparse embedder override.
            dense_dim: Dense vector dimension override. Defaults to the
                embedder's dimension (probed lazily for LangChain
                embeddings, so pass this to skip the probe call).
            always_include: Tool names added to every selection on top of
                top_k. Useful for tools the model may need without asking,
                like a todo list or handoff tool.
            reranker: Optional zvec reranker override for fusing the dense
                and sparse rankings.
            on_error: 'fallback_all' keeps all tools when the search fails;
                'raise' propagates the error instead.

        Raises:
            ValueError: If top_k is less than 1, no BaseTool is given,
                or always_include names a tool that is not in tools.
        """
        super().__init__()
        if top_k < 1:
            msg = f"top_k must be >= 1, got {top_k}"
            raise ValueError(msg)
        indexable = [tool for tool in tools if isinstance(tool, BaseTool)]
        if not indexable:
            msg = "tools must contain at least one BaseTool instance to index"
            raise ValueError(msg)
        self._top_k = top_k
        self._always_include = list(always_include or [])
        self._on_error = on_error

        known_names = {tool.name for tool in indexable}
        missing = [name for name in self._always_include if name not in known_names]
        if missing:
            msg = (
                f"always_include contains unknown tool names: {missing}. "
                f"Known tools: {sorted(known_names)}"
            )
            raise ValueError(msg)

        self._tools = indexable
        self._index = ToolVectorIndex(
            indexable,
            path=path,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            dense_dim=dense_dim,
            reranker=reranker,
        )
        logger.info(
            "DynamicToolSelectorMiddleware indexed %d tool(s) at %r (top_k=%d)",
            len(indexable),
            path,
            top_k,
        )

    @property
    def top_k(self) -> int:
        """Maximum number of tools handed to the model per step."""
        return self._top_k

    def sync(self) -> None:
        """Re-index the tools this middleware was constructed with.

        Call this after editing a tool's description or arguments schema so
        the index picks up the new text. Unchanged tools are not re-embedded.
        """
        self._index.sync(self._tools, prune=True)

    def close(self) -> None:
        """Close the underlying zvec collection and release its file lock."""
        self._index.close()

    def _sync_missing(self, base_tools: list[BaseTool]) -> None:
        """Index any request tools that are not in the index yet."""
        indexed = self._index.indexed_names
        missing = [tool for tool in base_tools if tool.name not in indexed]
        if missing:
            logger.debug("Indexing %d new tool(s) found in the request", len(missing))
            self._index.sync(missing, prune=False)

    def _select_tools(
        self, request: ModelRequest[ContextT]
    ) -> list[BaseTool | dict[str, Any]] | None:
        """Pick the tools for this step, or return None to keep the request as is.

        None means pass-through: nothing to select from, no user message to
        search with, no match found, or a failed search under the
        fallback_all policy.
        """
        if not request.tools:
            return None
        base_tools = [tool for tool in request.tools if isinstance(tool, BaseTool)]
        provider_tools = [tool for tool in request.tools if isinstance(tool, dict)]
        if not base_tools:
            return None
        query_text = _selection_query(request.messages)
        if query_text is None:
            return None

        try:
            self._sync_missing(base_tools)
            ranked = self._index.search(query_text, self._top_k)
        except Exception:
            if self._on_error == "raise":
                raise
            logger.warning(
                "Dynamic tool selection failed; falling back to all tools",
                exc_info=True,
            )
            return None

        by_name = {tool.name: tool for tool in base_tools}
        selected_names: list[str] = []
        for name, _score in ranked:
            if name in by_name and name not in selected_names:
                selected_names.append(name)
        for name in self._always_include:
            if name in by_name and name not in selected_names:
                selected_names.append(name)

        if not selected_names:
            logger.debug("No indexed tool matched the query; keeping all tools")
            return None
        logger.debug("Selected tools for this step: %s", selected_names)
        selected = [by_name[name] for name in selected_names]
        return [*selected, *provider_tools]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Filter the request's tools to the most relevant ones, then call the model."""
        selection = self._select_tools(request)
        if selection is None:
            return handler(request)
        return handler(request.override(tools=selection))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Async variant; the blocking search runs in a worker thread."""
        selection = await asyncio.to_thread(self._select_tools, request)
        if selection is None:
            return await handler(request)
        return await handler(request.override(tools=selection))
