"""Shared harness for the benchmark suite.

Provides the three agent configurations under test, a timing-instrumented
version of the middleware, token accounting from provider usage metadata,
cost estimation, and JSON result storage. Benchmarks are scripts, not part
of the installed package: run them with `uv run python benchmarks/...`.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from langchain_dynamic_tools import DynamicToolSelectorMiddleware

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool

load_dotenv()

warnings.filterwarnings("ignore", message=".*is not known to support tools.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"langchain_nvidia_ai_endpoints.*")

RESULTS_DIR = Path(__file__).parent / "results"

CONFIGS = ("baseline", "llm_selector", "dynamic")
"""The three configurations every benchmark compares."""

MAX_COMPLETION_TOKENS = 4096
"""Completion budget per model call. nemotron reasons before tool calls, and
the integration's 1024 default truncates those responses."""

_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = [0.0]
_MIN_REQUEST_INTERVAL = 1.7
"""Minimum seconds between request starts: NVIDIA allows 40 requests per
minute, and parallel workers would otherwise burst past it."""


def _respect_rate_limit() -> None:
    """Hold every caller until at least the minimum interval has passed.

    Shared across threads so the whole harness, benchmark wide, stays under
    the provider's requests-per-minute ceiling.
    """
    with _REQUEST_LOCK:
        wait = _LAST_REQUEST_AT[0] + _MIN_REQUEST_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT[0] = time.monotonic()

# OpenAI list prices in USD per million tokens. Update together with the
# tables in RESULTS.md when the benchmark model changes. Models not listed
# here (OpenRouter free tiers, for example) are treated as zero cost, and
# token counts are still reported.
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


class TimedDynamicToolSelectorMiddleware(DynamicToolSelectorMiddleware):
    """DynamicToolSelectorMiddleware that records per-step selection latency."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.selection_seconds: list[float] = []

    def _select_tools(self, request: Any) -> Any:
        start = time.perf_counter()
        result = super()._select_tools(request)
        self.selection_seconds.append(time.perf_counter() - start)
        return result


class StopAfterFirstModelMiddleware(AgentMiddleware):
    """Ends the agent right after the first model response.

    BFCL scores the function call a model makes, not what happens after
    execution, so the harness stops the loop there. Applied identically to
    every configuration under test.
    """

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any]:
        return {"jump_to": "end"}


def _make_nvidia_model(
    model_name: str, timeout_seconds: float = 240.0, usage_sink: dict[str, int] | None = None
) -> Any:
    """Build a ChatNVIDIA wired for the benchmark harness.

    Two adaptations, both documented in benchmarks/README.md:

    - Structured output goes through native tool calling: NVIDIA's hosted
      endpoint rejects the guided_json field that the integration's default
      structured output path emits (400: unknown field). The harness needs it
      for LangChain's LLMToolSelectorMiddleware baseline; the dynamic
      configuration never calls an LLM and is unaffected.
    - Every model call reports its provider-billed tokens into usage_sink.
      LangChain hides the selector's internal calls from agent state, so
      state-based token counts understate the LLM selector's true cost;
      source-level counting captures all calls exactly once.

    The timeout is forwarded to the client transport: big selection prompts
    regularly exceed the integration's 60 second default. max_tokens is
    raised above the integration's 1024 default because nemotron is a
    reasoning model and spends completion tokens thinking before it emits a
    tool call; capped responses come back truncated and agents lose tool
    calls.
    """
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    def _collect(result: Any) -> None:
        if usage_sink is None:
            return
        for generation in result.generations:
            usage = getattr(generation.message, "usage_metadata", None)
            if usage:
                usage_sink["prompt"] += usage.get("input_tokens", 0)
                usage_sink["completion"] += usage.get("output_tokens", 0)
                usage_sink["calls"] += 1

    class NvidiaBenchmarkModel(ChatNVIDIA):
        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> Any:
            _respect_rate_limit()
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            _collect(result)
            return result

        async def _agenerate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> Any:
            _respect_rate_limit()
            result = await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            _collect(result)
            return result

        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
            from langchain_core.runnables import RunnableLambda

            def _parse(message: AIMessage) -> Any:
                if getattr(message, "tool_calls", None):
                    return message.tool_calls[0]["args"]
                return {}

            tool = {
                "type": "function",
                "function": {
                    "name": "structured_output",
                    "description": "Return the structured selection.",
                    "parameters": schema,
                },
            }
            return self.bind_tools([tool]) | RunnableLambda(_parse)

    return NvidiaBenchmarkModel(
        model=model_name, timeout=timeout_seconds, max_tokens=MAX_COMPLETION_TOKENS
    )


def new_usage_sink() -> dict[str, int]:
    """Create an empty usage sink for one benchmark run."""
    return {"prompt": 0, "completion": 0, "calls": 0}


def make_model(
    provider: str,
    model_name: str,
    timeout_seconds: float = 240.0,
    usage_sink: dict[str, int] | None = None,
) -> Any:
    """Build a chat model for the given provider ('nvidia', 'openai', or 'openrouter').

    Calls are bounded by a timeout where the client supports one, so a hung
    free-tier request fails into the harness retry logic instead of stalling
    the run. NVIDIA's hosted endpoint is the primary benchmark provider; only
    that provider reports into usage_sink, and callers that pass no sink get
    plain uninstrumented models.
    """
    if provider == "nvidia":
        return _make_nvidia_model(model_name, timeout_seconds, usage_sink)
    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(model=model_name, timeout=timeout_seconds)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, timeout=timeout_seconds)
    msg = f"Unknown provider {provider!r}; use 'nvidia', 'openai', or 'openrouter'"
    raise ValueError(msg)


def build_agent(
    config: str,
    tools: Sequence[BaseTool],
    model: Any,
    *,
    top_k: int = 4,
    index_path: str,
    dense_embedder: Any = None,
    sparse_embedder: Any = None,
    stop_after_first_model: bool = False,
) -> tuple[Any, TimedDynamicToolSelectorMiddleware | None]:
    """Build an agent for one of the three configurations under test.

    Returns the compiled agent and the instrumented middleware when the
    configuration is 'dynamic', otherwise None.
    """
    middleware: list[Any] = []
    if stop_after_first_model:
        middleware.append(StopAfterFirstModelMiddleware())

    if config == "baseline":
        return create_agent(model=model, tools=list(tools), middleware=middleware), None
    if config == "llm_selector":
        from langchain.agents.middleware import LLMToolSelectorMiddleware

        middleware.append(LLMToolSelectorMiddleware(model=model, max_tools=top_k))
        return create_agent(model=model, tools=list(tools), middleware=middleware), None
    if config == "dynamic":
        selector = TimedDynamicToolSelectorMiddleware(
            tools=list(tools),
            top_k=top_k,
            path=index_path,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
        middleware.append(selector)
        return create_agent(model=model, tools=list(tools), middleware=middleware), selector
    msg = f"Unknown config {config!r}; expected one of {CONFIGS}"
    raise ValueError(msg)


def cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost from provider list prices; unlisted models cost nothing."""
    prices = PRICES_PER_MTOK.get(model_name)
    if prices is None:
        return 0.0
    return (
        prompt_tokens / 1_000_000 * prices["input"]
        + completion_tokens / 1_000_000 * prices["output"]
    )


def stats(values: Sequence[float]) -> dict[str, float]:
    """Mean, median, p95, min, and max of a list of numbers."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    count = len(ordered)
    p95_index = min(count - 1, math.ceil(0.95 * count) - 1)
    return {
        "mean": sum(ordered) / count,
        "p50": ordered[count // 2],
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def save_results(name: str, payload: Any) -> Path:
    """Persist a benchmark result payload as JSON under benchmarks/results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def invoke_with_retry(
    agent: Any,
    payload: dict[str, Any],
    *,
    attempts: int = 3,
    base_delay: float = 4.0,
    label: str = "",
) -> Any:
    """Invoke an agent, retrying with backoff on transient provider errors.

    Rate limit responses (429) get long sleeps because provider limit windows
    run into minutes; other errors get short exponential backoff. Every retry
    is logged so long background runs stay observable.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return agent.invoke(payload)
        except Exception as exc:  # noqa: BLE001 - provider errors vary widely
            last_error = exc
            if attempt == attempts - 1:
                break
            text = str(exc)
            if "429" in text or "Too Many Requests" in text.lower():
                delay = 45.0 * (attempt + 1)
            else:
                delay = base_delay * (2**attempt)
            print(
                f"  retry {attempt + 1}/{attempts - 1} for {label or 'request'} "
                f"after {type(exc).__name__}: {text[:120]}; sleeping {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise last_error  # type: ignore[misc]
