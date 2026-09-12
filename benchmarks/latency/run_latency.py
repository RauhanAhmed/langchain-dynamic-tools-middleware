"""Measure selection latency and index build time against tool count.

Real measurements end to end: the real zvec engine, the real local embedding
models (all-MiniLM-L6-v2 dense, SPLADE sparse), and real user questions from
the BFCL dataset. For comparison, the harness also times the dominant cost of
LLM-based tool selection: one extra model call per step whose prompt carries
every tool schema.

Run from the project root:

    uv run python -m benchmarks.latency.run_latency
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from benchmarks.bfcl.data import function_pool, load_category, tools_from_schemas_dedup
from benchmarks.common import RESULTS_DIR, make_model, save_results, stats
from langchain_dynamic_tools import DefaultDenseEmbedder, DefaultSparseEmbedder, ToolVectorIndex

SELECTION_SYSTEM_PROMPT = (
    "Your goal is to select the most relevant tools for answering the user's query."
)


def sample_questions(count: int) -> list[str]:
    """Realistic user questions drawn from the BFCL simple category."""
    entries = load_category("simple", limit=count)
    questions = [entry["question"] for entry in entries if entry["question"]]
    return questions[:count]


def measure_dynamic(
    tool_counts: Sequence[int],
    questions: Sequence[str],
    top_k: int,
) -> dict[str, Any]:
    """Index each tool count and measure real per-query selection latency."""
    dense = DefaultDenseEmbedder()
    sparse = DefaultSparseEmbedder()
    rows: list[dict[str, Any]] = []

    for tool_count in tool_counts:
        schemas = function_pool()[:tool_count]
        tools = tools_from_schemas_dedup(schemas)
        with tempfile.TemporaryDirectory(prefix="latency-index-") as index_dir:
            build_started = time.perf_counter()
            index = ToolVectorIndex(
                tools,
                path=index_dir,
                dense_embedder=dense,
                sparse_embedder=sparse,
            )
            build_seconds = time.perf_counter() - build_started
            try:
                per_query: list[float] = []
                for question in questions:
                    started = time.perf_counter()
                    index.search(question, top_k)
                    per_query.append(time.perf_counter() - started)
            finally:
                index.close()
        row = {
            "tool_count": len(tools),
            "index_build_seconds": build_seconds,
            "per_tool_ms": build_seconds / len(tools) * 1000,
            "query_ms": {key: value * 1000 for key, value in stats(per_query).items()},
        }
        rows.append(row)
        print(
            f"  {len(tools):>5} tools: build={build_seconds:6.2f}s "
            f"query mean={row['query_ms']['mean']:6.1f}ms "
            f"p95={row['query_ms']['p95']:6.1f}ms",
            flush=True,
        )
    return {"rows": rows}


def measure_llm_selector(
    tools: Sequence[Any],
    questions: Sequence[str],
    model: Any,
    runs_per_question: int,
) -> dict[str, Any]:
    """Time the extra model call an LLM tool selector makes per step.

    LLMToolSelectorMiddleware asks a model to pick tools, so its selection
    cost is one additional model round trip whose prompt carries every tool
    schema. We measure exactly that call.
    """
    bound = model.bind_tools(list(tools))
    per_query: list[float] = []
    for question in questions:
        for _ in range(runs_per_question):
            messages = [
                SystemMessage(content=SELECTION_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
            started = time.perf_counter()
            bound.invoke(messages)
            per_query.append(time.perf_counter() - started)
    return {"selection_ms": {key: value * 1000 for key, value in stats(per_query).items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Selection latency benchmark")
    parser.add_argument(
        "--tool-counts", default="10,50,100,250,500,1000", help="Comma separated sizes"
    )
    parser.add_argument("--queries", type=int, default=20, help="Questions per size")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--llm-selector-runs",
        type=int,
        default=3,
        help="Runs per question when timing the LLM selector call (0 to skip)",
    )
    parser.add_argument("--provider", default="nvidia", choices=["nvidia", "openai", "openrouter"])
    parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    parser.add_argument("--out", default="latency_results")
    args = parser.parse_args()

    tool_counts = [int(value) for value in args.tool_counts.split(",") if value.strip()]
    questions = sample_questions(args.queries)
    print(f"Measuring with {len(questions)} real BFCL questions", flush=True)

    print("\nDynamic tool selector (local hybrid vector search):", flush=True)
    dynamic = measure_dynamic(tool_counts, questions, args.top_k)

    llm_selector: dict[str, Any] | None = None
    if args.llm_selector_runs > 0:
        print("\nLLM tool selector (one extra model call per step):", flush=True)
        model = make_model(args.provider, args.model)
        pool = function_pool()
        tools = tools_from_schemas_dedup(pool[:100])
        llm_selector = measure_llm_selector(tools, questions[:5], model, args.llm_selector_runs)
        print(
            f"  100 tools: selection mean={llm_selector['selection_ms']['mean']:.0f}ms "
            f"p95={llm_selector['selection_ms']['p95']:.0f}ms",
            flush=True,
        )

    payload = {
        "dynamic": dynamic,
        "llm_selector": llm_selector,
        "top_k": args.top_k,
        "questions": len(questions),
    }
    path = save_results(args.out, payload)
    print(f"\nSaved results to {path}")

    if llm_selector is not None:
        dynamic_100 = next((row for row in dynamic["rows"] if row["tool_count"] == 100), None)
        if dynamic_100 is not None:
            speedup = llm_selector["selection_ms"]["mean"] / dynamic_100["query_ms"]["mean"]
            print(
                f"\nAt 100 tools: local search {dynamic_100['query_ms']['mean']:.1f}ms vs "
                f"LLM selector call {llm_selector['selection_ms']['mean']:.0f}ms "
                f"({speedup:.0f}x faster)"
            )

    chart_path = RESULTS_DIR / "latency.png"
    try:
        import matplotlib.pyplot as plt

        counts = [row["tool_count"] for row in dynamic["rows"]]
        means = [row["query_ms"]["mean"] for row in dynamic["rows"]]
        p95s = [row["query_ms"]["p95"] for row in dynamic["rows"]]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(counts, means, marker="o", label="mean")
        axis.plot(counts, p95s, marker="^", label="p95")
        if llm_selector is not None:
            axis.axhline(
                llm_selector["selection_ms"]["mean"],
                color="red",
                linestyle="--",
                label="LLM selector call (100 tools)",
            )
        axis.set_xlabel("Tools in the index")
        axis.set_ylabel("Selection latency (ms)")
        axis.set_title("Dynamic tool selection latency vs tool count")
        axis.legend()
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(chart_path, dpi=150)
        print(f"Chart saved to {chart_path}")
    except ImportError:
        print("matplotlib not installed; skipping the chart")


if __name__ == "__main__":
    main()
