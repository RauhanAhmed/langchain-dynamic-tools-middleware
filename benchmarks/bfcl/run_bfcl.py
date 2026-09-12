"""Run the BFCL harness comparing the three agent configurations.

Real evaluation on real data: the Berkeley Function Calling Leaderboard
dataset, live model calls through the full create_agent stack, and BFCL's
possible-answer matching semantics. Three configurations run on identical
questions and tool sets: no middleware (baseline), LangChain's built-in
LLMToolSelectorMiddleware (extra LLM call per step), and this package's
DynamicToolSelectorMiddleware (local hybrid vector search).

Run from the project root:

    uv run python -m benchmarks.bfcl.run_bfcl \
        --categories simple,irrelevance,relevance --tool-counts 100 --limit 100
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from benchmarks.bfcl.data import function_pool, load_category, tools_from_schemas_dedup
from benchmarks.bfcl.scoring import score_entry
from benchmarks.common import (
    CONFIGS,
    build_agent,
    cost_usd,
    invoke_with_retry,
    make_model,
    new_usage_sink,
    save_results,
)


def entry_tool_schemas(
    entry: dict[str, Any],
    pool: Sequence[dict[str, Any]],
    tool_count: int,
) -> list[dict[str, Any]]:
    """Build the tool schema list for one entry, padded from the pool."""
    schemas = list(entry["functions"])
    if tool_count > len(schemas):
        own_names = {schema["name"] for schema in schemas}
        rng = random.Random(f"{entry['id']}:{tool_count}")
        candidates = [schema for schema in pool if schema["name"] not in own_names]
        rng.shuffle(candidates)
        schemas.extend(candidates[: tool_count - len(schemas)])
    return schemas


def predicted_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the first model response's tool calls as {name, args} dicts."""
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            return [
                {"name": call["name"], "args": dict(call["args"])} for call in message.tool_calls
            ]
    return []


def run_entry(
    entry: dict[str, Any],
    pool: Sequence[dict[str, Any]],
    tool_count: int,
    args: argparse.Namespace,
    index_dir: str,
) -> dict[str, Any]:
    """Run one question end to end and return its per-entry metrics.

    Each entry gets its own model with a usage sink, so the selector's hidden
    internal calls are counted exactly as the provider billed them.
    """
    record: dict[str, Any] = {
        "id": entry["id"],
        "correct": False,
        "error": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "wall_seconds": 0.0,
        "index_seconds": 0.0,
    }
    sink = new_usage_sink()
    model = make_model(args.provider, args.model, usage_sink=sink)
    tools = tools_from_schemas_dedup(entry_tool_schemas(entry, pool, tool_count))
    index_started = time.perf_counter()
    agent, middleware = build_agent(
        args.config,
        tools,
        model,
        top_k=args.top_k,
        index_path=index_dir,
        stop_after_first_model=True,
    )
    record["index_seconds"] = time.perf_counter() - index_started
    try:
        started = time.perf_counter()
        result = invoke_with_retry(
            agent,
            {"messages": [HumanMessage(content=entry["question"])]},
            label=entry["id"],
        )
        record["wall_seconds"] = time.perf_counter() - started
        predicted = predicted_calls(result)
        record["correct"] = score_entry(args.category, predicted, entry["ground_truth"])
        record["prompt_tokens"] = sink["prompt"]
        record["completion_tokens"] = sink["completion"]
        record["model_calls"] = sink["calls"]
        if middleware is not None and middleware.selection_seconds:
            record["selection_seconds_mean"] = sum(middleware.selection_seconds) / len(
                middleware.selection_seconds
            )
    except Exception as exc:  # noqa: BLE001 - record and continue the run
        record["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        if middleware is not None:
            middleware.close()
    return record


def run_config(
    config: str,
    category: str,
    entries: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    tool_count: int,
    args: argparse.Namespace,
    index_base: str,
) -> dict[str, Any]:
    """Run one configuration over all entries and return aggregate metrics.

    Entries run on a small thread pool: each entry gets its own index
    directory, so two workers never contend for a zvec write lock. (Sharing
    one directory per worker thread caused 'Can't lock read-write
    collection' failures whenever a straggler overlapped the next entry on
    the same directory.)
    """
    args.config = config
    args.category = category
    workers = max(1, args.workers)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for position, entry in enumerate(entries):
            index_dir = f"{index_base}-e{position}"
            future = executor.submit(run_entry, entry, pool, tool_count, args, index_dir)
            futures[future] = entry
        for done, future in enumerate(as_completed(futures), start=1):
            entry = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001 - anything outside run_entry
                records.append({"correct": False, "error": str(exc)[:200]})
            print(
                f"  {config} [{done}/{len(entries)}] {entry['id']} done",
                flush=True,
            )
            time.sleep(args.sleep)

    total = len(records)
    correct_count = sum(1 for record in records if record.get("correct"))
    errors = sum(1 for record in records if record.get("error"))
    prompt_tokens = sum(record.get("prompt_tokens", 0) for record in records)
    completion_tokens = sum(record.get("completion_tokens", 0) for record in records)
    wall_seconds = [record["wall_seconds"] for record in records if record.get("wall_seconds")]
    selection_seconds = [
        value
        for record in records
        for value in [record.get("selection_seconds_mean", 0.0)]
        if value
    ]
    index_seconds = [record["index_seconds"] for record in records if record.get("index_seconds")]
    error_samples = sorted({str(record.get("error")) for record in records if record.get("error")})[
        :5
    ]
    return {
        "config": config,
        "category": category,
        "tool_count": tool_count,
        "entries": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "errors": errors,
        "error_samples": error_samples,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "prompt_tokens_mean": prompt_tokens / total if total else 0.0,
        "completion_tokens_mean": completion_tokens / total if total else 0.0,
        "wall_seconds_mean": sum(wall_seconds) / len(wall_seconds) if wall_seconds else 0.0,
        "selection_seconds_mean": (
            sum(selection_seconds) / len(selection_seconds) if selection_seconds else 0.0
        ),
        "index_seconds_mean": sum(index_seconds) / len(index_seconds) if index_seconds else 0.0,
        "cost_usd": cost_usd(args.model, prompt_tokens, completion_tokens),
    }


def selftest() -> None:
    """Validate data loading, tool conversion, and scoring without any LLM.

    Checks canned scoring cases that mirror the BFCL answer format, then two
    real entries: one with a hand-built correct prediction, one padded and
    converted through the real tool pipeline.
    """
    from benchmarks.bfcl.data import function_pool, load_category, sanitize_name

    # Canned cases in the exact possible-answer format.
    assert score_entry(
        "simple",
        [{"name": "math_factorial", "args": {"number": 5}}],
        [{"math.factorial": {"number": [5]}}],
    )
    assert score_entry(
        "simple",
        [{"name": "calculate_triangle_area", "args": {"base": 10, "height": 5}}],
        [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}],
    )
    assert not score_entry(
        "simple",
        [{"name": "calculate_triangle_area", "args": {"base": 10, "height": 5, "unit": "m"}}],
        [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}],
    )
    assert not score_entry(
        "simple",
        [{"name": "calculate_triangle_area", "args": {"base": 10}}],
        [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}],
    )
    assert score_entry("irrelevance", [], None)
    assert not score_entry("irrelevance", [{"name": "anything", "args": {}}], None)
    assert score_entry("relevance", [{"name": "anything", "args": {}}], None)
    assert not score_entry("relevance", [], None)
    print("Scoring semantics: OK")

    entries = load_category("simple", limit=2)
    pool = function_pool()
    tools = tools_from_schemas_dedup(entry_tool_schemas(entries[0], pool, 50))
    names = {tool.name for tool in tools}
    assert len(names) == 50, f"expected 50 tools, got {len(names)}"
    assert len(tools) == 50, "sanitized-name collisions were not deduplicated"
    assert sanitize_name(entries[0]["functions"][0]["name"]) in names
    own_tools = tools_from_schemas_dedup(entries[1]["functions"])
    expected_name = sanitize_name(entries[1]["functions"][0]["name"])
    assert expected_name in {tool.name for tool in own_tools}
    # A correct call built from the ground truth must score.
    truth = load_category("simple", limit=3)[1]
    assert truth["ground_truth"], "ground truth missing for simple_1"
    call_spec = truth["ground_truth"][0]
    truth_name, truth_params = next(iter(call_spec.items()))
    predicted_args = {param: options[0] for param, options in truth_params.items()}
    predicted_args = {
        param: value for param, value in predicted_args.items() if value != ""
    }
    assert score_entry(
        "simple",
        [{"name": sanitize_name(truth_name), "args": predicted_args}],
        truth["ground_truth"],
    )
    print(f"Data pipeline: OK ({len(names)} tools converted for {entries[0]['id']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="BFCL harness for tool selection")
    parser.add_argument(
        "--categories",
        default="simple,irrelevance,relevance",
        help="Comma separated BFCL categories to evaluate",
    )
    parser.add_argument(
        "--tool-counts",
        default="100",
        help="Comma separated tool set sizes to pad each question to",
    )
    parser.add_argument("--limit", type=int, default=100, help="Entries per category")
    parser.add_argument("--top-k", type=int, default=4, help="Tools per step for selectors")
    parser.add_argument("--provider", default="nvidia", choices=["nvidia", "openai", "openrouter"])
    parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between questions")
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Parallel workers; each entry gets its own index directory",
    )
    parser.add_argument("--out", default="bfcl_results", help="Output file name")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Validate loading and scoring without any LLM calls, then exit",
    )
    parser.add_argument(
        "--configs",
        default="baseline,llm_selector,dynamic",
        help="Comma separated subset of configs to run (e.g. --configs dynamic "
        "to redo one block without repeating the others)",
    )
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    categories = [name.strip() for name in args.categories.split(",") if name.strip()]
    tool_counts = [int(value) for value in args.tool_counts.split(",") if value.strip()]
    configs = [name.strip() for name in args.configs.split(",") if name.strip()]
    for name in configs:
        if name not in CONFIGS:
            msg = f"Unknown config {name!r}; expected one of {CONFIGS}"
            raise ValueError(msg)

    print("Loading BFCL data and building the function pool...", flush=True)
    pool = function_pool()
    print(f"Function pool: {len(pool)} unique functions", flush=True)

    results: list[dict[str, Any]] = []
    for category in categories:
        entries = load_category(category, limit=args.limit)
        print(f"\nCategory {category}: {len(entries)} entries", flush=True)
        for tool_count in tool_counts:
            for config in configs:
                print(
                    f"\nRunning config={config} tools={tool_count} category={category}",
                    flush=True,
                )
                with tempfile.TemporaryDirectory(prefix="bfcl-index-") as index_dir:
                    summary = run_config(
                        config, category, entries, pool, tool_count, args, index_dir
                    )
                results.append(summary)
                # Save after every block so long background runs keep partial progress.
                save_results(
                    args.out,
                    {
                        "model": f"{args.provider}:{args.model}",
                        "top_k": args.top_k,
                        "limit_per_category": args.limit,
                        "results": results,
                    },
                )
                print(
                    f"  accuracy={summary['accuracy']:.1%} "
                    f"prompt_tokens_mean={summary['prompt_tokens_mean']:.0f} "
                    f"wall_mean={summary['wall_seconds_mean']:.2f}s",
                    flush=True,
                )

    payload = {
        "model": f"{args.provider}:{args.model}",
        "top_k": args.top_k,
        "limit_per_category": args.limit,
        "results": results,
    }
    path = save_results(args.out, payload)
    print(f"\nSaved results to {path}")

    print("\nSummary (accuracy / mean prompt tokens per question):")
    for summary in results:
        print(
            f"  {summary['category']:>12} tools={summary['tool_count']:>4} "
            f"{summary['config']:>12}  acc={summary['accuracy']:.1%}  "
            f"tokens={summary['prompt_tokens_mean']:.0f}  "
            f"wall={summary['wall_seconds_mean']:.2f}s  "
            f"select={summary['selection_seconds_mean'] * 1000:.1f}ms"
        )


if __name__ == "__main__":
    main()
