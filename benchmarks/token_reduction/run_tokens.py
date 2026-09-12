"""Measure token usage and cost across a multi-step conversation.

A fixed ten turn conversation runs through the full create_agent stack with
real tool execution, once per configuration, against a 100 tool set built
from the BFCL function pool. Token counts come from provider usage metadata,
so the numbers are what the provider actually billed.

Run from the project root:

    uv run python -m benchmarks.token_reduction.run_tokens
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage

from benchmarks.bfcl.data import function_pool, load_category, tools_from_schemas_dedup
from benchmarks.common import (
    CONFIGS,
    build_agent,
    cost_usd,
    invoke_with_retry,
    make_model,
    new_usage_sink,
    save_results,
)


def build_conversation(
    turns: int, pool: Sequence[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Pick real BFCL questions and the schemas needed to answer them."""
    entries = [entry for entry in load_category("simple") if entry["question"]]
    chosen = entries[:turns]
    questions = [entry["question"] for entry in chosen]
    needed: dict[str, dict[str, Any]] = {}
    for entry in chosen:
        for schema in entry["functions"]:
            needed.setdefault(schema["name"], schema)
    for schema in pool:
        needed.setdefault(schema["name"], schema)
    return questions, list(needed.values())


def run_conversation(
    config: str,
    questions: Sequence[str],
    schemas: Sequence[dict[str, Any]],
    tool_count: int,
    args: argparse.Namespace,
    index_dir: str,
) -> dict[str, Any]:
    """Run the whole conversation once and account for every token.

    Tokens are counted at the model source through a usage sink, so hidden
    internal calls like the LLM selector's per-step selection request are
    included in the totals exactly as the provider billed them.
    """
    sink = new_usage_sink()
    model = make_model(args.provider, args.model, usage_sink=sink)
    tools = tools_from_schemas_dedup(schemas[:tool_count])
    agent, middleware = build_agent(config, tools, model, top_k=args.top_k, index_path=index_dir)
    history: list[Any] = []
    per_turn: list[dict[str, Any]] = []
    failed_turns = 0
    try:
        for turn, question in enumerate(questions):
            before = dict(sink)
            try:
                result = invoke_with_retry(
                    agent,
                    {"messages": [*history, HumanMessage(content=question)]},
                    label=f"{config} turn {turn + 1}",
                )
            except Exception as exc:  # noqa: BLE001 - a failed turn must not kill the run
                failed_turns += 1
                print(
                    f"  {config} turn {turn + 1}/{len(questions)} FAILED: "
                    f"{type(exc).__name__}: {str(exc)[:120]}",
                    flush=True,
                )
                continue
            history = result["messages"]
            turn_prompt = sink["prompt"] - before["prompt"]
            turn_completion = sink["completion"] - before["completion"]
            per_turn.append(
                {
                    "turn": turn,
                    "question": question,
                    "prompt_tokens": turn_prompt,
                    "completion_tokens": turn_completion,
                    "model_calls": sink["calls"] - before["calls"],
                }
            )
            print(
                f"  {config} turn {turn + 1}/{len(questions)}: "
                f"prompt={turn_prompt} completion={turn_completion}",
                flush=True,
            )
            time.sleep(args.sleep)
    finally:
        if middleware is not None:
            middleware.close()

    if not per_turn:
        msg = f"config {config}: every turn failed"
        raise RuntimeError(msg)
    prompt_total = sink["prompt"]
    completion_total = sink["completion"]
    return {
        "config": config,
        "tool_count": len(tools),
        "turns": len(per_turn),
        "failed_turns": failed_turns,
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "prompt_tokens_per_turn": prompt_total / len(per_turn) if per_turn else 0.0,
        "selection_seconds_mean": (
            sum(middleware.selection_seconds) / len(middleware.selection_seconds)
            if middleware is not None and middleware.selection_seconds
            else 0.0
        ),
        "per_turn": per_turn,
        "cost_usd": cost_usd(args.model, prompt_total, completion_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Token reduction benchmark")
    parser.add_argument("--tool-count", type=int, default=100)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--provider", default="nvidia", choices=["nvidia", "openai", "openrouter"])
    parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--out", default="token_results")
    args = parser.parse_args()

    pool = function_pool()
    questions, schemas = build_conversation(args.turns, pool)
    print(f"Conversation: {len(questions)} turns, {len(schemas)} schemas available", flush=True)

    results: list[dict[str, Any]] = []
    for config in CONFIGS:
        print(f"\nRunning config={config}", flush=True)
        with tempfile.TemporaryDirectory(prefix="tokens-index-") as index_dir:
            summary = run_conversation(
                config, questions, schemas, args.tool_count, args, index_dir
            )
        results.append(summary)
        # Checkpoint after every config so a late failure never discards
        # completed runs (e.g. baseline + llm_selector survive a dynamic crash).
        save_results(
            args.out,
            {
                "model": f"{args.provider}:{args.model}",
                "top_k": args.top_k,
                "results": results,
            },
        )
        print(
            f"  prompt_tokens={summary['prompt_tokens']} "
            f"completion_tokens={summary['completion_tokens']} "
            f"cost=${summary['cost_usd']:.4f}",
            flush=True,
        )

    baseline = next((row for row in results if row["config"] == "baseline"), None)
    if baseline is not None:
        for row in results:
            if baseline["prompt_tokens"]:
                row["prompt_reduction_vs_baseline"] = (
                    1 - row["prompt_tokens"] / baseline["prompt_tokens"]
                )
            if baseline["cost_usd"]:
                row["cost_reduction_vs_baseline"] = 1 - row["cost_usd"] / baseline["cost_usd"]

    payload = {
        "model": f"{args.provider}:{args.model}",
        "top_k": args.top_k,
        "results": results,
    }
    path = save_results(args.out, payload)
    print(f"\nSaved results to {path}")

    print("\nSummary over the whole conversation:")
    for row in results:
        reduction = row.get("prompt_reduction_vs_baseline")
        reduction_text = f"{reduction:.1%} fewer prompt tokens" if reduction is not None else ""
        print(
            f"  {row['config']:>12}: prompt={row['prompt_tokens']:>7} "
            f"completion={row['completion_tokens']:>6} cost=${row['cost_usd']:.4f} "
            f"{reduction_text}"
        )


if __name__ == "__main__":
    main()
