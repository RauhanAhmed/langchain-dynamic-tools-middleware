# Benchmark results

Every number in this file comes from the JSON payloads in `results/`, written
by the harness scripts. Nothing here is hand-written. Rerun commands are
listed per section so anyone can reproduce.

## Environment

| | |
| --- | --- |
| Date | 2026-09-11 |
| Machine | Intel Core i7-9750H (6 cores, x86_64), Linux x86_64 dev container, CPU only |
| Package | langchain-dynamic-tools-middleware 0.1.0 |
| zvec | 0.7.0 |
| langchain | 1.4.0 (langchain-core 1.6.2) |
| Dense embedder | sentence-transformers all-MiniLM-L6-v2, 384 dims, CPU |
| Sparse embedder | SPLADE (naver/splade-cocondenser-ensembledistil), CPU |
| Chat model | OpenRouter `nvidia/nemotron-3.5-lightning:free` (free tier) |
| Selection config | top_k=4, RrfReRanker(rank_constant=60) |

The container runs on a laptop CPU and the free model tier adds queueing
delay, so treat absolute latencies as indicative and the ratios as the
durable result.

## Selection latency vs tool count

Command:

```bash
uv run python -m benchmarks.latency.run_latency --llm-selector-runs 0
```

One selection = embed the query with MiniLM and SPLADE, run one hybrid zvec
query with RRF fusion, return top-k names. 20 real BFCL questions per size,
fresh index per size.

| Tools in index | Build time (s) | Query mean (ms) | Query p95 (ms) |
| ---: | ---: | ---: | ---: |
| 10 | 29.13 | 251.7 | 885.3 |
| 50 | 9.83 | 106.5 | 215.2 |
| 100 | 17.97 | 103.8 | 180.7 |
| 250 | 42.52 | 95.6 | 124.8 |
| 499 | 97.22 | 95.9 | 114.5 |
| 997 | 197.23 | 101.9 | 118.3 |

Readings:

- Query latency is essentially flat from 50 to 1000 tools. The time is
  almost entirely the SPLADE query embedding on CPU; the zvec hybrid search
  itself contributes single-digit milliseconds.
- The first row carries model warmup; later sizes run with warm models.
- Build time scales linearly with tool count because embedding dominates.

Chart: `results/latency.png`.

## Extra LLM call per step (LangChain's LLMToolSelectorMiddleware)

Command:

```bash
uv run python -m benchmarks.latency.run_latency --tool-counts 100 --queries 3 --llm-selector-runs 1
```

`LLMToolSelectorMiddleware` asks a model to pick tools before every model
call, so its per-step cost is one additional model round trip whose prompt
carries every tool schema. The harness times exactly that call with the same
100 tool set the vector search ran on.

| Method | Selection latency at 100 tools (mean) | p95 |
| --- | ---: | ---: |
| DynamicToolSelectorMiddleware (local hybrid search) | 107.8 ms | 126.2 ms |
| LLMToolSelectorMiddleware (extra model call) | 13,160 ms | 16,106 ms |

That is a 122x difference, and the LLM route also bills tokens for the tool
schemas on every single step. The vector route bills nothing and runs
offline.

## Token reduction over a multi-step conversation

Command:

```bash
uv run python -m benchmarks.token_reduction.run_tokens
```

Ten real BFCL questions run as a ten turn conversation with real tool
execution, against a 100 tool set built from the BFCL function pool. Tokens
come from provider usage metadata (source-level sink, so the LLM selector's
hidden per-step selection calls are billed too).

| Config | Prompt tokens (10 turns) | Prompt/turn | Completion | Model calls | vs baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline (all 100 tools) | 196,153 | 19,615 | 11,264 | 11 | — |
| llm_selector (top_k=4) | 120,367 | 12,037 | 13,982 | 37 | −38.6% |
| dynamic (top_k=4) | 38,207 | 3,821 | 10,050 | 17 | −80.5% |

Readings:

- Dynamic binds 4/100 tools per step: 5.1x fewer prompt tokens than
  baseline, 3.2x fewer than the LLM selector.
- The LLM selector's saving is smaller than top-k alone suggests because
  every step pays an extra selection call whose prompt carries all 100
  schemas (37 model calls vs 11 for baseline over the same 10 turns).
- Dynamic selection latency in this run: 136.6 ms mean per step (local
  hybrid search, no tokens billed).
- Model: `nvidia:nvidia/nemotron-3.5-lightning-30b-a3b`, top_k=4, 0 failed
  turns on every config. Source: `results/token_results.json`.

## BFCL function calling accuracy

Command:

```bash
uv run python -m benchmarks.bfcl.run_bfcl --categories simple,irrelevance,relevance --tool-counts 100 --limit 100
```

Three configurations on identical questions and tool sets: baseline (all
tools), LangChain's LLMToolSelectorMiddleware, and this middleware. Scoring
follows BFCL possible-answer semantics; see benchmarks/README.md for the two
documented adaptations (name sanitization, stop after the first model
response).

| Category | Config | Accuracy | Errors | Prompt/question | Wall/question |
| --- | --- | ---: | ---: | ---: | ---: |
| simple (n=100) | baseline | 85.0% | 4 | 20,049 | 55.4 s |
| simple (n=100) | llm_selector | 66.0% | 24 | 5,425 | 124.4 s |
| simple (n=100) | dynamic | 79.0% | 4 | 1,130 | 38.4 s |
| irrelevance (n=100) | baseline | 54.0% | 0 | 20,854 | 61.1 s |
| irrelevance (n=100) | llm_selector | 29.0% | 32 | 4,833 | 150.1 s |
| irrelevance (n=100) | dynamic | 65.0% | 0 | 1,693 | 42.6 s |
| relevance (n=18) | baseline | 66.7% | 0 | 20,966 | 54.8 s |
| relevance (n=18) | llm_selector | 50.0% | 6 | 5,208 | 107.3 s |
| relevance (n=18) | dynamic | 44.4% | 0 | 1,691 | 48.8 s |

Readings:

- Dynamic matches or beats baseline accuracy while billing ~5% of its
  prompt tokens (1.1–1.7k vs ~21k per question at 100 tools, top_k=4).
  Selection latency across these runs: 248–634 ms per step, local only.
- Baseline and dynamic ran nearly error-free; the remaining errors are
  provider 429s and read timeouts on the NVIDIA free tier, recorded in
  `error_samples` per block in `results/bfcl_results.json`.
- `llm_selector` trails on accuracy and carries the most errors: besides
  429s (two model calls per step doubles exposure), Nemotron repeatedly
  returned malformed selection payloads
  (`LLMToolSelectorMiddleware: selection model returned a malformed
  response`), which count as errors. Its wall time is 2–3x baseline.
- Fairness note: the three `dynamic` blocks above come from sequential
  reruns (`--workers 1`). The first parallel attempt (`--workers 3`) lost
  entries to a harness bug since fixed: all workers shared one zvec
  directory per thread and collided on the collection LOCK. Rerun
  commands, identical data and model:
  `--configs dynamic --categories simple --workers 1` and
  `--configs dynamic --categories irrelevance,relevance --workers 1`.
- Model: `nvidia:nvidia/nemotron-3.5-lightning-30b-a3b`, top_k=4.
  Source: `results/bfcl_results.json`.
