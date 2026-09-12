# Benchmarks

Everything in this directory is a real measurement, not a simulation: the
real zvec engine, the real local embedding models (all-MiniLM-L6-v2 for dense
vectors, SPLADE for sparse), live model calls through the full
`create_agent` stack, and the public Berkeley Function Calling Leaderboard
dataset. Numbers in `RESULTS.md` come only from the JSON files these scripts
write into `results/`, and every table there names the command that produced
it.

## The three configurations

Every benchmark compares the same three agent stacks on identical inputs and
the same chat model:

| Config | What the model sees per step | Selection cost |
| --- | --- | --- |
| `baseline` | All tools, no middleware | None |
| `llm_selector` | Top-k tools picked by LangChain's built-in `LLMToolSelectorMiddleware` | One extra LLM round trip per step |
| `dynamic` | Top-k tools picked by this package's `DynamicToolSelectorMiddleware` | Local hybrid vector search, milliseconds |

## The benchmarks

### BFCL accuracy (`bfcl/run_bfcl.py`)

Function calling accuracy on the Berkeley Function Calling Leaderboard
(Apache 2.0, downloaded from the official Hugging Face dataset repository).
Questions from the `simple`, `live_simple`, and `live_multiple` categories
are padded with real function schemas from the dataset's own pool to reach
the tool counts under test (50, 100, 500). The `irrelevance` category checks
that a stack does not call tools when none fit, and `relevance` checks that
it still calls when a tool does fit.

Scoring follows BFCL's possible-answer semantics: each expected call is
`{function_name: {param: [acceptable values]}}`, a predicted call matches
when every parameter equals one of the acceptable values (or is omitted when
the empty string is acceptable), recursively for nested structures, and
extra parameters fail the match. Two adaptations are documented here:

- BFCL function names can contain dots (`math.factorial`) while provider
  tool-name rules cannot. Names are sanitized for binding, and the same
  transform is applied to expected answers before comparison.
- The harness runs the full `create_agent` stack and stops after the first
  model response, which is the response BFCL scores. The stop middleware is
  applied identically to all three configurations.

### Selection latency (`latency/run_latency.py`)

Wall-clock time for one selection (embed the query both ways, hybrid zvec
search, reciprocal rank fusion) at index sizes from 10 to 1000 tools, plus
index build time. For contrast, the harness also times the dominant cost of
LLM-based selection: one extra model call whose prompt carries every tool
schema, which is exactly what `LLMToolSelectorMiddleware` does per step.

### Token reduction (`token_reduction/run_tokens.py`)

A fixed ten turn conversation with real tool execution runs once per
configuration against a 100 tool set built from the BFCL function pool.
Token counts come from provider usage metadata, so they are what the
provider actually billed, and costs use the price table in `common.py`.

## Reproducing

Run from the project root with the `[local]` and `[bench]` extras installed:

```bash
uv run python -m benchmarks.latency.run_latency
uv run python -m benchmarks.token_reduction.run_tokens
uv run python -m benchmarks.bfcl.run_bfcl --categories simple,irrelevance,relevance --tool-counts 100 --limit 100
```

Model calls need an API key. By default the harness uses OpenRouter (set
`OPENROUTER_API_KEY`); `--provider openai --model gpt-4o-mini` switches to
OpenAI (set `OPENAI_API_KEY`). Keys are read from the environment or a
`.env` file, which is gitignored and never committed.

The BFCL data files are downloaded once into `bfcl/data/` and cached. That
cache directory is gitignored; the runner fetches fresh copies as needed.

## Fairness notes

- Same questions, same tool sets, same model, same top-k for both selectors.
- The dynamic configuration pays its one-time indexing cost per run; it is
  reported separately (`index_seconds_mean`) and never mixed into per-step
  selection latency, because a deployed application indexes once and then
  serves many steps.
- Free-tier models rate limit under load. The harness sleeps between
  questions and retries with backoff; failed requests are counted as errors
  and reported, never silently dropped.
