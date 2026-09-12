# langchain-dynamic-tools-middleware

[![PyPI version](https://img.shields.io/pypi/v/langchain-dynamic-tools-middleware.svg)](https://pypi.org/project/langchain-dynamic-tools-middleware/)
[![Python versions](https://img.shields.io/pypi/pyversions/langchain-dynamic-tools-middleware.svg)](https://pypi.org/project/langchain-dynamic-tools-middleware/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/RauhanAhmed/langchain-dynamic-tools-middleware/actions/workflows/ci.yml/badge.svg)](https://github.com/RauhanAhmed/langchain-dynamic-tools-middleware/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen.svg)](benchmarks/RESULTS.md)

LangChain agent middleware that hands the model only the tools it needs for
the current step. Local hybrid vector search picks the top-k relevant tools
in about 100 ms, with zero extra LLM calls and zero selection tokens.

| | Full tool set | LLM selector | **This middleware** |
| --- | ---: | ---: | ---: |
| Prompt tokens, 10-turn run (100 tools) | 196,153 | 120,367 | **38,207 (−80.5%)** |
| Selection latency at 100 tools | 0 ms | 13,160 ms | **108 ms (122x faster)** |
| BFCL simple accuracy | 85% | 66% | **79%** |
| BFCL irrelevance accuracy | 54% | 29% | **65%** |
| Selection API cost | $0 | billed every step | **$0, runs offline** |

Every number above is measured, not claimed. Methodology, environment, raw
JSON, and reproduction commands live in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

## Contents

- [The problem](#the-problem-it-solves)
- [Install](#install)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Benchmarks](#benchmarks)
- [Comparison](#comparison-with-the-built-in-llm-tool-selector)
- [Configuration](#configuration)
- [Custom embedders](#custom-embedders)
- [Multi-turn conversations](#multi-turn-conversations)
- [Requirements](#requirements)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Development](#development)
- [License](#license)

## The problem it solves

Agents with dozens or hundreds of tools pay for all of them on every single
model call. Tool schemas eat context, cost money, and make the model worse at
choosing: lookalike options blur together. LangChain's built-in
`LLMToolSelectorMiddleware` fixes the symptom with another LLM call per step,
which adds its own latency, tokens, and bill (our measurements: +26 model
calls and only −38.6% tokens over 10 turns, plus malformed-selection errors
on smaller models).

This middleware takes a different path. It indexes every tool once into a
local [zvec](https://github.com/alibaba/zvec) collection (an in-process
vector engine built in Rust) using both a dense embedding and a sparse
embedding. Before each model call it embeds the search query, runs one hybrid
search, fuses the two rankings with reciprocal rank fusion, and hands the
model only the top-k winners. The full tool set stays registered with the
agent, so any selected tool still executes normally.

## Install

```bash
pip install "langchain-dynamic-tools-middleware[local]"
```

The `[local]` extra installs sentence-transformers, which powers the default
embedding models (about a 500 MB download on first use, then cached and fully
offline). Skip the extra if you bring your own embedders (OpenAI, Jina, Qwen,
Ollama, or any LangChain `Embeddings` object).

Requires Python 3.10 or newer and `langchain>=1.0`.

## Quickstart

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_dynamic_tools import DynamicToolSelectorMiddleware


# 1. Fifty or hundreds of available tools
all_tools = [get_weather, query_sql, send_email, git_push, ...]

# 2. Add the dynamic middleware. Every tool is indexed once, locally.
tool_router = DynamicToolSelectorMiddleware(
    tools=all_tools,
    top_k=4,
)

# 3. Create the agent. The full set stays available for execution, but each
#    model call only carries the tools relevant to the user's message.
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=all_tools,
    middleware=[tool_router],
)
```

That is the whole integration. Tools are standard LangChain tools built with
the `@tool` decorator or `StructuredTool`. A runnable six-tool demo lives in
[`examples/quickstart.py`](examples/quickstart.py), and
[`examples/custom_embedders.py`](examples/custom_embedders.py) shows OpenAI
embeddings.

## How it works

```text
index time (once)                    every model call
────────────────                    ─────────────────
tool name + description              user message
  + parameters                           │
       │                          embed dense + sparse
 embed dense (MiniLM)                    │
 embed sparse (SPLADE)            zvec hybrid query
       │                          RRF fusion
 zvec collection ──────────────► top-k tool names
 on local disk                   model sees only those
```

1. **Indexing, once.** Each tool is rendered as one compact text string:
   name, description, parameters with types and descriptions. The text is
   embedded into a dense vector (all-MiniLM-L6-v2, 384 dims) and a sparse
   lexical vector (SPLADE), and stored in a zvec collection on local disk at
   `.dynamicToolsMiddleware/`.
2. **Per model call.** The middleware builds a search query from the
   conversation (latest user message, plus the previous one when the latest
   is a very short follow-up like "and in Berlin?"), embeds it both ways,
   and queries the collection once. zvec fuses the dense and sparse rankings
   with RRF and returns the best tool names in milliseconds.
3. **Override.** The model request is rewritten so the model sees only the
   selected tools. Provider tool dicts pass through untouched, tools
   registered after startup are indexed on the fly, and if a search ever
   fails the agent falls back to all tools rather than breaking.

Because the collection persists, restarting your application with the same
tools re-embeds nothing. Change a tool's description and only that tool is
re-embedded on the next sync.

## Benchmarks

Measured with the harness in [`benchmarks/`](benchmarks/README.md): real
zvec engine, real local embedding models, live model calls through the full
`create_agent` stack, Berkeley Function Calling Leaderboard data, token
counts from provider usage metadata. Full tables in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

**Selection latency** stays flat while the tool count grows 20x (the cost is
almost entirely the local SPLADE embedding, zvec search is single-digit ms):

| Tools | Build (one time) | Query mean | Query p95 |
| ---: | ---: | ---: | ---: |
| 50 | 9.8 s | 106.5 ms | 215.2 ms |
| 100 | 18.0 s | 103.8 ms | 180.7 ms |
| 250 | 42.5 s | 95.6 ms | 124.8 ms |
| 997 | 197.2 s | 101.9 ms | 118.3 ms |

**10-turn conversation, 100 tools:** dynamic bills 38,207 prompt tokens vs
196,153 for the full set (−80.5%) and 120,367 for the LLM selector (−68.3%
relative). The LLM selector needs 37 model calls for the same 10 turns
(dynamic: 17, baseline: 11).

**BFCL accuracy at 100 tools** (dynamic matches or beats the full set while
billing ~5% of its tokens):

| Category | Baseline (all tools) | LLM selector | Dynamic (top 4) |
| --- | ---: | ---: | ---: |
| simple (n=100) | 85% | 66% | 79% |
| irrelevance (n=100) | 54% | 29% | 65% |
| relevance (n=18) | 66.7% | 50% | 44.4% |

## Comparison with the built-in LLM tool selector

| | `LLMToolSelectorMiddleware` | `DynamicToolSelectorMiddleware` |
| --- | --- | --- |
| Selection method | Extra LLM call per step | Local hybrid vector search |
| Selection latency (100 tools) | ~13.2 s mean | ~108 ms mean |
| Selection token cost | Full schemas billed every step | Zero, runs offline |
| Lexical matching (exact names, rare terms) | Depends on the model | Native, via sparse SPLADE vectors |
| Semantic matching (paraphrases) | Yes | Yes, via dense embeddings |
| Robustness on small models | Malformed selections observed | No generation involved |
| Privacy | Sends conversation to selector model | Nothing leaves your machine |

The two compose: cut 200 tools to 20 with vector search, then 20 to 5 with
an LLM selector. For most applications the vector step alone is enough.

## Configuration

```python
DynamicToolSelectorMiddleware(
    tools=all_tools,  # required: the agent's tools
    top_k=4,  # tools handed to the model per step
    path=".dynamicToolsMiddleware",  # local zvec collection location
    dense_embedder=None,  # optional: .dimension/.embed(text), or any LangChain Embeddings
    sparse_embedder=None,  # optional: any object with .embed_document and .embed_query
    dense_dim=None,  # optional: override; skips dimension probing for LangChain embedders
    always_include=None,  # tool names added to every selection, beyond top_k
    reranker=None,  # optional: zvec reranker, defaults to RrfReRanker(60)
    on_error="fallback_all",  # or "raise" on search failure
)
```

| Parameter | Default | Notes |
| --- | --- | --- |
| `tools` | required | BaseTool instances get indexed, provider tool dicts pass through |
| `top_k` | `4` | Must be at least 1 |
| `path` | `.dynamicToolsMiddleware` | Collection persists between runs, one path per agent |
| `always_include` | `None` | Never filtered out, does not count against top_k |
| `on_error` | `fallback_all` | `fallback_all` keeps every tool if the search fails |
| `dense_embedder` | local MiniLM | LangChain `Embeddings`, OpenAI, Jina, Qwen, or your own model |
| `sparse_embedder` | local SPLADE | Any `{index: weight}` embedder works |
| `dense_dim` | embedder's own | Override when the dimension is known up front |

Other useful methods: `middleware.sync()` re-indexes after you edit tool
descriptions, `middleware.close()` releases the collection's file lock.

## Custom embedders

Standard LangChain embeddings drop straight in (adapted automatically,
`embed()` delegates to `embed_query()`):

```python
from langchain_openai import OpenAIEmbeddings
from langchain_dynamic_tools import DefaultSparseEmbedder, DynamicToolSelectorMiddleware

tool_router = DynamicToolSelectorMiddleware(
    tools=all_tools,
    top_k=4,
    dense_embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
    dense_dim=1536,  # skips the one-time dimension probe call
    sparse_embedder=DefaultSparseEmbedder(),
)
```

zvec's own dense wrappers work too (needs `pip install openai`), as do Jina,
Qwen, Ollama, or LM Studio endpoints:

```python
from zvec.extension import HTTPDenseEmbedding, OpenAIDenseEmbedding

dense_embedder=OpenAIDenseEmbedding(model="text-embedding-3-small"),
# or fully local servers, stdlib HTTP only, dimension auto-detected:
dense_embedder=HTTPDenseEmbedding(base_url="http://localhost:11434", model="nomic-embed-text"),
```

Cloud sparse embeddings need a tiny two-sided wrapper, because zvec fixes
the query/document encoding per instance:

```python
from zvec.extension import QwenSparseEmbedding

class DualSparse:
    def __init__(self, doc_model, query_model):
        self._doc, self._query = doc_model, query_model
    def embed_document(self, text): return self._doc.embed(text)
    def embed_query(self, text): return self._query.embed(text)

sparse_embedder=DualSparse(
    QwenSparseEmbedding(dimension=1024, encoding_type="document"),
    QwenSparseEmbedding(dimension=1024, encoding_type="query"),
),
```

The index adapts its vector dimension to whatever your dense embedder
produces, and rebuilds the collection automatically if you switch embedders
later. See [`examples/custom_embedders.py`](examples/custom_embedders.py).

## Multi-turn conversations

The search query is the latest user message. When it is very short (under 40
characters, e.g. "and in Berlin?"), the previous user message is prepended
so follow-ups still match the right tools. Queries are capped at 2000
characters to keep embedding cost flat. Tool outputs are never part of the
query: they are large and lexically noisy.

## Requirements

- Python 3.10+
- `langchain>=1.0`, `zvec>=0.5`
- Default embedders: `pip install "langchain-dynamic-tools-middleware[local]"`
  (sentence-transformers, ~500 MB first download, then offline)
- zvec wheels: Linux x86_64/aarch64, macOS ARM64, Windows x86_64. Intel Macs
  use the Linux dev container (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## API reference

| Name | Kind | One line |
| --- | --- | --- |
| `DynamicToolSelectorMiddleware` | class | The middleware, drop into `create_agent(middleware=[...])` |
| `ToolVectorIndex` | class | The zvec-backed index: `sync()`, `search()`, `close()` |
| `DefaultDenseEmbedder` | class | Local all-MiniLM-L6-v2 wrapper, 384 dims, lazy load |
| `DefaultSparseEmbedder` | class | Local SPLADE wrapper, query/document encodings |
| `LangChainDenseEmbedder` | class | Adapter for any LangChain `Embeddings` object |
| `DenseEmbedder` | protocol | `.dimension` + `.embed(text)` |
| `SparseEmbedder` | protocol | `.embed_document(text)` + `.embed_query(text)` |
| `render_tool_text` | function | The compact tool string that gets embedded |

## Troubleshooting

**`Can't lock read-write collection ... LOCK`**
Two writers share one collection path. Give each middleware its own `path`
and call `close()` (or use a context manager) before reopening. Parallel
benchmark workers each get their own directory for the same reason.

**First run is slow.**
Model downloads (~500 MB) plus one-time indexing, linear in tool count
(about 200 s for 1000 tools on CPU). Later runs skip unchanged tools
entirely. Switch to API embedders to skip downloads.

**Selection looks wrong for a tool.**
Print `render_tool_text(tool)`: vague descriptions embed vaguely. A
one-sentence description naming the entity and action ("Send an email
message to a recipient") beats a clever one.

**`AttributeError` on a custom embedder.**
Dense needs `.dimension` and `.embed(text)`; standard LangChain embeddings
are adapted automatically. Sparse needs `.embed_document` and
`.embed_query` on one object; wrap two single-encoding zvec models as shown
above.

**Dimension mismatch after switching embedders.**
The collection rebuilds automatically when the stored dimension differs.
A stale lock or foreign path raises a clear error naming the path.

## FAQ

**Does the agent still execute all tools?**
Yes. Selection only changes what the model sees per step. Tool execution is
untouched, and a selected tool always resolves to the real implementation.

**What happens on the first run?**
The default embedders download their models once, index your tools, and
store everything under the collection path. Later runs are instant until a
tool changes.

**Can I use one collection for several agents?**
Give each middleware its own `path`. Collections hold an exclusive write
lock, so sharing one path between concurrently writing middlewares is not
supported.

**Does my conversation leave my machine?**
Selection never calls out with the default or Ollama-style embedders: search
is local. Only your actual model provider sees your conversation, same as
without this middleware. API embedders (OpenAI, Jina, Qwen) send short tool
and query texts to that provider.

**Which top_k should I use?**
4 is the measured default: 79% BFCL simple accuracy at ~5% of baseline
tokens. Raise it when tools overlap heavily, lower it for cost. The
benchmark harness reruns any top_k with `--top-k`.

**Windows support?**
Yes, zvec ships Windows x86_64 wheels. macOS runs on Apple Silicon; Intel
Macs can use the Linux dev container, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Development

```bash
uv sync --all-extras
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide, including the
container workflow and the benchmark harness.

## License

[MIT](LICENSE)

## Author

Built by [Rauhan Ahmed Siddiqui](https://rauhanahmed.in).

- GitHub: [RauhanAhmed](https://github.com/RauhanAhmed)
- LinkedIn: [Rauhan Ahmed](https://www.linkedin.com/in/rauhan-ahmed)
- X: [@ahmed_rauh46040](https://x.com/ahmed_rauh46040)
