# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `LangChainDenseEmbedder`, an adapter that lets any standard LangChain
  `Embeddings` object (e.g. `OpenAIEmbeddings`) be used as `dense_embedder`.
  `ToolVectorIndex` applies it automatically; `dense_dim` doubles as the
  explicit dimension so no probing API call is needed.
- `dense_dim` parameter on `DynamicToolSelectorMiddleware`, forwarded to the
  index.
- Multi-turn aware search query: the latest user message is used, plus the
  previous one when the latest is a very short follow-up. Queries are capped
  at 2000 characters.

### Fixed

- Benchmark harness: set the missing category on shared args (every BFCL
  entry used to fail), unique index directory per parallel entry (zvec lock
  collisions), per-block checkpointing for token runs, `--configs` filter for
  rerunning single blocks, and per-block error samples in saved results.

## [0.1.0] - 2026-09-11

### Added

- `DynamicToolSelectorMiddleware`, a LangChain agent middleware that indexes
  every tool into a local zvec collection with hybrid dense and sparse
  embeddings and hands the model only the top-k most relevant tools per step.
- `ToolVectorIndex`, the zvec-backed index with incremental sync, reciprocal
  rank fusion of dense and sparse rankings, and collection lifecycle handling.
- `DefaultDenseEmbedder` and `DefaultSparseEmbedder`, thin wrappers around the
  local models that ship with zvec (all-MiniLM-L6-v2 and SPLADE), plus
  protocols for plugging in any embedding backend.
- `render_tool_text`, the compact tool representation used for embedding.
- Graceful degradation: search failures fall back to all tools unless the
  caller opts into raising.
- Benchmark suite: BFCL accuracy harness, selection latency at scale, and
  token reduction measurements with committed, reproducible results.
