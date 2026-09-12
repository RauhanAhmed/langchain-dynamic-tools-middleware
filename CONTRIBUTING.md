# Contributing

Thanks for wanting to help. This guide covers setup, the workflow, and the
house style.

## Setup

You need Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/RauhanAhmed/langchain-dynamic-tools-middleware.git
cd langchain-dynamic-tools-middleware
uv sync --all-extras
```

That installs the package (editable), the dev tools, the local embedding
models, and the benchmark extras.

### A note on platforms

zvec, the vector engine underneath this package, publishes wheels for Linux
(x86_64 and aarch64), macOS ARM64, and Windows x86_64. If you are on one of
those, `uv sync` just works. On an Intel Mac there are no zvec wheels, so run
everything through the Linux dev container instead:

```bash
./scripts/dev-container.sh uv sync --all-extras
./scripts/dev-container.sh uv run pytest
```

The container mounts the repository and keeps its virtual environment and
model caches in Docker volumes, so nothing pollutes your checkout.

## Everyday commands

```bash
uv run pytest                 # fast tests, no model downloads
uv run pytest -m local         # adds the real local model smoke test
uv run ruff check .            # lint
uv run mypy                    # strict type check of the package
uv build                       # sdist and wheel
```

## Tests

Fast tests use deterministic fake embedders so they run offline in seconds.
The `local` marker runs the real embedding models once; keep anything that
downloads a model or calls an API behind that marker.

New behavior needs a test. If you touch the middleware, cover both the sync
and async paths.

## Benchmarks

The benchmark suite lives in `benchmarks/` and produces the numbers committed
in `benchmarks/RESULTS.md`. Runs use real components: the real zvec engine,
the real local embedding models, and live model calls.

```bash
uv run python -m benchmarks.latency.run_latency
uv run python -m benchmarks.token_reduction.run_tokens
uv run python -m benchmarks.bfcl.run_bfcl --categories simple --limit 50
```

API keys come from environment variables or a `.env` file (never committed).
If your change affects selection quality, latency, or token usage, rerun the
relevant benchmark and update `RESULTS.md` with the fresh numbers and the
exact command you ran.

## House style

- No em-dashes anywhere: README, docstrings, comments, changelog entries.
  Use commas, colons, or parentheses instead.
- Plain, direct prose. Short sentences. Say what the code does.
- Every public function and class gets a docstring with one short example.
- Type hints everywhere; `mypy --strict` runs on the package.
- Keep the public API small. Internal modules start with an underscore.

## Pull requests

1. Fork, branch, make the change with tests.
2. Run the full check locally: `uv run ruff check .`, `uv run mypy`,
   `uv run pytest`.
3. Open a pull request describing what changed and how you verified it.

Small, focused pull requests get reviewed faster.
