#!/usr/bin/env bash
# Run a command inside the Linux dev container.
#
# Usage:
#   ./scripts/dev-container.sh uv sync --all-extras
#   ./scripts/dev-container.sh uv run pytest
#   ./scripts/dev-container.sh uv run ruff check .
#
# The project directory is mounted at /workspace and the virtual environment
# lives at /opt/venv inside the container, so the host repo stays clean.
# Model downloads and package caches persist in the named volume
# langchain-dynamic-tools-cache.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="langchain-dynamic-tools-dev"
docker build --quiet -t "$IMAGE" -f Dockerfile.dev .

ENV_ARGS=()
if [ -f .env ]; then
  ENV_ARGS=(--env-file .env)
fi

TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-it)
fi

exec docker run --rm \
  ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
  ${TTY_ARGS[@]+"${TTY_ARGS[@]}"} \
  -v "$PWD":/workspace \
  -v langchain-dynamic-tools-cache:/root/.cache \
  -w /workspace \
  "$IMAGE" "$@"
