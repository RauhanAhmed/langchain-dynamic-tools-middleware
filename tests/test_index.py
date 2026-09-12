"""Tests for the zvec-backed ToolVectorIndex."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import (
    CountingDenseEmbedder,
    CountingSparseEmbedder,
    FakeDenseEmbedder,
    FakeSparseEmbedder,
)
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool

from langchain_dynamic_tools._index import ToolVectorIndex


@pytest.fixture
def index_path(tmp_path):
    return str(tmp_path / "tool-index")


def _make_index(path, tools, dense=None, sparse=None, dense_dim=None):
    return ToolVectorIndex(
        tools,
        path=path,
        dense_embedder=dense or FakeDenseEmbedder(),
        sparse_embedder=sparse or FakeSparseEmbedder(),
        dense_dim=dense_dim,
    )


def test_search_ranks_the_matching_tool_first(index_path, sample_tools) -> None:
    index = _make_index(index_path, sample_tools)
    try:
        hits = index.search("what is the weather forecast in tokyo", top_k=3)
        assert hits[0][0] == "get_weather"
    finally:
        index.close()


def test_search_returns_at_most_top_k(index_path, sample_tools) -> None:
    index = _make_index(index_path, sample_tools)
    try:
        assert len(index.search("send an email", top_k=5)) == 5
        assert len(index.search("send an email", top_k=50)) == len(sample_tools)
    finally:
        index.close()


def test_search_with_blank_query_returns_nothing(index_path, sample_tools) -> None:
    index = _make_index(index_path, sample_tools)
    try:
        assert index.search("   ", top_k=3) == []
    finally:
        index.close()


def test_search_rejects_bad_arguments(index_path, sample_tools) -> None:
    index = _make_index(index_path, sample_tools)
    try:
        with pytest.raises(ValueError, match="top_k"):
            index.search("anything", top_k=0)
        with pytest.raises(TypeError, match="str"):
            index.search(42, top_k=3)  # type: ignore[arg-type]
    finally:
        index.close()


def test_reopening_skips_unchanged_tools(index_path, sample_tools) -> None:
    first_dense = CountingDenseEmbedder()
    first_sparse = CountingSparseEmbedder()
    index = _make_index(index_path, sample_tools, dense=first_dense, sparse=first_sparse)
    index.close()
    assert first_dense.calls == len(sample_tools)
    assert first_sparse.calls == len(sample_tools)

    second_dense = CountingDenseEmbedder()
    second_sparse = CountingSparseEmbedder()
    reopened = _make_index(index_path, sample_tools, dense=second_dense, sparse=second_sparse)
    try:
        assert second_dense.calls == 0
        assert second_sparse.calls == 0
        assert reopened.search("weather in tokyo", top_k=1)[0][0] == "get_weather"
    finally:
        reopened.close()


def test_sync_reembeds_changed_tools_and_deletes_removed_ones(index_path) -> None:
    old_email = StructuredTool.from_function(
        func=lambda recipient: "ok",
        name="send_email",
        description="Send an email message to a recipient.",
    )
    weather = StructuredTool.from_function(
        func=lambda city: "sunny",
        name="get_weather",
        description="Get the current weather for a city.",
    )
    dense = CountingDenseEmbedder()
    sparse = CountingSparseEmbedder()
    index = _make_index(index_path, [old_email, weather], dense=dense, sparse=sparse)
    try:
        assert dense.calls == 2

        changed_email = StructuredTool.from_function(
            func=lambda recipient: "ok",
            name="send_email",
            description="Send an email message to a recipient with a subject and a body.",
        )
        index.sync([changed_email, weather])
        assert dense.calls == 3  # only the changed tool was re-embedded
        assert sparse.calls == 3

        index.sync([weather])  # prune=True drops send_email
        hits = index.search("send an email to the team", top_k=10)
        assert all(name != "send_email" for name, _ in hits)
    finally:
        index.close()


def test_sync_without_prune_keeps_existing_tools(index_path) -> None:
    weather = StructuredTool.from_function(
        func=lambda city: "sunny",
        name="get_weather",
        description="Get the current weather for a city.",
    )
    email = StructuredTool.from_function(
        func=lambda recipient: "ok",
        name="send_email",
        description="Send an email message to a recipient.",
    )
    index = _make_index(index_path, [weather])
    try:
        index.sync([email], prune=False)
        found = {name for name, _ in index.search("weather", top_k=10)}
        assert "get_weather" in found
        found = {name for name, _ in index.search("email", top_k=10)}
        assert "send_email" in found
    finally:
        index.close()


def test_duplicate_tool_names_are_rejected(index_path) -> None:
    first = StructuredTool.from_function(func=lambda: "ok", name="clash", description="First tool.")
    second = StructuredTool.from_function(
        func=lambda: "ok", name="clash", description="Second tool."
    )
    with pytest.raises(ValueError, match="Duplicate tool name"):
        _make_index(index_path, [first, second])


def test_dimension_mismatch_rebuilds_the_collection(index_path, sample_tools) -> None:
    index = _make_index(index_path, sample_tools, dense_dim=384)
    index.close()

    rebuilt = _make_index(index_path, sample_tools, dense=FakeDenseEmbedder(128), dense_dim=128)
    try:
        hits = rebuilt.search("weather forecast for tokyo", top_k=3)
        assert hits[0][0] == "get_weather"
    finally:
        rebuilt.close()


def test_context_manager_closes_the_index(index_path, sample_tools) -> None:
    with _make_index(index_path, sample_tools) as index:
        index.search("weather", top_k=1)
    with pytest.raises(RuntimeError, match="closed"):
        index.search("weather", top_k=1)


def test_close_is_idempotent(index_path, sample_tools) -> None:
    index = _make_index(index_path, sample_tools)
    index.close()
    index.close()  # no error


def test_sync_after_close_raises(index_path, sample_tools) -> None:
    index = _make_index(index_path, sample_tools)
    index.close()
    with pytest.raises(RuntimeError, match="closed"):
        index.sync(sample_tools)


def test_empty_directory_at_path_is_recovered(index_path, sample_tools) -> None:
    os.makedirs(index_path)  # a pre-created empty stub, e.g. from a temp dir helper
    index = _make_index(index_path, sample_tools)
    try:
        assert index.search("weather in tokyo", top_k=1)[0][0] == "get_weather"
    finally:
        index.close()


def test_unusable_collection_path_raises_a_clear_error(index_path, sample_tools) -> None:
    os.makedirs(index_path)
    (Path(index_path) / "stray.txt").write_text("not a zvec collection")
    with pytest.raises(RuntimeError, match="not a usable zvec collection"):
        _make_index(index_path, sample_tools)


def test_empty_path_is_rejected(sample_tools) -> None:
    with pytest.raises(ValueError, match="path"):
        ToolVectorIndex(sample_tools, path="   ")


def test_bad_dense_dimension_is_rejected(sample_tools) -> None:
    with pytest.raises(ValueError, match="dense_dim"):
        ToolVectorIndex(
            sample_tools,
            path="/tmp/never-created-index",
            dense_embedder=FakeDenseEmbedder(),
            sparse_embedder=FakeSparseEmbedder(),
            dense_dim=0,
        )


def test_tools_with_blank_names_are_rejected(index_path) -> None:
    class NamelessTool(BaseTool):
        name: str = "   "
        description: str = "A tool whose name is only whitespace."

        def _run(self, *args: object, **kwargs: object) -> str:
            return "ok"

    with pytest.raises(ValueError, match="empty name"):
        _make_index(index_path, [NamelessTool()])


class _FakeLangChainEmbeddings(Embeddings):
    """Deterministic stand-in for e.g. OpenAIEmbeddings (no network)."""

    calls: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        type(self).calls += 1
        return [[float(len(text) % 7)] * 8 for text in texts]

    def embed_query(self, text: str) -> list[float]:
        type(self).calls += 1
        return [float(len(text) % 7)] * 8


def test_index_accepts_langchain_embeddings_directly(index_path, sample_tools) -> None:
    _FakeLangChainEmbeddings.calls = 0
    index = ToolVectorIndex(
        sample_tools,
        path=index_path,
        dense_embedder=_FakeLangChainEmbeddings(),  # type: ignore[arg-type]
        sparse_embedder=FakeSparseEmbedder(),
        dense_dim=8,
    )
    try:
        assert index.search("weather in tokyo", top_k=2)
    finally:
        index.close()
    assert _FakeLangChainEmbeddings.calls > 0


def test_index_rejects_objects_matching_neither_protocol(index_path, sample_tools) -> None:
    with pytest.raises(TypeError, match="dense_embedder"):
        ToolVectorIndex(sample_tools, path=index_path, dense_embedder=object())
