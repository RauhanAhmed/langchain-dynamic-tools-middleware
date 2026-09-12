"""Tests for embedder protocols, defaults, and tool text rendering."""

from __future__ import annotations

import sys
import types

import pytest
from conftest import FakeDenseEmbedder, FakeSparseEmbedder
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field

from langchain_dynamic_tools._embeddings import (
    DefaultDenseEmbedder,
    DefaultSparseEmbedder,
    DenseEmbedder,
    LangChainDenseEmbedder,
    SparseEmbedder,
    render_tool_text,
)


def test_render_includes_name_description_and_parameters() -> None:
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return "sunny"

    text = render_tool_text(get_weather)
    assert "get_weather" in text
    assert "Get the current weather for a city." in text
    assert "city (string, required)" in text


def test_render_includes_parameter_descriptions() -> None:
    class FlightArgs(BaseModel):
        origin: str = Field(description="IATA code of the departure airport")
        date: str = Field(default="today", description="Departure date")

    @tool(args_schema=FlightArgs)
    def book_flight(origin: str, date: str) -> str:
        """Book a flight."""
        return "booked"

    text = render_tool_text(book_flight)
    assert "origin (string, required): IATA code of the departure airport" in text
    assert "date (string, optional): Departure date" in text


def test_render_falls_back_when_description_is_empty() -> None:
    no_description = StructuredTool.from_function(
        func=lambda value: "ok", name="do_thing", description=""
    )
    text = render_tool_text(no_description)
    assert "do_thing: No description provided." in text


def test_render_survives_a_broken_args_schema() -> None:
    class BrokenArgsTool(BaseTool):
        name: str = "broken_tool"
        description: str = "A tool whose args schema explodes."

        @property
        def args(self) -> dict:  # type: ignore[override]
            raise RuntimeError("boom")

        def _run(self, *args: object, **kwargs: object) -> str:
            return "ok"

    text = render_tool_text(BrokenArgsTool())
    assert "broken_tool" in text
    assert "boom" not in text


def test_fake_embedders_satisfy_the_protocols() -> None:
    assert isinstance(FakeDenseEmbedder(), DenseEmbedder)
    assert isinstance(FakeSparseEmbedder(), SparseEmbedder)


def test_default_dense_embedder_reports_384_dimensions_without_loading() -> None:
    assert DefaultDenseEmbedder().dimension == 384


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_default_embedders_reject_blank_text(blank: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        DefaultDenseEmbedder().embed(blank)
    with pytest.raises(ValueError, match="empty"):
        DefaultSparseEmbedder().embed_document(blank)
    with pytest.raises(ValueError, match="empty"):
        DefaultSparseEmbedder().embed_query(blank)


def test_default_embedders_reject_non_string_text() -> None:
    with pytest.raises(TypeError, match="str"):
        DefaultDenseEmbedder().embed(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="str"):
        DefaultSparseEmbedder().embed_query(["not", "text"])  # type: ignore[arg-type]


class _FakeLocalDense:
    """Stand-in for zvec's DefaultLocalDenseEmbedding."""

    dimension = 384
    instances = 0

    def __init__(self, **kwargs: object) -> None:
        type(self).instances += 1
        self.kwargs = kwargs

    def embed(self, text: str) -> list[float]:
        return [0.5] * self.dimension


class _FakeLocalSparse:
    """Stand-in for zvec's DefaultLocalSparseEmbedding."""

    instances = 0

    def __init__(self, *, encoding_type: str, **kwargs: object) -> None:
        type(self).instances += 1
        self.encoding_type = encoding_type
        self.kwargs = kwargs

    def embed(self, text: str) -> dict[int, float]:
        return {7: 0.25}


@pytest.fixture
def fake_zvec_extension(monkeypatch):
    """Install fake zvec.extension classes and reset their counters."""
    _FakeLocalDense.instances = 0
    _FakeLocalSparse.instances = 0
    module = types.ModuleType("zvec.extension")
    module.DefaultLocalDenseEmbedding = _FakeLocalDense  # type: ignore[attr-defined]
    module.DefaultLocalSparseEmbedding = _FakeLocalSparse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zvec.extension", module)
    return module


def test_default_dense_embedder_loads_lazily_and_reuses_the_model(
    fake_zvec_extension,
) -> None:
    embedder = DefaultDenseEmbedder(device="cpu")
    assert _FakeLocalDense.instances == 0  # nothing loads at construction

    first = embedder.embed("hello")
    second = embedder.embed("again")
    assert first == [0.5] * 384
    assert second == first
    assert _FakeLocalDense.instances == 1  # one model, many calls
    assert embedder.dimension == 384


def test_default_sparse_embedder_uses_both_encodings(fake_zvec_extension) -> None:
    embedder = DefaultSparseEmbedder()
    assert embedder.embed_document("some tool") == {7: 0.25}
    assert embedder.embed_query("some query") == {7: 0.25}
    assert _FakeLocalSparse.instances == 2  # document side and query side


def test_default_embedders_raise_a_helpful_error_when_zvec_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "zvec.extension", None)
    with pytest.raises(ImportError, match=r"\[local\]"):
        DefaultDenseEmbedder().embed("hello")
    with pytest.raises(ImportError, match=r"\[local\]"):
        DefaultSparseEmbedder().embed_document("hello")


def test_default_embedders_raise_a_helpful_error_when_the_model_needs_an_extra(
    fake_zvec_extension, monkeypatch
) -> None:
    class _NeedsExtra:
        def __init__(self, **kwargs: object) -> None:
            msg = "requires sentence-transformers"
            raise ImportError(msg)

    monkeypatch.setattr(fake_zvec_extension, "DefaultLocalDenseEmbedding", _NeedsExtra)
    with pytest.raises(ImportError, match=r"\[local\]"):
        DefaultDenseEmbedder().embed("hello")


class _FakeLangChainEmbeddings(Embeddings):
    """Deterministic stand-in for e.g. OpenAIEmbeddings (no network)."""

    calls: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        type(self).calls += 1
        return [[float(len(text) % 7)] * 8 for text in texts]

    def embed_query(self, text: str) -> list[float]:
        type(self).calls += 1
        return [float(len(text) % 7)] * 8


@pytest.fixture
def _reset_calls():
    _FakeLangChainEmbeddings.calls = 0


def test_langchain_adapter_satisfies_the_dense_protocol(_reset_calls) -> None:
    adapter = LangChainDenseEmbedder(_FakeLangChainEmbeddings())
    assert isinstance(adapter, DenseEmbedder)
    assert adapter.embed("hello world") == [float(len("hello world") % 7)] * 8


def test_langchain_adapter_probes_dimension_lazily(_reset_calls) -> None:
    adapter = LangChainDenseEmbedder(_FakeLangChainEmbeddings())
    assert _FakeLangChainEmbeddings.calls == 0  # nothing at construction
    assert adapter.dimension == 8
    assert _FakeLangChainEmbeddings.calls == 1  # one probe, then cached
    assert adapter.dimension == 8
    assert _FakeLangChainEmbeddings.calls == 1


def test_langchain_adapter_embed_first_costs_no_extra_call(_reset_calls) -> None:
    adapter = LangChainDenseEmbedder(_FakeLangChainEmbeddings())
    adapter.embed("hello")
    assert adapter.dimension == 8
    assert _FakeLangChainEmbeddings.calls == 1  # dimension came from that vector


def test_langchain_adapter_explicit_dimension_wins_and_validates(_reset_calls) -> None:
    adapter = LangChainDenseEmbedder(_FakeLangChainEmbeddings(), dimension=8)
    assert adapter.dimension == 8
    assert _FakeLangChainEmbeddings.calls == 0  # no probe needed
    with pytest.raises(ValueError, match="mismatch"):
        LangChainDenseEmbedder(_FakeLangChainEmbeddings(), dimension=3).embed("hello")
    with pytest.raises(ValueError, match="dimension"):
        LangChainDenseEmbedder(_FakeLangChainEmbeddings(), dimension=0)


def test_langchain_adapter_rejects_bad_text(_reset_calls) -> None:
    adapter = LangChainDenseEmbedder(_FakeLangChainEmbeddings())
    with pytest.raises(ValueError, match="empty"):
        adapter.embed("   ")
    with pytest.raises(TypeError, match="str"):
        adapter.embed(123)  # type: ignore[arg-type]
