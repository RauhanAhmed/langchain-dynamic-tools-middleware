"""End to end tests for DynamicToolSelectorMiddleware with a recording model."""

from __future__ import annotations

import pytest
from conftest import (
    ExplodingSparseEmbedder,
    FakeDenseEmbedder,
    FakeSparseEmbedder,
    make_recording_model,
)
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from langchain_dynamic_tools import DynamicToolSelectorMiddleware
from langchain_dynamic_tools._middleware import _selection_query


def _make_middleware(tools, tmp_path, **kwargs):
    defaults = dict(
        path=str(tmp_path / "tool-index"),
        dense_embedder=FakeDenseEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
    )
    defaults.update(kwargs)
    return DynamicToolSelectorMiddleware(tools=tools, **defaults)


def test_only_top_k_tools_reach_the_model(tmp_path, sample_tools) -> None:
    model, recorded = make_recording_model(responses=[AIMessage("All done.")])
    middleware = _make_middleware(sample_tools, tmp_path, top_k=2)
    agent = create_agent(model=model, tools=sample_tools, middleware=[middleware])
    try:
        agent.invoke({"messages": [HumanMessage("What is the weather in Tokyo right now?")]})
        assert len(recorded) == 1
        assert len(recorded[0]) == 2
        assert "get_weather" in recorded[0]
    finally:
        middleware.close()


def test_always_include_tools_are_appended_beyond_top_k(tmp_path, sample_tools) -> None:
    model, recorded = make_recording_model(responses=[AIMessage("All done.")])
    middleware = _make_middleware(sample_tools, tmp_path, top_k=1, always_include=["send_email"])
    agent = create_agent(model=model, tools=sample_tools, middleware=[middleware])
    try:
        agent.invoke({"messages": [HumanMessage("Book a flight to Berlin")]})
        assert recorded[0] == ["book_flight", "send_email"]
    finally:
        middleware.close()


def test_provider_tool_dicts_are_passed_through(tmp_path, sample_tools) -> None:
    model, recorded = make_recording_model(responses=[AIMessage("All done.")])
    middleware = _make_middleware(sample_tools, tmp_path, top_k=1)
    agent_tools = [*sample_tools, {"type": "web_search_preview"}]
    agent = create_agent(model=model, tools=agent_tools, middleware=[middleware])
    try:
        agent.invoke({"messages": [HumanMessage("Push my commits to main")]})
        assert "web_search_preview" in recorded[0]
        assert "git_push" in recorded[0]
    finally:
        middleware.close()


def test_all_tools_pass_through_without_a_user_message(tmp_path, sample_tools) -> None:
    model, recorded = make_recording_model(responses=[AIMessage("Nothing to do.")])
    middleware = _make_middleware(sample_tools, tmp_path, top_k=2)
    agent = create_agent(model=model, tools=sample_tools, middleware=[middleware])
    try:
        agent.invoke({"messages": [AIMessage("The user has not spoken yet.")]})
        assert len(recorded[0]) == len(sample_tools)
    finally:
        middleware.close()


def test_failed_search_falls_back_to_all_tools(tmp_path, sample_tools) -> None:
    model, recorded = make_recording_model(responses=[AIMessage("All done.")])
    middleware = _make_middleware(
        sample_tools, tmp_path, top_k=2, sparse_embedder=ExplodingSparseEmbedder()
    )
    agent = create_agent(model=model, tools=sample_tools, middleware=[middleware])
    try:
        agent.invoke({"messages": [HumanMessage("What is the weather?")]})
        assert len(recorded[0]) == len(sample_tools)
    finally:
        middleware.close()


def test_failed_search_can_raise_instead(tmp_path, sample_tools) -> None:
    model, _recorded = make_recording_model(responses=[AIMessage("All done.")])
    middleware = _make_middleware(
        sample_tools,
        tmp_path,
        top_k=2,
        sparse_embedder=ExplodingSparseEmbedder(),
        on_error="raise",
    )
    agent = create_agent(model=model, tools=sample_tools, middleware=[middleware])
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            agent.invoke({"messages": [HumanMessage("What is the weather?")]})
    finally:
        middleware.close()


def test_async_selection_matches_sync_selection(tmp_path, sample_tools) -> None:
    import asyncio

    model, recorded = make_recording_model(responses=[AIMessage("All done.")])
    middleware = _make_middleware(sample_tools, tmp_path, top_k=3)
    agent = create_agent(model=model, tools=sample_tools, middleware=[middleware])
    try:
        asyncio.run(agent.ainvoke({"messages": [HumanMessage("Translate this to French")]}))
        assert len(recorded[0]) == 3
        assert "translate_text" in recorded[0]
    finally:
        middleware.close()


def test_sync_picks_up_changed_tool_descriptions(tmp_path, sample_tools) -> None:
    model, recorded = make_recording_model(responses=[AIMessage("All done.")])
    middleware = _make_middleware(sample_tools, tmp_path, top_k=1)
    agent = create_agent(model=model, tools=sample_tools, middleware=[middleware])
    try:
        target = next(tool for tool in sample_tools if tool.name == "order_pizza")
        target.description = "Order a deep dish pizza with extra cheese and toppings."
        middleware.sync()
        agent.invoke({"messages": [HumanMessage("Order a deep dish pizza please")]})
        assert recorded[0] == ["order_pizza"]
    finally:
        middleware.close()


def test_request_tools_missing_from_the_index_are_added(tmp_path, sample_tools) -> None:
    from langchain_core.tools import tool

    model, recorded = make_recording_model(responses=[AIMessage("All done.")])
    middleware = _make_middleware(sample_tools, tmp_path, top_k=2)

    @tool
    def wind_surf_report(beach: str) -> str:
        """Get the wind and surf report for a beach."""

    agent = create_agent(
        model=model, tools=[*sample_tools, wind_surf_report], middleware=[middleware]
    )
    try:
        agent.invoke({"messages": [HumanMessage("Wind and surf report for Bondi beach")]})
        assert "wind_surf_report" in recorded[0]
    finally:
        middleware.close()


def test_constructor_validates_arguments(tmp_path, sample_tools) -> None:
    with pytest.raises(ValueError, match="top_k"):
        _make_middleware(sample_tools, tmp_path, top_k=0)
    with pytest.raises(ValueError, match="at least one BaseTool"):
        _make_middleware([{"type": "web_search_preview"}], tmp_path)
    with pytest.raises(ValueError, match="always_include"):
        _make_middleware(sample_tools, tmp_path, always_include=["not_a_tool"])


def test_selection_query_variants() -> None:
    long_enough = "What is the weather in Tokyo right now, please?"
    assert _selection_query([HumanMessage(long_enough)]) == long_enough
    blocks = HumanMessage(
        content=[{"type": "text", "text": "what is "}, {"type": "text", "text": "the weather"}]
    )
    assert _selection_query([blocks]) == "what is  the weather"
    assert _selection_query([HumanMessage("   ")]) is None
    assert _selection_query([AIMessage("no user here")]) is None
    assert _selection_query([]) is None


def test_short_follow_up_borrows_previous_user_message() -> None:
    messages = [
        HumanMessage("What is the weather in Tokyo right now?"),
        AIMessage("Sunny, 22C."),
        HumanMessage("and in Berlin?"),
    ]
    assert (
        _selection_query(messages)
        == "What is the weather in Tokyo right now?\nand in Berlin?"
    )


def test_short_first_message_needs_no_context() -> None:
    assert _selection_query([HumanMessage("hi")]) == "hi"


def test_long_query_is_truncated() -> None:
    long_text = "x" * 3000
    assert _selection_query([HumanMessage(long_text)]) == "x" * 2000


def test_select_tools_returns_none_without_base_tools(tmp_path, sample_tools) -> None:
    from types import SimpleNamespace

    middleware = _make_middleware(sample_tools, tmp_path, top_k=2)
    try:
        request = SimpleNamespace(
            tools=[{"type": "web_search_preview"}],
            messages=[HumanMessage("weather?")],
        )
        assert middleware._select_tools(request) is None
        request = SimpleNamespace(tools=[], messages=[HumanMessage("weather?")])
        assert middleware._select_tools(request) is None
    finally:
        middleware.close()


def test_index_directory_is_created_at_the_given_path(tmp_path, sample_tools) -> None:
    index_dir = tmp_path / "custom-location"
    middleware = _make_middleware(sample_tools, index_dir, top_k=1)
    try:
        assert index_dir.exists()
    finally:
        middleware.close()


@pytest.mark.local
def test_default_local_embedders_end_to_end(tmp_path, sample_tools) -> None:
    """Real model smoke test: MiniLM plus SPLADE index and rank tools correctly."""
    model, recorded = make_recording_model(responses=[AIMessage("All done.")])
    middleware = DynamicToolSelectorMiddleware(
        tools=sample_tools,
        top_k=2,
        path=str(tmp_path / "tool-index"),
    )
    agent = create_agent(model=model, tools=sample_tools, middleware=[middleware])
    try:
        agent.invoke({"messages": [HumanMessage("What is the weather in Tokyo?")]})
        assert "get_weather" in recorded[0]
    finally:
        middleware.close()


def test_tools_argument_accepts_plain_callables(tmp_path, sample_tools) -> None:
    """BaseTool instances are indexed; callables are ignored, not crashed on."""

    def plain_callable(x: str) -> str:  # noqa: ARG001
        """A plain callable without the tool decorator."""
        return "ok"

    model, recorded = make_recording_model(responses=[AIMessage("All done.")])
    middleware = _make_middleware([*sample_tools, plain_callable], tmp_path, top_k=1)
    agent = create_agent(
        model=model, tools=[*sample_tools, plain_callable], middleware=[middleware]
    )
    try:
        agent.invoke({"messages": [HumanMessage("Clear the redis cache on localhost")]})
        assert "clear_redis_cache" in recorded[0]
    finally:
        middleware.close()
