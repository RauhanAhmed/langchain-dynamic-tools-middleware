"""Shared fixtures: deterministic fake embedders, sample tools, recording model.

The fake embedders are deterministic: texts that share words map to vectors
that share dimensions, so "weather" queries reliably match weather tools
without downloading any model. Tests therefore run offline in seconds.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens, with snake_case identifiers split into parts."""
    prepared = text.replace("_", " ").lower()
    return [chunk for chunk in prepared.split() if chunk]


def _stable_index(token: str, modulo: int) -> int:
    """Map a token to a stable index with md5, unaffected by hash randomization."""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()  # noqa: S324
    return int(digest, 16) % modulo


class FakeDenseEmbedder:
    """Deterministic dense embedder where shared words produce similar vectors."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _tokens(text)
        if not tokens:
            vector[0] = 1.0
            return vector
        for token in tokens:
            vector[_stable_index(token, self.dimension)] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class CountingDenseEmbedder(FakeDenseEmbedder):
    """FakeDenseEmbedder that counts embed calls for sync assertions."""

    def __init__(self, dimension: int = 384) -> None:
        super().__init__(dimension)
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return super().embed(text)


class FakeSparseEmbedder:
    """Deterministic sparse embedder keyed on stable token hashes."""

    def embed_document(self, text: str) -> dict[int, float]:
        return self._embed(text)

    def embed_query(self, text: str) -> dict[int, float]:
        return self._embed(text)

    def _embed(self, text: str) -> dict[int, float]:
        weights: dict[int, float] = {}
        for token in _tokens(text):
            index = _stable_index(token, 30_000)
            weights[index] = weights.get(index, 0.0) + 1.0
        return dict(sorted(weights.items()))


class CountingSparseEmbedder(FakeSparseEmbedder):
    """FakeSparseEmbedder that counts embed calls for sync assertions."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_document(self, text: str) -> dict[int, float]:
        self.calls += 1
        return super().embed_document(text)

    def embed_query(self, text: str) -> dict[int, float]:
        self.calls += 1
        return super().embed_query(text)


class ExplodingSparseEmbedder(FakeSparseEmbedder):
    """Sparse embedder whose query side always fails, for fallback tests."""

    def embed_query(self, text: str) -> dict[int, float]:
        msg = "simulated sparse embedder outage"
        raise RuntimeError(msg)


@tool
def get_weather(city: str) -> str:
    """Get the current weather conditions and forecast for a city."""
    return f"Sunny, 22C in {city}"


@tool
def query_sql_database(sql_query: str) -> str:
    """Run a SQL query against the sales database and return matching rows."""
    return "3 rows"


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email message to a recipient with a subject and a body."""
    return "email sent"


@tool
def git_push(branch: str, commit_message: str) -> str:
    """Push commits on a branch to the git repository with a commit message."""
    return "pushed"


@tool
def create_calendar_event(title: str, start_time: str) -> str:
    """Create a calendar event with a title and a start time."""
    return "event created"


@tool
def post_slack_message(channel: str, message: str) -> str:
    """Post a message to a Slack channel."""
    return "posted"


@tool
def search_files(directory: str, pattern: str) -> str:
    """Search for files in a directory matching a name pattern."""
    return "2 matches"


@tool
def make_http_request(url: str, method: str) -> str:
    """Make an HTTP request to a URL with a method like GET or POST."""
    return "200 OK"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression, like arithmetic or percentages."""
    return "42"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert a money amount from one currency to another currency."""
    return "82.50"


@tool
def translate_text(text: str, target_language: str) -> str:
    """Translate text into a target language."""
    return "bonjour"


@tool
def resize_image(file_path: str, width: int, height: int) -> str:
    """Resize an image file to a new width and height in pixels."""
    return "resized"


@tool
def extract_pdf_text(file_path: str) -> str:
    """Extract the text content from a PDF file."""
    return "pdf text"


@tool
def analyze_csv(file_path: str, column: str) -> str:
    """Analyze a column of a CSV file and return summary statistics."""
    return "mean 12.3"


@tool
def create_jira_ticket(project: str, summary: str) -> str:
    """Create a Jira ticket in a project with a summary."""
    return "JIRA-42"


@tool
def deploy_service(service_name: str, environment: str) -> str:
    """Deploy a service to an environment like staging or production."""
    return "deployed"


@tool
def restart_kubernetes_pod(namespace: str, pod_name: str) -> str:
    """Restart a pod in a Kubernetes namespace."""
    return "restarted"


@tool
def clear_redis_cache(host: str) -> str:
    """Clear the Redis cache on a host."""
    return "cleared"


@tool
def upload_to_s3(file_path: str, bucket: str) -> str:
    """Upload a local file to an S3 storage bucket."""
    return "uploaded"


@tool
def search_logs(service_name: str, minutes: int) -> str:
    """Search the application logs of a service for the last N minutes."""
    return "5 hits"


@tool
def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight from an origin airport to a destination on a date."""
    return "booked"


@tool
def order_pizza(toppings: str, size: str) -> str:
    """Order a pizza with toppings and a size."""
    return "ordered"


SAMPLE_TOOLS: list[BaseTool] = [
    get_weather,
    query_sql_database,
    send_email,
    git_push,
    create_calendar_event,
    post_slack_message,
    search_files,
    make_http_request,
    calculate,
    convert_currency,
    translate_text,
    resize_image,
    extract_pdf_text,
    analyze_csv,
    create_jira_ticket,
    deploy_service,
    restart_kubernetes_pod,
    clear_redis_cache,
    upload_to_s3,
    search_logs,
    book_flight,
    order_pizza,
]


@pytest.fixture
def sample_tools() -> list[BaseTool]:
    """A fresh list of the sample tools."""
    return list(SAMPLE_TOOLS)


def _tool_names(tools: Sequence[Any]) -> list[str]:
    """Extract display names from a mixed list of tools and provider dicts."""
    names: list[str] = []
    for item in tools:
        if isinstance(item, BaseTool):
            names.append(item.name)
        elif isinstance(item, dict):
            label = item.get("name", item.get("type", "dict-tool"))
            names.append(str(label))
    return names


def make_recording_model(
    responses: Sequence[AIMessage],
) -> tuple[BaseChatModel, list[list[str]]]:
    """Build a chat model that records bound tool names and returns scripted messages.

    Returns the model and the list it appends one entry to per bind_tools
    call, each entry holding the names of the tools bound for that call.
    """
    recorded: list[list[str]] = []
    state = {"index": 0}

    class _RecordingChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "recording-test-model"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            recorded.append(_tool_names(tools))
            return self

        def _generate(
            self,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = responses[min(state["index"], len(responses) - 1)]
            state["index"] += 1
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])

    return _RecordingChatModel(), recorded
