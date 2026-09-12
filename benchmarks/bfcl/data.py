"""Download and load the Berkeley Function Calling Leaderboard dataset.

Files come straight from the official Hugging Face dataset repository
(gorilla-llm/Berkeley-Function-Calling-Leaderboard, Apache 2.0). The dataset
card asks users not to use the datasets library, so we fetch the raw JSON
files once and cache them under benchmarks/bfcl/data/.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

BASE_URL = (
    "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
)
DATA_DIR = Path(__file__).parent / "data"

CATEGORY_FILES = {
    "simple": "BFCL_v3_simple.json",
    "live_simple": "BFCL_v3_live_simple.json",
    "live_multiple": "BFCL_v3_live_multiple.json",
    "irrelevance": "BFCL_v3_live_irrelevance.json",
    "relevance": "BFCL_v3_live_relevance.json",
}
ANSWER_FILES = {
    "simple": "possible_answer/BFCL_v3_simple.json",
    "live_simple": "possible_answer/BFCL_v3_live_simple.json",
    "live_multiple": "possible_answer/BFCL_v3_live_multiple.json",
}

TOOL_RESULT = "{'status': 'success', 'data': 'sample result'}"
"""Canned executor response. BFCL scoring only reads the model's call."""


def sanitize_name(name: str) -> str:
    """Make a BFCL function name safe for provider tool-name rules.

    BFCL names can contain dots (math.factorial); providers require
    [a-zA-Z0-9_-]. The same transform is applied to expected answers before
    comparison, so scoring stays faithful to the dataset.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"tool_{cleaned}"
    return cleaned[:64]


def _download(relative_path: str) -> Path:
    """Download one dataset file into the local cache and return its path."""
    target = DATA_DIR / relative_path
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{relative_path}"
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https URL
        target.write_bytes(response.read())
    return target


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a BFCL JSONL file (one JSON object per line)."""
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def load_category(
    category: str,
    *,
    limit: int | None = None,
    download: bool = True,
) -> list[dict[str, Any]]:
    """Load one BFCL category as a list of normalized entries.

    Each entry carries: id, question text, function schemas, and ground truth
    calls (empty for irrelevance, absent for relevance, where any call counts).
    """
    if category not in CATEGORY_FILES:
        msg = f"Unknown category {category!r}; known: {sorted(CATEGORY_FILES)}"
        raise ValueError(msg)
    if download:
        question_path = _download(CATEGORY_FILES[category])
    else:
        question_path = DATA_DIR / CATEGORY_FILES[category]
    entries = _load_jsonl(question_path)

    answers_by_id: dict[str, Any] = {}
    answer_file = ANSWER_FILES.get(category)
    if answer_file:
        answer_path = _download(answer_file) if download else DATA_DIR / answer_file
        for record in _load_jsonl(answer_path):
            answers_by_id[record["id"]] = record.get("ground_truth", [])

    normalized = []
    for entry in entries:
        question = entry["question"]
        text = question[0][0]["content"] if question and question[0] else ""
        normalized.append(
            {
                "id": entry["id"],
                "question": text,
                "functions": entry.get("function", []),
                "ground_truth": answers_by_id.get(entry["id"]),
            }
        )
    if limit is not None:
        normalized = normalized[:limit]
    return normalized


def function_pool(categories: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Collect the deduplicated pool of function schemas across categories.

    Used to pad tool sets to a target size and as a realistic corpus for the
    latency benchmark.
    """
    if categories is None:
        categories = ["simple", "live_simple", "live_multiple", "irrelevance"]
    pool: dict[str, dict[str, Any]] = {}
    for category in categories:
        for entry in load_category(category):
            for function in entry["functions"]:
                pool.setdefault(function["name"], function)
    return list(pool.values())


_TYPE_NORMALIZATION: dict[str, str] = {
    "dict": "object",
    "object": "object",
    "float": "number",
    "tuple": "array",
}
"""BFCL type names that are not valid JSON Schema types."""

_KEEP_KEYS = ("description", "required", "enum", "default")


def _normalize_schema(spec: Any) -> Any:
    """Normalize one BFCL parameter spec into valid JSON Schema.

    BFCL uses "dict" for objects, "float" for numbers, and a few non standard
    names; providers need plain JSON Schema, so normalize recursively while
    keeping descriptions, enums, defaults, and required lists intact.
    """
    if not isinstance(spec, dict):
        return spec
    normalized: dict[str, Any] = {}
    for key, value in spec.items():
        if key == "type" and isinstance(value, str):
            normalized["type"] = _TYPE_NORMALIZATION.get(value, value)
        elif key == "items":
            normalized["items"] = _normalize_schema(value)
        elif key == "properties" and isinstance(value, dict):
            normalized["properties"] = {
                name: _normalize_schema(child) for name, child in value.items()
            }
        elif key in _KEEP_KEYS:
            normalized[key] = value
    return normalized


def tool_from_schema(schema: dict[str, Any]) -> BaseTool:
    """Convert one BFCL function schema into a real LangChain tool.

    The tool carries the exact name, description, and parameter schema from
    the dataset (names sanitized for provider rules). The args schema stays a
    plain JSON Schema dict so dataset parameter names like "_class" survive
    untouched. Executing the tool returns a canned response, which is all the
    benchmark loop needs.
    """

    def _run(**kwargs: Any) -> str:
        return TOOL_RESULT

    parameters = schema.get("parameters", {}) or {}
    args_schema = _normalize_schema(parameters) if parameters else None
    if isinstance(args_schema, dict):
        args_schema.setdefault("type", "object")
    return StructuredTool.from_function(
        func=_run,
        name=sanitize_name(schema["name"]),
        description=schema.get("description", ""),
        args_schema=args_schema,
    )


def tools_from_schemas(schemas: Sequence[dict[str, Any]]) -> list[BaseTool]:
    """Convert a list of BFCL function schemas into LangChain tools."""
    return [tool_from_schema(schema) for schema in schemas]


def tools_from_schemas_dedup(schemas: Sequence[dict[str, Any]]) -> list[BaseTool]:
    """Convert schemas to tools, dropping sanitized-name collisions."""
    tools: dict[str, BaseTool] = {}
    for schema in schemas:
        tool = tool_from_schema(schema)
        tools.setdefault(tool.name, tool)
    return list(tools.values())
