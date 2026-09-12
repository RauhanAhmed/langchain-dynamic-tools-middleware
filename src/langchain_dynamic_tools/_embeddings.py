"""Embedding protocols, default local embedders, and tool text rendering.

The middleware treats embedders as simple protocols so you can plug in any
embedding backend. The defaults wrap the local models that ship with zvec's
extension module: all-MiniLM-L6-v2 for dense vectors and SPLADE for sparse
vectors. Both run on your machine, no API keys, no network calls after the
first model download.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, runtime_checkable

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

SparseEncoding = Literal["query", "document"]

_INSTALL_HINT = (
    "The default embedders need the sentence-transformers package. Install it "
    "with: pip install 'langchain-dynamic-tools-middleware[local]'"
)


@runtime_checkable
class DenseEmbedder(Protocol):
    """Anything that maps text to a fixed-size float vector."""

    @property
    def dimension(self) -> int:
        """The dimension of the vectors returned by embed()."""
        ...

    def embed(self, text: str) -> list[float]:
        """Return the dense embedding for a piece of text."""
        ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Anything that maps text to a sparse {dimension index: weight} mapping."""

    def embed_document(self, text: str) -> dict[int, float]:
        """Return the sparse embedding used when indexing a document."""
        ...

    def embed_query(self, text: str) -> dict[int, float]:
        """Return the sparse embedding used when searching with a query."""
        ...


class DefaultDenseEmbedder:
    """Dense embedder backed by zvec's all-MiniLM-L6-v2 wrapper (384 dims).

    The model loads lazily on the first embed call so that constructing the
    middleware stays cheap. Works with any keyword arguments accepted by
    ``zvec.extension.DefaultLocalDenseEmbedding``.

    Example:
        ```python
        from langchain_dynamic_tools import DefaultDenseEmbedder

        embedder = DefaultDenseEmbedder()
        vector = embedder.embed("Send an email to the team")
        len(vector)  # 384
        ```
    """

    def __init__(self, **model_kwargs: Any) -> None:
        """Store model kwargs; the model itself loads on first use."""
        self._model_kwargs = model_kwargs
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        """Always 384, matching the all-MiniLM-L6-v2 model."""
        return 384

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from zvec.extension import DefaultLocalDenseEmbedding
            except ImportError as exc:  # pragma: no cover - depends on zvec build
                raise ImportError(_INSTALL_HINT) from exc
            try:
                self._model = DefaultLocalDenseEmbedding(**self._model_kwargs)
            except ImportError as exc:
                raise ImportError(_INSTALL_HINT) from exc
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed one piece of text into a 384-dimensional float vector."""
        if not isinstance(text, str):
            msg = f"Expected text to be str, got {type(text).__name__}"
            raise TypeError(msg)
        if not text.strip():
            msg = "Text cannot be empty or whitespace only"
            raise ValueError(msg)
        return [float(value) for value in self._load_model().embed(text)]


class DefaultSparseEmbedder:
    """Sparse embedder backed by zvec's SPLADE wrapper.

    SPLADE produces lexical sparse vectors that complement the dense model:
    dense catches meaning, sparse catches exact terms like function names.
    Query and document sides use different encodings, as SPLADE expects.
    Both share one underlying model through zvec's class-level cache.

    Example:
        ```python
        from langchain_dynamic_tools import DefaultSparseEmbedder

        embedder = DefaultSparseEmbedder()
        vector = embedder.embed_query("query the sql database")
        # {10412: 0.83, 8871: 1.02, ...}
        ```
    """

    def __init__(self, **model_kwargs: Any) -> None:
        """Store model kwargs; models load lazily on first use."""
        self._model_kwargs = model_kwargs
        self._document_model: Any | None = None
        self._query_model: Any | None = None

    def _build_model(self, encoding_type: SparseEncoding) -> Any:
        try:
            from zvec.extension import DefaultLocalSparseEmbedding
        except ImportError as exc:  # pragma: no cover - depends on zvec build
            raise ImportError(_INSTALL_HINT) from exc
        try:
            return DefaultLocalSparseEmbedding(encoding_type=encoding_type, **self._model_kwargs)
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

    def _load_document_model(self) -> Any:
        if self._document_model is None:
            self._document_model = self._build_model("document")
        return self._document_model

    def _load_query_model(self) -> Any:
        if self._query_model is None:
            self._query_model = self._build_model("query")
        return self._query_model

    def embed_document(self, text: str) -> dict[int, float]:
        """Embed text with the document-side encoding, used when indexing."""
        if not isinstance(text, str):
            msg = f"Expected text to be str, got {type(text).__name__}"
            raise TypeError(msg)
        if not text.strip():
            msg = "Text cannot be empty or whitespace only"
            raise ValueError(msg)
        raw = self._load_document_model().embed(text)
        return {int(index): float(weight) for index, weight in raw.items()}

    def embed_query(self, text: str) -> dict[int, float]:
        """Embed text with the query-side encoding, used when searching."""
        if not isinstance(text, str):
            msg = f"Expected text to be str, got {type(text).__name__}"
            raise TypeError(msg)
        if not text.strip():
            msg = "Text cannot be empty or whitespace only"
            raise ValueError(msg)
        raw = self._load_query_model().embed(text)
        return {int(index): float(weight) for index, weight in raw.items()}


class LangChainDenseEmbedder:
    """Dense embedder backed by any standard LangChain Embeddings object.

    LangChain's Embeddings interface (``embed_query``/``embed_documents``)
    does not match the DenseEmbedder protocol, so this adapter bridges the
    two: ``embed()`` delegates to ``embed_query()`` (tool texts are short,
    so the query/document distinction does not matter here), and
    ``dimension`` is the explicit value when given, otherwise probed once
    from the first embedding and cached. The probe costs no extra API call
    when the first use is ``embed()`` rather than ``dimension``.

    ``ToolVectorIndex`` applies this adapter automatically, so you can pass
    e.g. ``OpenAIEmbeddings`` directly as ``dense_embedder``:

    Example:
        ```python
        from langchain_openai import OpenAIEmbeddings
        from langchain_dynamic_tools import DynamicToolSelectorMiddleware

        tool_router = DynamicToolSelectorMiddleware(
            tools=all_tools,
            top_k=4,
            dense_embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
            dense_dim=1536,
        )
        ```
    """

    def __init__(self, embeddings: Embeddings, dimension: int | None = None) -> None:
        """Wrap a LangChain Embeddings object, optionally fixing its dimension."""
        if dimension is not None and dimension < 1:
            msg = f"dimension must be >= 1, got {dimension}"
            raise ValueError(msg)
        self._embeddings = embeddings
        self._explicit_dimension = dimension
        self._probed_dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Explicit dimension, or the probed embedding length (cached)."""
        if self._explicit_dimension is not None:
            return self._explicit_dimension
        if self._probed_dimension is None:
            self._probed_dimension = len(self._embeddings.embed_query("dimension probe"))
        return self._probed_dimension

    def embed(self, text: str) -> list[float]:
        """Embed one piece of text via the wrapped model's embed_query()."""
        if not isinstance(text, str):
            msg = f"Expected text to be str, got {type(text).__name__}"
            raise TypeError(msg)
        if not text.strip():
            msg = "Text cannot be empty or whitespace only"
            raise ValueError(msg)
        vector = [float(value) for value in self._embeddings.embed_query(text)]
        if self._explicit_dimension is not None:
            if len(vector) != self._explicit_dimension:
                msg = (
                    f"Embedding dimension mismatch: expected {self._explicit_dimension}, "
                    f"got {len(vector)}"
                )
                raise ValueError(msg)
        elif self._probed_dimension is None:
            self._probed_dimension = len(vector)
        return vector


def render_tool_text(tool: BaseTool) -> str:
    """Render a tool as one compact string for embedding and storage.

    The string carries the tool name, its description, and its parameters with
    types, required flags, and per-parameter descriptions. This is the exact
    lexical material a retrieval step needs to tell tools apart.

    Example:
        ```python
        from langchain_core.tools import tool
        from langchain_dynamic_tools import render_tool_text

        @tool
        def get_weather(city: str) -> str:
            "Get the current weather for a city."
            ...

        print(render_tool_text(get_weather))
        # get_weather: Get the current weather for a city.
        # Parameters: city (string, required)
        ```
    """
    description = (tool.description or "No description provided.").strip()
    parts = [f"{tool.name}: {description}"]
    parameters = _render_parameters(tool)
    if parameters:
        parts.append(f"Parameters: {parameters}")
    return " ".join(parts)


def _render_parameters(tool: BaseTool) -> str:
    """Format a tool's arguments as a short readable list.

    Handles both shapes ``tool.args`` produces across langchain-core
    versions: the flat ``{param: {type, description, default}}`` mapping of
    current versions, and the nested JSON schema with a "properties" key.
    """
    try:
        args = tool.args
    except Exception:  # noqa: BLE001 - a broken args schema must not break indexing
        logger.debug("Could not read args schema for tool %r", tool.name)
        return ""
    if not isinstance(args, dict) or not args:
        return ""

    if isinstance(args.get("properties"), dict):
        properties = args["properties"]
        required_names = set(args.get("required") or [])
        nested = True
    else:
        properties = args
        required_names = set()
        nested = False

    rendered = []
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        type_name = str(spec.get("type", "any"))
        # Flat mapping (current langchain-core): optional parameters carry a default.
        is_required = name in required_names if nested else "default" not in spec
        flag = "required" if is_required else "optional"
        piece = f"{name} ({type_name}, {flag})"
        param_description = str(spec.get("description", "")).strip()
        if param_description:
            piece = f"{piece}: {param_description}"
        rendered.append(piece)
    return "; ".join(rendered)
