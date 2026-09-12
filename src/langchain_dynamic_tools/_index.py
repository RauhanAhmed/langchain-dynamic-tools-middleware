"""The zvec-backed index that stores tool embeddings and answers hybrid searches.

Every tool becomes one zvec document with a dense vector, a sparse vector, and
the rendered tool text. A search embeds the query both ways, runs both vectors
through a single zvec query, and fuses the two ranked lists with reciprocal
rank fusion. zvec is an in-process engine, so the whole index lives on local
disk and searches take milliseconds with no server to run.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any

import zvec

from langchain_dynamic_tools._embeddings import (
    DefaultDenseEmbedder,
    DefaultSparseEmbedder,
    DenseEmbedder,
    LangChainDenseEmbedder,
    SparseEmbedder,
    render_tool_text,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.embeddings import Embeddings
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

DENSE_FIELD = "denseEmbedding"
SPARSE_FIELD = "sparseEmbedding"
TEXT_FIELD = "text"
DEFAULT_PATH = ".dynamicToolsMiddleware"
DEFAULT_COLLECTION_NAME = "VectorSearch"


def _default_reranker() -> Any:
    """Build the default reciprocal rank fusion reranker from zvec."""
    from zvec.extension import RrfReRanker

    return RrfReRanker(rank_constant=60)


def _schema_vectors(schema: Any) -> list[Any]:
    """Return the vector schemas of a collection schema as a list."""
    vectors = schema.vectors
    if isinstance(vectors, (list, tuple)):
        return list(vectors)
    return [vectors]


def _stored_dense_dimension(collection: Any) -> int | None:
    """Read the dense vector dimension from an opened collection, if present."""
    for vector in _schema_vectors(collection.schema):
        if vector.name == DENSE_FIELD:
            dimension = getattr(vector, "dimension", None)
            if dimension:
                return int(dimension)
    return None


class ToolVectorIndex:
    """A local zvec collection holding hybrid embeddings of tool definitions.

    The index is the storage and retrieval half of the middleware. Give it
    tools, it embeds and upserts them once; give it a query, it returns the
    most similar tool names. Syncs are incremental: unchanged tools are never
    re-embedded, so restarting an application with the same tool set costs
    nothing beyond opening the collection.

    Example:
        ```python
        from langchain_dynamic_tools import ToolVectorIndex

        index = ToolVectorIndex(tools=[get_weather, send_email])
        index.search("what is the weather in tokyo", top_k=2)
        # [("get_weather", 0.93), ("convert_currency", 0.12)]
        index.close()
        ```
    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        path: str = DEFAULT_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        dense_embedder: DenseEmbedder | Embeddings | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        dense_dim: int | None = None,
        reranker: Any | None = None,
    ) -> None:
        """Open (or create) the collection and index the given tools.

        Args:
            tools: Tools to index right away. May be empty for an index you
                fill in later with sync().
            path: Filesystem path of the zvec collection.
            collection_name: Name recorded in the collection schema.
            dense_embedder: Maps text to dense vectors. Defaults to the local
                all-MiniLM-L6-v2 model (384 dims). A standard LangChain
                Embeddings object (e.g. OpenAIEmbeddings) is adapted
                automatically.
            sparse_embedder: Maps text to sparse vectors. Defaults to the
                local SPLADE model.
            dense_dim: Dense vector dimension. Defaults to the embedder's
                dimension. If an existing collection was built with a
                different dimension, it is rebuilt.
            reranker: zvec reranker that fuses the dense and sparse result
                lists. Defaults to RrfReRanker(rank_constant=60).

        Raises:
            ValueError: If path is empty or dense_dim is less than 1.
            ValueError: If tools contain duplicate names or empty renders.
        """
        if not isinstance(path, str) or not path.strip():
            msg = "path must be a non-empty string"
            raise ValueError(msg)
        dense: DenseEmbedder
        if dense_embedder is None:
            dense = DefaultDenseEmbedder()
        elif isinstance(dense_embedder, DenseEmbedder):
            dense = dense_embedder
        else:
            from langchain_core.embeddings import Embeddings

            if not isinstance(dense_embedder, Embeddings):
                msg = (
                    "dense_embedder must expose .dimension and .embed(text), or be a "
                    f"standard LangChain Embeddings object; got {type(dense_embedder).__name__}"
                )
                raise TypeError(msg)
            # Accept standard LangChain embeddings (e.g. OpenAIEmbeddings)
            # directly; an explicit dense_dim doubles as the adapter's
            # dimension so no probing API call is needed.
            dense = LangChainDenseEmbedder(dense_embedder, dimension=dense_dim)
        self._dense = dense
        self._sparse = sparse_embedder if sparse_embedder is not None else DefaultSparseEmbedder()
        self._dense_dim = dense_dim if dense_dim is not None else self._dense.dimension
        if self._dense_dim < 1:
            msg = f"dense_dim must be >= 1, got {self._dense_dim}"
            raise ValueError(msg)
        self._path = path
        self._collection_name = collection_name
        self._reranker = reranker if reranker is not None else _default_reranker()
        self._lock = threading.Lock()
        self._indexed: dict[str, str] = {}
        self._collection = self._open_or_create()
        if tools:
            self.sync(tools)

    @property
    def path(self) -> str:
        """Filesystem path of the underlying zvec collection."""
        return self._path

    @property
    def indexed_names(self) -> set[str]:
        """Names of the tools this instance has synced into the collection."""
        return set(self._indexed)

    def _build_schema(self) -> Any:
        """Build the collection schema: one dense and one sparse vector field."""
        return zvec.CollectionSchema(
            name=self._collection_name,
            vectors=[
                zvec.VectorSchema(
                    name=DENSE_FIELD,
                    data_type=zvec.DataType.VECTOR_FP32,
                    dimension=self._dense_dim,
                    index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.IP),
                ),
                zvec.VectorSchema(
                    name=SPARSE_FIELD,
                    data_type=zvec.DataType.SPARSE_VECTOR_FP32,
                    index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.IP),
                ),
            ],
            fields=[zvec.FieldSchema(name=TEXT_FIELD, data_type=zvec.DataType.STRING)],
        )

    def _open_or_create(self) -> Any:
        """Open the collection at path, creating it if needed.

        An existing collection built with a different dense dimension is
        destroyed and rebuilt, since its vectors cannot be reused. A path
        that exists but holds no usable collection raises a clear error
        rather than deleting anything, except for one case: an empty
        leftover directory holds no data, so the stub is removed and the
        collection created fresh.
        """
        schema = self._build_schema()
        collection: Any | None = None
        if os.path.exists(self._path):
            try:
                collection = zvec.open(self._path)
            except Exception as open_error:
                if os.path.isdir(self._path) and not os.listdir(self._path):
                    os.rmdir(self._path)
                    return zvec.create_and_open(path=self._path, schema=schema)
                try:
                    collection = zvec.create_and_open(path=self._path, schema=schema)
                except Exception:
                    msg = (
                        f"Path {self._path!r} exists but is not a usable zvec "
                        f"collection (open failed with: {open_error}). Remove the "
                        f"path or choose a different one."
                    )
                    raise RuntimeError(msg) from open_error
        if collection is None:
            return zvec.create_and_open(path=self._path, schema=schema)
        stored_dimension = _stored_dense_dimension(collection)
        if stored_dimension is not None and stored_dimension != self._dense_dim:
            logger.info(
                "Collection at %r has dense dimension %d but the embedder produces %d; "
                "rebuilding the collection",
                self._path,
                stored_dimension,
                self._dense_dim,
            )
            collection.destroy()
            return zvec.create_and_open(path=self._path, schema=schema)
        return collection

    def _stored_texts(self) -> dict[str, str]:
        """Read the id and text of every document currently stored."""
        texts: dict[str, str] = {}
        with self._collection.iter_docs(include_vector=False) as docs:
            for doc in docs:
                texts[doc.id] = doc.field(TEXT_FIELD) or ""
        return texts

    def _make_doc(self, name: str, text: str) -> Any:
        """Build one zvec document for a tool from both embeddings."""
        return zvec.Doc(
            id=name,
            vectors={
                DENSE_FIELD: self._dense.embed(text),
                SPARSE_FIELD: self._sparse.embed_document(text),
            },
            fields={TEXT_FIELD: text},
        )

    def sync(self, tools: Sequence[BaseTool], *, prune: bool = True) -> None:
        """Bring the collection in line with the given tools.

        New and changed tools are embedded and upserted. With prune=True,
        stored documents whose tool is absent from the list are deleted.
        Unchanged tools are left alone, so re-syncing the same tool set
        embeds nothing.

        Args:
            tools: The tools the collection should hold afterwards.
            prune: Also delete stored tools missing from the list. Set to
                False when adding tools found in a request on the fly.

        Raises:
            ValueError: If two tools share a name or one has an empty name.
        """
        wanted: dict[str, str] = {}
        for tool in tools:
            if not tool.name.strip():
                msg = f"Tool {tool.name!r} has an empty name after stripping whitespace"
                raise ValueError(msg)
            text = render_tool_text(tool)
            if tool.name in wanted:
                msg = f"Duplicate tool name {tool.name!r}; tool names must be unique"
                raise ValueError(msg)
            wanted[tool.name] = text

        with self._lock:
            collection = self._collection
            if collection is None:
                msg = "The index is closed; create a new ToolVectorIndex to reuse it"
                raise RuntimeError(msg)
            existing = self._stored_texts()
            to_delete = [name for name in existing if prune and name not in wanted]
            to_upsert = [
                (name, text)
                for name, text in wanted.items()
                if name not in existing or existing[name] != text
            ]
            if to_delete:
                collection.delete(to_delete)
                for name in to_delete:
                    self._indexed.pop(name, None)
            if to_upsert:
                collection.upsert([self._make_doc(name, text) for name, text in to_upsert])
                for name, text in to_upsert:
                    self._indexed[name] = text
            collection.flush()
            if to_upsert or to_delete:
                logger.debug(
                    "Synced tool index: %d upsert(s), %d deletion(s)",
                    len(to_upsert),
                    len(to_delete),
                )

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return the top_k most similar tools as (name, score) pairs, best first.

        Args:
            query: Natural language text to search with.
            top_k: Maximum number of tools to return.

        Returns:
            A list of (tool name, fused score) pairs, highest score first.

        Raises:
            TypeError: If query is not a string.
            ValueError: If top_k is less than 1.
            RuntimeError: If the index has been closed.
        """
        if not isinstance(query, str):
            msg = f"Expected query to be str, got {type(query).__name__}"
            raise TypeError(msg)
        if top_k < 1:
            msg = f"top_k must be >= 1, got {top_k}"
            raise ValueError(msg)
        query = query.strip()
        if not query:
            return []
        collection = self._collection
        if collection is None:
            msg = "The index is closed; create a new ToolVectorIndex to reuse it"
            raise RuntimeError(msg)
        dense_vector = self._dense.embed(query)
        sparse_vector = self._sparse.embed_query(query)
        queries = [zvec.Query(field_name=DENSE_FIELD, vector=dense_vector)]
        if sparse_vector:
            # zvec rejects empty query vectors; skip the sparse clause instead.
            queries.append(zvec.Query(field_name=SPARSE_FIELD, vector=sparse_vector))
        results = collection.query(
            queries=queries,
            topk=top_k,
            reranker=self._reranker,
        )
        hits: list[tuple[str, float]] = []
        for doc in results:
            score = float(doc.score) if doc.score is not None else 0.0
            hits.append((doc.id, score))
        return hits

    def close(self) -> None:
        """Flush and close the collection, releasing its file lock.

        Closing an already closed index is a no-op. Do not close while
        searches from other threads are still running.
        """
        with self._lock:
            if self._collection is not None:
                self._collection.close()
                self._collection = None

    def __enter__(self) -> ToolVectorIndex:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
