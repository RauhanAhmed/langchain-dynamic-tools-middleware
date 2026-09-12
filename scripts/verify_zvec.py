"""Verify the zvec behaviors this package relies on.

Diagnostic tool for the development container: checks open/create error
handling, schema introspection, upsert/fetch/iter_docs, hybrid query with a
reranker, and deletion. Run with:

    uv run python scripts/verify_zvec.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import zvec
from zvec.extension import RrfReRanker

SCHEMA = zvec.CollectionSchema(
    name="VectorSearch",
    vectors=[
        zvec.VectorSchema(
            name="denseEmbedding",
            data_type=zvec.DataType.VECTOR_FP32,
            dimension=384,
            index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.IP),
        ),
        zvec.VectorSchema(
            name="sparseEmbedding",
            data_type=zvec.DataType.SPARSE_VECTOR_FP32,
            index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.IP),
        ),
    ],
    fields=[zvec.FieldSchema(name="text", data_type=zvec.DataType.STRING)],
)


def schema_vectors(schema: Any) -> list[Any]:
    vectors = schema.vectors
    return list(vectors) if isinstance(vectors, (list, tuple)) else [vectors]


def main() -> None:
    print("zvec version:", getattr(zvec, "__version__", "unknown"))
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "probe")

        try:
            zvec.open(path)
            print("open(missing path): NO ERROR (unexpected)")
        except Exception as exc:  # noqa: BLE001
            print(f"open(missing path) raised {type(exc).__name__}: {exc}")

        collection = zvec.create_and_open(path=path, schema=SCHEMA)
        vectors = schema_vectors(collection.schema)
        for vector in vectors:
            print(
                "schema vector:",
                vector.name,
                "dim=",
                getattr(vector, "dimension", None),
                "type=",
                getattr(vector, "data_type", None),
            )

        docs = [
            zvec.Doc(
                id="tool_a",
                vectors={
                    "denseEmbedding": [0.1] * 384,
                    "sparseEmbedding": {5: 1.0, 9: 0.5},
                },
                fields={"text": "tool_a text"},
            ),
            zvec.Doc(
                id="tool_b",
                vectors={
                    "denseEmbedding": [0.95] * 384,
                    "sparseEmbedding": {11: 1.0},
                },
                fields={"text": "tool_b text"},
            ),
        ]
        print("upsert:", collection.upsert(docs))
        collection.flush()

        fetched = collection.fetch(["tool_a", "tool_b", "missing"], include_vector=False)
        print("fetch keys:", sorted(fetched))
        print("fetch text:", fetched["tool_a"].field("text"))

        with collection.iter_docs(include_vector=False) as stored:
            print("iter_docs ids:", sorted(doc.id for doc in stored))

        results = collection.query(
            queries=[
                zvec.Query(field_name="denseEmbedding", vector=[0.95] * 384),
                zvec.Query(field_name="sparseEmbedding", vector={11: 1.0}),
            ],
            topk=2,
            reranker=RrfReRanker(rank_constant=60),
        )
        print("hybrid query result type:", type(results).__name__)
        for doc in results:
            print("  hit:", doc.id, doc.score)

        single = collection.query(
            queries=[zvec.Query(field_name="denseEmbedding", vector=[0.95] * 384)],
            topk=2,
        )
        print("single query no reranker:", [(doc.id, doc.score) for doc in single])

        print("delete tool_b:", collection.delete("tool_b"))
        collection.flush()
        with collection.iter_docs(include_vector=False) as stored:
            print("after delete:", sorted(doc.id for doc in stored))

        collection.close()
        reopened = zvec.open(path)
        print("reopen ok:", reopened.path)
        try:
            empty_sparse = reopened.query(
                queries=[zvec.Query(field_name="sparseEmbedding", vector={})],
                topk=2,
            )
            print("empty sparse query ok:", [(doc.id, doc.score) for doc in empty_sparse])
        except Exception as exc:  # noqa: BLE001
            print(f"empty sparse query raised {type(exc).__name__}: {exc}")
        reopened.close()

    print("verify_zvec: OK")


if __name__ == "__main__":
    main()
