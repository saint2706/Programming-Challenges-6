"""LanceDB table of job vectors for top-k dense retrieval.

Same store the Duplicate Product Listing Detector uses. The app retrieves a
candidate set here and then lets the sparse scorers re-rank and explain it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import lancedb
import numpy as np
import pyarrow as pa

TABLE = "jobs"
QUERY_CHUNK = 256


class JobIndex:
    def __init__(self, table):
        self.table = table

    @classmethod
    def build(cls, db_dir: Path, job_ids: list[str], vectors: np.ndarray) -> Self:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        data = pa.table(
            {
                "job_id": pa.array(job_ids),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(vectors.ravel()), vectors.shape[1]
                ),
            }
        )
        db = lancedb.connect(str(db_dir))
        return cls(db.create_table(TABLE, data=data, mode="overwrite"))

    @classmethod
    def open(cls, db_dir: Path) -> Self:
        return cls(lancedb.connect(str(db_dir)).open_table(TABLE))

    def search(self, queries: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        """Per query: the ``k`` nearest ``(job_id, cosine similarity)``, best first."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        out: list[list[tuple[str, float]]] = [[] for _ in range(len(queries))]
        for start in range(0, len(queries), QUERY_CHUNK):
            chunk = queries[start : start + QUERY_CHUNK]
            # lancedb only adds ``query_index`` for a multi-vector search
            single = len(chunk) == 1
            res = (
                self.table.search(
                    chunk[0] if single else chunk, vector_column_name="vector"
                )
                .metric("cosine")
                .limit(k)
                .select(["job_id"])
                .to_arrow()
                .to_pydict()
            )
            query_index = [0] * len(res["job_id"]) if single else res["query_index"]
            for q, job_id, dist in zip(
                query_index, res["job_id"], res["_distance"], strict=True
            ):
                out[start + q].append((job_id, 1.0 - float(dist)))
        for hits in out:
            hits.sort(key=lambda h: -h[1])
        return out
