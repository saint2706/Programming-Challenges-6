"""LanceDB vector index and candidate-pair retrieval.

One LanceDB table per split holds both embeddings as separate vector columns
(``text_vector``, ``image_vector``) next to the row index, so a single embedded
store answers nearest-neighbour queries in either modality. Retrieval is
deliberately cheap and recall-oriented -- top-k per modality, unioned; the
precise decision is made afterwards by the tuned fused score in ``match.py``.
"""

from pathlib import Path

import lancedb
import numpy as np
import polars as pl
import pyarrow as pa
from embed import Embeddings

QUERY_CHUNK = 256


def _vector_column(matrix: np.ndarray) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(matrix.astype(np.float32).ravel()), matrix.shape[1]
    )


def build_index(
    db_dir: Path, name: str, df: pl.DataFrame, emb: Embeddings
) -> "lancedb.table.Table":
    """(Re)create table `name`; the ``row`` column is the listing's position in `df`."""
    db = lancedb.connect(str(db_dir))
    table = pa.table(
        {
            "row": pa.array(range(df.height), pa.int32()),
            "posting_id": pa.array(df["posting_id"].to_list()),
            "title": pa.array(df["title"].to_list()),
            "image": pa.array(df["image"].to_list()),
            "text_vector": _vector_column(emb.text),
            "image_vector": _vector_column(emb.image),
        }
    )
    return db.create_table(name, data=table, mode="overwrite")


def open_index(db_dir: Path, name: str) -> "lancedb.table.Table":
    return lancedb.connect(str(db_dir)).open_table(name)


def _neighbors(
    table, queries: np.ndarray, column: str, k: int
) -> list[tuple[int, int]]:
    """(query_row, neighbour_row) for each query's top-k cosine neighbours, excluding itself."""
    out: list[tuple[int, int]] = []
    for start in range(0, len(queries), QUERY_CHUNK):
        chunk = queries[start : start + QUERY_CHUNK]
        res = (
            table.search(chunk, vector_column_name=column)
            .metric("cosine")
            .limit(
                k + 1
            )  # the query row itself comes back as its own nearest neighbour
            .select(["row"])
            .to_arrow()
            .to_pydict()
        )
        for q, r in zip(res["query_index"], res["row"], strict=True):
            query_row = start + q
            if r != query_row:
                out.append((query_row, r))
    return out


def candidate_pairs(table, emb: Embeddings, k: int = 10) -> np.ndarray:
    """Union of top-k neighbours in text space and in image space, as unique ``[a, b]`` with a < b.

    Rows with an all-zero vector (image unavailable) are skipped for that
    modality: cosine against a zero vector is undefined, and such a row should
    neither query nor rank by a similarity that does not exist.
    """
    pairs: set[tuple[int, int]] = set()
    for column, matrix in (("text_vector", emb.text), ("image_vector", emb.image)):
        usable = np.flatnonzero(np.linalg.norm(matrix, axis=1) > 0)
        if len(usable) == 0:
            continue
        for q, r in _neighbors(table, matrix[usable], column, k):
            a, b = int(usable[q]), int(r)
            pairs.add((min(a, b), max(a, b)))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(sorted(pairs), dtype=np.int64)


def candidate_recall(retrieved: np.ndarray, true_pairs: set[tuple[int, int]]) -> float:
    """Fraction of true duplicate pairs that candidate retrieval surfaced at all."""
    if not true_pairs:
        return 1.0
    got = {(int(a), int(b)) for a, b in retrieved}
    return len(got & true_pairs) / len(true_pairs)
