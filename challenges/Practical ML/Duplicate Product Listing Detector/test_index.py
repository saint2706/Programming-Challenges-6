"""Tests for index.py -- real LanceDB on tiny hand-built vectors (embedded, no server)."""

from pathlib import Path

import embed
import index
import numpy as np
import polars as pl
import pytest


def unit(rows: list[list[float]]) -> np.ndarray:
    return embed.l2_normalize(np.array(rows, dtype=np.float32))


def make(text_rows, image_rows, missing=()):
    n = len(text_rows)
    ids = [f"p{i}" for i in range(n)]
    df = pl.DataFrame(
        {
            "posting_id": ids,
            "title": [f"t{i}" for i in range(n)],
            "image": [f"{i}.jpg" for i in range(n)],
        }
    )
    return df, embed.Embeddings(ids, unit(text_rows), unit(image_rows), list(missing))


def test_build_index_stores_every_row_and_reopens(tmp_path: Path):
    df, emb = make([[1, 0], [0, 1], [1, 1]], [[1, 0], [0, 1], [1, 1]])
    table = index.build_index(tmp_path / "db", "val", df, emb)
    assert table.count_rows() == 3
    reopened = index.open_index(tmp_path / "db", "val")
    assert reopened.count_rows() == 3


def test_build_index_overwrites_previous_table(tmp_path: Path):
    df, emb = make([[1, 0], [0, 1], [1, 1]], [[1, 0], [0, 1], [1, 1]])
    index.build_index(tmp_path / "db", "val", df, emb)
    df2, emb2 = make([[1, 0], [0, 1]], [[1, 0], [0, 1]])
    table = index.build_index(tmp_path / "db", "val", df2, emb2)
    assert table.count_rows() == 2


def test_candidates_find_text_neighbors_without_self_pairs(tmp_path: Path):
    # rows 0,1 are text-near; rows 2,3 are text-near; images all orthogonal-ish noise.
    text = [[1, 0.05, 0], [1, 0.0, 0.05], [0, 1, 0.0], [0.05, 1, 0]]
    image = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    df, emb = make(text, image)
    table = index.build_index(tmp_path / "db", "t", df, emb)
    pairs = index.candidate_pairs(table, emb, k=1)
    got = {tuple(p) for p in pairs.tolist()}
    assert (0, 1) in got and (2, 3) in got
    assert all(a < b for a, b in got)
    assert all(a != b for a, b in got)


def test_candidates_include_image_only_neighbors(tmp_path: Path):
    # Text says nothing (all orthogonal); images say 0~1 and 2~3.
    text = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    image = [[1, 0.02], [1, 0.0], [0.0, 1], [0.02, 1]]
    df, emb = make(text, image)
    table = index.build_index(tmp_path / "db", "t", df, emb)
    got = {tuple(p) for p in index.candidate_pairs(table, emb, k=1).tolist()}
    assert (0, 1) in got and (2, 3) in got


def test_candidates_are_unique_and_k_larger_than_table_is_safe(tmp_path: Path):
    df, emb = make([[1, 0], [0, 1], [1, 1]], [[1, 0], [0, 1], [1, 1]])
    table = index.build_index(tmp_path / "db", "t", df, emb)
    pairs = index.candidate_pairs(table, emb, k=50)
    assert (
        len({tuple(p) for p in pairs.tolist()}) == len(pairs) == 3
    )  # all 3 unordered pairs


def test_zero_image_vector_does_not_break_search_and_keeps_text_candidates(
    tmp_path: Path,
):
    text = [[1, 0.0], [1, 0.05], [0, 1]]
    image = [[1, 0], [0, 0], [0, 1]]  # row 1's image is missing -> zero vector
    df, emb = make(text, image, missing=["p1"])
    table = index.build_index(tmp_path / "db", "t", df, emb)
    pairs = index.candidate_pairs(table, emb, k=1)
    got = {tuple(p) for p in pairs.tolist()}
    assert (0, 1) in got  # found via text
    scores_ok = np.isfinite(pairs).all()
    assert scores_ok


def test_candidate_recall_counts_retrieved_true_pairs():
    retrieved = np.array([[0, 1], [2, 3], [0, 4]])
    true = {(0, 1), (2, 3), (1, 2)}
    assert index.candidate_recall(retrieved, true) == pytest.approx(2 / 3)


def test_candidate_recall_with_no_true_pairs_is_one():
    assert index.candidate_recall(np.array([[0, 1]]), set()) == 1.0
