"""Shared fixtures."""

import json

import polars as pl
import pytest
from helpers import make_pool

CLASSES = [f"c{i}" for i in range(5)]


@pytest.fixture(scope="session")
def pool():
    """``(X, y)``: 600 items, 10 imbalanced classes, cleanly separable."""
    return make_pool()


@pytest.fixture
def banking_dir(tmp_path):
    """A tiny fake Banking77 on disk: same files and columns as the real one."""
    rows = [(f"class {c} sample {j}", CLASSES[c]) for c in range(5) for j in range(30)]
    rows += [("class 0 sample 0", "c0")] * 2  # exact duplicates
    rows.append(("multi\nline text", "c1"))
    pl.DataFrame(rows, schema=["text", "category"], orient="row").write_csv(
        tmp_path / "train.csv"
    )
    test = [(f"test text {j}", CLASSES[j % 5]) for j in range(20)]
    test += [("class 2 sample 4", "c2"), ("class 3 sample 9", "c3")]  # also in train
    pl.DataFrame(test, schema=["text", "category"], orient="row").write_csv(
        tmp_path / "test.csv"
    )
    (tmp_path / "categories.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    return tmp_path
