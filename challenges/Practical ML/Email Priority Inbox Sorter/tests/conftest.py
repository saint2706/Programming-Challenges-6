"""Shared fixture: a tiny trained pipeline on the synthetic CSV, built once."""

import polars as pl
import pytest
from helpers import synthetic_csv
from inbox_sorter.pipeline import load_artifacts, run_all


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trained")
    csv = synthetic_csv(tmp)
    run_all(tmp, tmp / "results", ["aa-b", "cc-d"], seed=0, csv=csv)
    return load_artifacts(tmp), pl.read_parquet(tmp / "dataset.parquet")
