from __future__ import annotations

import pytest
from comp_finder import data, pipeline
from comp_finder.engine import Engine
from helpers import synthetic_sales


@pytest.fixture(scope="session")
def sales():
    return synthetic_sales()


@pytest.fixture(scope="session")
def engine(sales):
    """Synthetic data, models fit on the first 60% of the date-sorted sales."""
    return Engine(sales, int(sales.height * 0.6))


@pytest.fixture(scope="session")
def real_sales():
    """Real King County data (from the parquet cache after the first fetch)."""
    return data.load()


@pytest.fixture(scope="session")
def result():
    """The full experiment on real data, run once (about a minute)."""
    return pipeline.run(reps=200, log=lambda *_: None)
