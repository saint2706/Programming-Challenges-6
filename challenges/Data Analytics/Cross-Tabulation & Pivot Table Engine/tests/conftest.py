"""Fixtures. The engine library is built with cargo on first use (the CI Python job has cargo)."""

from pathlib import Path

import pytest
from pivot_tool.engine import Table
from pivot_tool.paths import ENV_VAR

ROOT = Path(__file__).resolve().parents[1]
ADULT = ROOT / "sample_data" / "adult.csv.gz"
UCB = ROOT / "sample_data" / "ucb_admissions.csv"


@pytest.fixture(autouse=True)
def _home(monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(ROOT))


@pytest.fixture(scope="session")
def adult():
    with Table.from_path(ADULT) as table:
        yield table


@pytest.fixture(scope="session")
def ucb():
    with Table.from_path(UCB) as table:
        yield table


@pytest.fixture(scope="session")
def adult_frame():
    import polars as pl

    return pl.read_csv(ADULT)


def cells(result: dict, measure: int = 0) -> dict:
    """A pivot result as {(row key, column key): value}; a key is (depth, labels of the first depth fields)."""

    def key(header: dict) -> tuple:
        d = header["depth"]
        return (d, tuple(header["labels"][:d]))

    return {
        (key(r), key(c)): result["cells"][i][j][measure]
        for i, r in enumerate(result["rows"])
        for j, c in enumerate(result["cols"])
    }
