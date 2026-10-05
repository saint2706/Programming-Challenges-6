"""The electricity-market stream, split in time order into train / reference / live.

Source: OpenML ``electricity`` (data_id 151), NSW electricity market, 45,312
half-hourly rows 1996-1998, label = price UP vs a moving average. It is the
standard concept-drift benchmark, so the live part genuinely drifts. Downloaded
once into ``data/`` (gitignored) and cached as parquet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OPENML_ID = 151
PARQUET = "electricity.parquet"

# ``date`` (a normalized day index) is deliberately not a feature: a tree model
# would only learn to extrapolate on it.
CATEGORICAL = ["day"]
NUMERIC = ["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
FEATURES = [*CATEGORICAL, *NUMERIC]
SPLIT_FRACS = (0.3, 0.1, 0.6)


def fetch(data_dir: Path = DATA_DIR) -> Path:
    """Download the dataset from OpenML (no token needed) and cache it as parquet."""
    from sklearn.datasets import fetch_openml

    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / PARQUET
    if target.exists():
        return target
    raw = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
    df = pl.DataFrame(
        {
            "day": raw["day"].astype(int).to_numpy(),
            **{c: raw[c].astype(float).to_numpy() for c in NUMERIC},
            "label": (raw["class"].astype(str) == "UP").astype(int).to_numpy(),
        }
    )
    df.write_parquet(target)
    return target


def load(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).select([*FEATURES, "label"])


def split(
    df: pl.DataFrame, fracs: tuple[float, float, float] = SPLIT_FRACS
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Contiguous train / reference / live slices, strictly in row (time) order."""
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fracs}")
    n = df.height
    a = round(n * fracs[0])
    b = a + round(n * fracs[1])
    return df[:a], df[a:b], df[b:]
