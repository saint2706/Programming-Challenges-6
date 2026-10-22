"""The Criteo Attribution Modeling for Bidding dataset: download, slim down, and turn into journeys.

16.5M ad impressions from 30 days of Criteo traffic (Diemert, Meynet, Galland and Lefortier, AdKDD
2017). Licence: CC BY-NC-SA 4.0, so the data is fetched on demand into the gitignored ``data/`` and
never committed. An impression that led to a conversion carries that conversion's id.

**What the journeys are.** A journey is the impressions one user saw of one campaign; it is
*converted* if those impressions carry a ``conversion_id``. (Each conversion is its own journey.) 99.3%
of converting journeys involve a single campaign, because the data records conversions against the
campaign that earned them, so *campaign* cannot be the channel: the first-, last- and linear models
would agree by construction. The nine anonymised context columns ``cat1`` to ``cat9`` are the
only dimension on which one journey meets different "places", so one of them is the channel. Criteo
does not say what they mean; they are used here as an opaque placement context, nothing more.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from attribution.build import from_touches
from attribution.journeys import AttributionError, Journeys
from attribution.paths import project_root

URL = (
    "https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/"
    "criteo_attribution_dataset.tsv.gz"
)
ROWS = 16_468_027
CATS = [f"cat{i}" for i in range(1, 10)]
COLUMNS = [
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp", "conversion_id",
    "attribution", "click", "cost", *CATS,
]  # fmt: skip
PAIR_KEY = 33_000_000  # uid * PAIR_KEY + campaign is unique: both are below 3.3e7


def data_dir() -> Path:
    return project_root() / "data"


def raw_path() -> Path:
    return data_dir() / "criteo_attribution_dataset.tsv.gz"


def parquet_path() -> Path:
    return data_dir() / "criteo.parquet"


def download(dest: Path | None = None) -> Path:
    """Download the 623 MB file (needs network)."""
    dest = dest or raw_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        URL, headers={"User-Agent": "attribution-comparator/0.1"}
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        dest.open("wb") as out,
    ):
        while chunk := response.read(1 << 20):
            out.write(chunk)
    return dest


def to_parquet(src: Path | None = None, dest: Path | None = None) -> Path:
    """The slim columns only, as zstd Parquet (about 4 s and 0.4 GB)."""
    src, dest = src or raw_path(), dest or parquet_path()
    lf = pl.scan_csv(src, separator="\t", infer_schema_length=100_000).select(COLUMNS)
    lf.sink_parquet(dest, compression="zstd")
    return dest


def load_journeys(
    channel: str = "cat9",
    top: int = 11,
    *,
    parquet: Path | None = None,
    sample_users: float | None = None,
    seed: int = 0,
) -> Journeys:
    """Journeys with the ``top`` most frequent values of ``channel`` and one 'other' bucket.

    ``sample_users`` keeps that fraction of users (all of each kept user's journeys), for quick runs.
    """
    path = parquet or parquet_path()
    if not path.exists():
        raise AttributionError(f"{path} is missing; run `uv run attribute fetch`")
    if channel not in CATS:
        raise AttributionError(f"channel must be one of {', '.join(CATS)}")
    lf = pl.scan_parquet(path).select(
        "uid", "campaign", "conversion", "conversion_id", "timestamp", "conversion_timestamp",
        "click", "cost", pl.col(channel).alias("ch"),
    )  # fmt: skip
    if sample_users is not None:
        keep = (pl.col("uid").hash(seed=seed) % 1_000_000) < int(
            sample_users * 1_000_000
        )
        lf = lf.filter(keep)
    df = lf.collect()
    freq = df.group_by("ch").len().sort(["len", "ch"], descending=[True, False])
    names = [f"{channel}={v}" for v in freq["ch"].head(top).to_list()]
    mapping = {v: i for i, v in enumerate(freq["ch"].head(top).to_list())}
    other = len(names)
    if freq.height > top:
        names.append(f"{channel}=other")
    chan = (
        df["ch"]
        .replace_strict(mapping, default=other, return_dtype=pl.Int32)
        .to_numpy()
    )

    pair = df["uid"] * PAIR_KEY + df["campaign"]
    converting_pairs = (
        df.filter(pl.col("conversion") == 1)
        .select((pl.col("uid") * PAIR_KEY + pl.col("campaign")).alias("pair"))
        .unique()
    )
    is_conv = (df["conversion"] == 1).to_numpy()
    in_converting = pair.is_in(converting_pairs["pair"].implode()).to_numpy()
    use = (
        is_conv | ~in_converting
    )  # impressions of a converting pair that did not convert are ambiguous
    journey = np.where(is_conv, df["conversion_id"].to_numpy(), -pair.to_numpy() - 1)
    return from_touches(
        journey[use], df["timestamp"].to_numpy()[use].astype(float), chan[use], is_conv[use], names,
        clicked=df["click"].to_numpy()[use] == 1, cost=df["cost"].to_numpy()[use],
        conv_time=df["conversion_timestamp"].to_numpy()[use].astype(float), unit="seconds",
    )  # fmt: skip
