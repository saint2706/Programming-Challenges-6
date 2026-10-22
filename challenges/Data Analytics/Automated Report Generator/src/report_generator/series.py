"""Turning a table into a regular time series: pick a grain, bucket, trim partial edges, fill gaps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from report_generator.profile import Profile

GRAINS = ("day", "week", "month", "quarter", "year")
PERIODS = {"day": 7, "week": 52, "month": 12, "quarter": 4, "year": None}
ADJECTIVE = {
    "day": "daily",
    "week": "weekly",
    "month": "monthly",
    "quarter": "quarterly",
    "year": "yearly",
}
UNIT_DAYS = {"day": 1, "week": 7, "month": 30.4375, "quarter": 91.3125, "year": 365.25}
MAX_BUCKETS = 160
MIN_BUCKETS = 12
EDGE_TOLERANCE = (
    0.15  # an edge bucket missing more than this share of its span is dropped
)
ROWS = "__rows__"


@dataclass
class Series:
    key: str
    label: str
    grain: str
    agg: str  # sum, mean or count
    dates: np.ndarray  # datetime64[D], the start of each bucket
    values: np.ndarray
    n_rows: np.ndarray
    period: int | None
    dropped: list[str] = field(default_factory=list)
    filled: int = 0

    @property
    def phrase(self) -> str:
        """'weekly total rentals', 'monthly average Sale_Price', 'monthly number of sales'."""
        adj = ADJECTIVE[self.grain]
        if self.key == ROWS:
            return f"{adj} {self.label}"
        kind = {"sum": "total", "mean": "average", "count": "count of"}[self.agg]
        return f"{adj} {kind} {self.label}"

    def __len__(self) -> int:
        return int(self.values.size)


def choose_grain(first: np.datetime64, last: np.datetime64, native: str) -> str | None:
    """The finest grain (not finer than the data) that gives a readable number of buckets."""
    days = float((last - first) / np.timedelta64(1, "D")) + 1
    start = GRAINS.index("month" if native == "month" else "day")
    for grain in GRAINS[start:]:
        n = days / UNIT_DAYS[grain]
        if MIN_BUCKETS <= n <= MAX_BUCKETS:
            return grain
        if n < MIN_BUCKETS:
            return None
    return None


def _bucket_expr(grain: str, start) -> pl.Expr:
    d = pl.col("_d")
    if grain == "day":
        return d
    if (
        grain == "week"
    ):  # weeks anchored on the first observed day, so the first one is whole
        return d - pl.duration(days=(d - start).dt.total_days() % 7)
    return d.dt.truncate({"month": "1mo", "quarter": "3mo", "year": "1y"}[grain])


def _next_bucket(start: np.datetime64, grain: str) -> np.datetime64:
    s = np.datetime64(start, "D")
    if grain == "day":
        return s + np.timedelta64(1, "D")
    if grain == "week":
        return s + np.timedelta64(7, "D")
    months = {"month": 1, "quarter": 3, "year": 12}[grain]
    m = s.astype("datetime64[M]") + np.timedelta64(months, "M")
    return m.astype("datetime64[D]")


def build_series(profile: Profile, key: str, grain: str | None = None) -> Series | None:
    """The series for a measure (``key`` = column name) or for the row count (``ROWS``)."""
    if profile.time is None:
        return None
    frame = profile.frame
    col = None if key == ROWS else profile.columns[key]
    value = pl.lit(1) if col is None else pl.col(key)
    d = frame.select(
        pl.col(profile.time).cast(pl.Date).alias("_d"), value.alias("_v")
    ).drop_nulls("_d")
    if d.is_empty():
        return None
    first, last = d["_d"].min(), d["_d"].max()
    first64, last64 = np.datetime64(first), np.datetime64(last)
    grain = grain or choose_grain(first64, last64, profile.native_grain or "day")
    if grain is None:
        return None
    agg = "count" if col is None else col.agg
    expr = {"count": pl.len(), "sum": pl.col("_v").sum(), "mean": pl.col("_v").mean()}[
        agg
    ]
    b = d.with_columns(_bucket_expr(grain, first).alias("_b"))
    g = (
        b.group_by("_b")
        .agg(
            expr.alias("v"),
            pl.len().alias("n"),
            pl.col("_d").min().alias("lo"),
            pl.col("_d").max().alias("hi"),
            pl.col("_v").is_not_null().sum().alias("nn"),
        )
        .sort("_b")
    )
    if agg in ("sum", "mean"):  # a bucket whose every value is missing has no value
        g = g.with_columns(
            pl.when(pl.col("nn") > 0).then(pl.col("v")).otherwise(None).alias("v")
        )
    # a regular grid, so a quiet bucket is a zero (or a gap to fill), not a missing row
    starts = [np.datetime64(g["_b"][0], "D")]
    while _next_bucket(starts[-1], grain) <= np.datetime64(g["_b"][-1], "D"):
        starts.append(_next_bucket(starts[-1], grain))
    grid = pl.DataFrame({"_b": np.array(starts, dtype="datetime64[D]")}).with_columns(
        pl.col("_b").cast(pl.Date)
    )
    g = grid.join(g, on="_b", how="left").sort("_b")
    dates = g["_b"].to_numpy().astype("datetime64[D]")
    n_rows = g["n"].fill_null(0).to_numpy().astype(int)
    dropped: list[str] = []
    keep = np.ones(dates.size, dtype=bool)
    native_month = profile.native_grain == "month"
    length = UNIT_DAYS[grain]
    first_obs = np.datetime64(g["lo"][0], "D") if g["lo"][0] is not None else dates[0]
    last_obs = np.datetime64(g["hi"][-1], "D") if g["hi"][-1] is not None else dates[-1]
    if native_month:  # month-labelled data: a month's label stands for the whole month
        last_obs = _next_bucket(
            last_obs.astype("datetime64[M]").astype("datetime64[D]"), "month"
        ) - np.timedelta64(1, "D")
        first_obs = first_obs.astype("datetime64[M]").astype("datetime64[D]")
    if grain != "day":
        lead = float((first_obs - dates[0]) / np.timedelta64(1, "D")) / length
        end = _next_bucket(dates[-1], grain) - np.timedelta64(1, "D")
        tail = float((end - last_obs) / np.timedelta64(1, "D")) / length
        if lead > EDGE_TOLERANCE:
            keep[0] = False
            dropped.append(f"first {grain} ({dates[0]}) starts late")
        if tail > EDGE_TOLERANCE:
            keep[-1] = False
            dropped.append(
                f"last {grain} ({dates[-1]}) is incomplete (data ends {last_obs})"
            )
    if profile.event_table and n_rows.size >= 12:
        # Row counts give away a bucket that was cut short even when its dates look complete
        # (month-labelled data cannot say how far into the month it runs).
        tail = np.median(n_rows[-7:-1])
        head = np.median(n_rows[1:7])
        if keep[-1] and tail > 0 and n_rows[-1] < 0.3 * tail:
            keep[-1] = False
            dropped.append(
                f"last {grain} ({dates[-1]}) has {n_rows[-1]} rows against a typical {tail:.0f}"
            )
        if keep[0] and head > 0 and n_rows[0] < 0.3 * head:
            keep[0] = False
            dropped.append(
                f"first {grain} ({dates[0]}) has {n_rows[0]} rows against a typical {head:.0f}"
            )
    values = g["v"].to_numpy().astype(float)
    if agg in ("sum", "count"):
        values = np.where(np.isnan(values), 0.0, values)
    values, dates, n_rows = values[keep], dates[keep], n_rows[keep]
    filled = 0
    if agg == "mean" and np.isnan(values).any():
        missing = np.isnan(values)
        if missing.mean() > 0.1:
            return None
        idx = np.arange(values.size)
        values[missing] = np.interp(idx[missing], idx[~missing], values[~missing])
        filled = int(missing.sum())
    if values.size < MIN_BUCKETS:
        return None
    label = profile.row_label if col is None else key
    return Series(
        key, label, grain, agg, dates, values, n_rows, PERIODS[grain], dropped, filled
    )
