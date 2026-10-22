"""Reading a table's structure: which column is the clock, the measures, the groupings.

The report engine never asks the user what a column means (though every guess can be overridden);
this module makes the guesses and keeps them inspectable. Roles:

``time``       the clock (a date column, or text that parses as dates)
``measure``    a quantity to summarise (sum it if it is additive, otherwise average it)
``dimension``  a few categories or coded values (<= 60 levels): something to split by
``entity``     many repeated values (customers, products): something whose concentration matters
``id``/``text``/``year``/``constant``  columns the findings leave alone
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import polars as pl

MAX_LEVELS = 60
ADDITIVE = re.compile(
    r"(^|_)(qty|quantity|count|cnt|revenue|sales|amount|units|orders|visits|clicks|"
    r"rentals|casual|registered|casualties)($|_)",
    re.IGNORECASE,
)
ID_NAME = re.compile(
    r"(^|_)(id|index|instant|rownames|uuid|invoice|order_no|zip|postal)($|_)",
    re.IGNORECASE,
)
ENTITY_NAME = re.compile(
    r"(customer|user|client|product|item|sku|stock|account|store|seller|vendor|supplier|member)",
    re.IGNORECASE,
)


@dataclass
class Column:
    name: str
    role: str
    dtype: str
    n_missing: int
    n_unique: int
    agg: str = "mean"  # for measures: "sum" or "mean" when a series is coarsened
    calendar: str | None = (
        None  # for dimensions that follow the calendar (weekday, month, ...)
    )


@dataclass
class Profile:
    frame: pl.DataFrame
    time: str | None
    columns: dict[str, Column]
    native_grain: str | None
    event_table: bool  # several rows share a date: rows are events to count and add up
    row_label: str = "rows"
    notes: list[str] = field(default_factory=list)

    def of(self, role: str) -> list[Column]:
        return [c for c in self.columns.values() if c.role == role]

    @property
    def measures(self) -> list[Column]:
        return self.of("measure")

    @property
    def dimensions(self) -> list[Column]:
        return self.of("dimension")

    @property
    def entities(self) -> list[Column]:
        return self.of("entity")


DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d %b %Y",
)


def _parse_dates(s: pl.Series) -> pl.Series | None:
    """Text that is a date (or datetime) in at least 95% of its non-null values, else None."""
    sample = s.drop_nulls()
    if sample.is_empty():
        return None
    head = sample.head(2000)
    for fmt in DATE_FORMATS:
        share = head.str.strptime(pl.Datetime, fmt, strict=False).is_not_null().mean()
        if share is not None and share >= 0.95:
            return s.str.strptime(pl.Datetime, fmt, strict=False).cast(pl.Date)
    return None


def _coerce_time(
    frame: pl.DataFrame, requested: str | None
) -> tuple[pl.DataFrame, str | None]:
    candidates: list[str] = []
    for name in frame.columns:
        dtype = frame[name].dtype
        if dtype in (pl.Date, pl.Datetime):
            candidates.append(name)
        elif (
            dtype == pl.String
        ):  # any text column that parses as dates is a candidate clock
            parsed = _parse_dates(frame[name])
            if parsed is not None:
                frame = frame.with_columns(parsed.alias(name))
                candidates.append(name)
    if requested:
        if requested not in frame.columns:
            raise ValueError(
                f"no column {requested!r}; columns are {', '.join(frame.columns)}"
            )
        if requested not in candidates:
            raise ValueError(f"column {requested!r} does not hold dates")
        return frame, requested
    if not candidates:
        return frame, None
    best = max(candidates, key=lambda c: frame[c].n_unique())
    return frame, best


def native_grain(dates: pl.Series) -> str:
    """The finest calendar unit the timestamps are aligned to: day or month."""
    d = dates.drop_nulls().cast(pl.Date).unique()
    if d.len() >= 3 and (d.dt.day() == 1).all():
        return "month"
    return "day"


def _is_ordinal_codes(s: pl.Series) -> bool:
    return s.dtype.is_integer() and 2 <= s.n_unique() <= 12


def _calendar_link(frame: pl.DataFrame, time: str | None, col: str) -> str | None:
    """Which date part (month, weekday, year, quarter) a grouping column is a function of."""
    if time is None:
        return None
    work = frame.select(
        pl.col(time).cast(pl.Date).alias("_d"), pl.col(col).cast(pl.String).alias("_c")
    )
    work = work.drop_nulls()
    if work.height < 20:
        return None
    base = _entropy(work["_c"])
    if base < 1e-9:
        return None
    parts = {
        "month": pl.col("_d").dt.month(),
        "weekday": pl.col("_d").dt.weekday(),
        "year": pl.col("_d").dt.year(),
        "quarter": pl.col("_d").dt.quarter(),
    }
    for name, expr in parts.items():
        grouped = (
            work.with_columns(expr.alias("_p"))
            .group_by("_p")
            .agg(pl.col("_c").alias("v"))
        )
        cond = 0.0
        for row in grouped.iter_rows():
            cond += len(row[1]) / work.height * _entropy(pl.Series(row[1]))
        if cond / base <= 0.1 and work["_d"].dt.month().n_unique() > 1:
            return name
    return None


def _entropy(s: pl.Series) -> float:
    counts = s.value_counts()["count"].to_numpy().astype(float)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def profile_table(
    frame: pl.DataFrame,
    *,
    time: str | None = None,
    ignore: tuple[str, ...] = (),
    sum_columns: tuple[str, ...] = (),
    mean_columns: tuple[str, ...] = (),
    row_label: str = "rows",
) -> Profile:
    frame = frame.drop([c for c in ignore if c in frame.columns])
    frame, time = _coerce_time(frame, time)
    notes: list[str] = []
    grain = None
    event_table = False
    if time is not None:
        dates = frame[time].cast(pl.Date)
        grain = native_grain(dates)
        per_day = frame.group_by(dates.alias("_d")).len()["len"]
        event_table = float(per_day.median()) > 1.5
        if frame[time].null_count():
            notes.append(
                f"{frame[time].null_count():,} rows have no date and are left out of time series."
            )
    n = frame.height
    columns: dict[str, Column] = {}
    for name in frame.columns:
        s = frame[name]
        n_missing = s.null_count()
        n_unique = s.n_unique()
        base = {
            "name": name,
            "dtype": str(s.dtype),
            "n_missing": n_missing,
            "n_unique": n_unique,
        }
        if name == time:
            columns[name] = Column(role="time", **base)
            continue
        if n_unique <= 1:
            columns[name] = Column(role="constant", **base)
            continue
        if s.dtype in (pl.Date, pl.Datetime):
            columns[name] = Column(role="text", **base)
            continue
        if s.dtype == pl.Boolean:
            col = Column(role="dimension", **base)
            col.calendar = _calendar_link(frame, time, name)
            columns[name] = col
            continue
        if s.dtype == pl.String or s.dtype == pl.Categorical:
            nn = n - n_missing
            if n_unique <= MAX_LEVELS:
                col = Column(role="dimension", **base)
                col.calendar = _calendar_link(frame, time, name)
            elif ID_NAME.search(name) or n_unique / max(nn, 1) >= 0.9:
                col = Column(role="id", **base)
            elif nn / n_unique >= 3:
                col = Column(role="entity", **base)
            else:
                col = Column(role="text", **base)
            columns[name] = col
            continue
        # numeric
        vals = s.drop_nulls()
        looks_year = (
            vals.dtype.is_integer()
            and re.search(r"year|yr", name, re.IGNORECASE) is not None
            and 1800 <= float(vals.min()) <= float(vals.max()) <= 2200
        )
        if looks_year:
            columns[name] = Column(role="year", **base)
        elif ENTITY_NAME.search(name) and ID_NAME.search(name) and n_unique < 0.5 * n:
            columns[name] = Column(role="entity", **base)
        elif ID_NAME.search(name) or (
            vals.dtype.is_integer()
            and n_unique == vals.len()
            and bool((vals.diff().drop_nulls() > 0).all())
        ):
            columns[name] = Column(role="id", **base)
        elif _is_ordinal_codes(s) and name not in sum_columns:
            col = Column(role="dimension", **base)
            col.calendar = _calendar_link(frame, time, name)
            columns[name] = col
        else:
            agg = (
                "sum"
                if name in sum_columns
                else "mean"
                if name in mean_columns
                else ("sum" if ADDITIVE.search(name) else "mean")
            )
            columns[name] = Column(role="measure", agg=agg, **base)
    # two grouping columns that carve the rows the same way (product code and its description)
    entities = [c for c in columns.values() if c.role == "entity"]
    keep: list[Column] = []
    for c in sorted(
        entities, key=lambda c: (not ENTITY_NAME.search(c.name), c.n_unique)
    ):
        twin = next(
            (k for k in keep if abs(k.n_unique - c.n_unique) <= 0.05 * k.n_unique), None
        )
        if twin is None:
            keep.append(c)
        else:
            columns[c.name].role = "text"
            notes.append(
                f"{c.name!r} duplicates {twin.name!r} (same number of groups); using the latter."
            )
    return Profile(frame, time, columns, grain, event_table, row_label, notes)
