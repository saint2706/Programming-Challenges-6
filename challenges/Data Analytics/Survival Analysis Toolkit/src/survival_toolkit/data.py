"""Loading time-to-event tables: three bundled datasets, or any CSV/Parquet file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from survival_toolkit.paths import project_root
from survival_toolkit.types import Survival, SurvivalError


@dataclass(frozen=True)
class Spec:
    """How a bundled table maps onto time, event and covariates."""

    name: str
    title: str
    file: str
    time: str
    event: str
    unit: str
    group: str  # the default column to split curves by
    covariates: tuple[str, ...]
    reference: dict[str, str] = field(
        default_factory=dict
    )  # category left out of the dummies
    note: str = ""
    horizons: tuple[float, ...] = ()
    strata: str | None = None  # a Cox baseline split the report adds
    caveat: str = ""
    tau: float | None = None


DATASETS: dict[str, Spec] = {
    "telco": Spec(
        name="telco",
        title="IBM Telco customer churn (7,032 subscribers with any tenure)",
        file="telco.parquet",
        time="tenure",
        event="churned",
        unit="months",
        group="Contract",
        covariates=(
            "Contract",
            "InternetService",
            "PaymentMethod",
            "PaperlessBilling",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "TechSupport",
            "OnlineSecurity",
            "MonthlyCharges",
        ),
        reference={
            "Contract": "Month-to-month",
            "InternetService": "DSL",
            "PaymentMethod": "Electronic check",
            "PaperlessBilling": "No",
            "Partner": "No",
            "Dependents": "No",
            "TechSupport": "No",
            "OnlineSecurity": "No",
        },
        note="tenure in months; churned = left; active subscribers are right-censored at tenure",
        horizons=(6, 12, 24, 48),
        strata="Contract",
        caveat=(
            "Telco's Churn flag means the subscriber left within the observation window (IBM's "
            "documentation says both 'within the last month' and 'this quarter'), so tenure is a "
            "snapshot of customers present in that window, not a full set of lifetimes. Read the "
            "curves as a description of this file; `--entry entry_1m` shows the window-aware view."
        ),
        tau=60,
    ),
    "lung": Spec(
        name="lung",
        title="NCCTG advanced lung cancer (228 patients)",
        file="lung.csv",
        time="time",
        event="event",
        unit="days",
        group="sex",
        covariates=("age", "sex", "ph_ecog"),
        note="the dataset behind R's survival vignettes; sex is 1 = male, 2 = female",
        horizons=(180, 365, 730),
        tau=500,
    ),
    "gbsg": Spec(
        name="gbsg",
        title="German Breast Cancer Study Group (686 node-positive patients)",
        file="gbsg.csv",
        time="rfstime",
        event="event",
        unit="days",
        group="hormon",
        covariates=("age", "meno", "size", "grade", "nodes", "pgr", "er", "hormon"),
        note="recurrence-free survival; hormon = hormonal therapy",
        horizons=(365, 730, 1095, 1825),
        tau=1825,
    ),
}


@dataclass(frozen=True)
class Prepared:
    """A cleaned table with its Survival arrays; ``dropped`` says what cleaning removed."""

    frame: pl.DataFrame
    survival: Survival
    dropped: dict[str, int]
    time: str
    event: str
    entry: str | None
    unit: str = "time units"


def data_dir() -> Path:
    return project_root() / "sample_data"


def read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pl.read_csv(
            path, separator=sep, infer_schema_length=10_000, null_values=["NA", ""]
        )
    raise SurvivalError(f"cannot read {path.suffix} files; use CSV, TSV or Parquet")


def _event_flags(col: pl.Series, event_value: str | None) -> pl.Series:
    if event_value is not None:
        return col.cast(pl.String) == event_value
    if col.dtype == pl.Boolean:
        return col
    if col.dtype.is_numeric():
        values = set(col.drop_nulls().unique().to_list())
        if values <= {0, 1}:
            return col == 1
        if values <= {1, 2}:
            return col == 2  # R's convention: 1 = censored, 2 = event
    raise SurvivalError(
        f"cannot tell which values of {col.name!r} mean an event; pass --event-value"
    )


def prepare(
    frame: pl.DataFrame,
    *,
    time: str,
    event: str,
    entry: str | None = None,
    event_value: str | None = None,
    unit: str = "time units",
) -> Prepared:
    """Validate and clean ``frame`` into Survival arrays, saying what was dropped and why."""
    needed = [time, event] + ([entry] if entry else [])
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise SurvivalError(
            f"no column {missing[0]!r}; columns are {', '.join(frame.columns)}"
        )
    frame = frame.with_columns(
        pl.col(time).cast(pl.Float64, strict=False).alias(time),
        *([pl.col(entry).cast(pl.Float64, strict=False).alias(entry)] if entry else []),
    )
    flags = _event_flags(frame[event], event_value)
    frame = frame.with_columns(flags.alias("_event_flag"))
    dropped: dict[str, int] = {}
    n0 = frame.height
    frame = frame.filter(pl.col(time).is_not_null() & pl.col(time).is_finite())
    frame = frame.filter(pl.col("_event_flag").is_not_null())
    if entry:
        frame = frame.filter(pl.col(entry).is_not_null() & pl.col(entry).is_finite())
    dropped["missing time, event or entry"] = n0 - frame.height
    n1 = frame.height
    low = pl.col(entry) if entry else pl.lit(0.0)
    frame = frame.filter(pl.col(time) > low)
    dropped["no exposure (time <= entry or <= 0)"] = n1 - frame.height
    if frame.height == 0:
        raise SurvivalError("no usable rows after cleaning")
    surv = Survival(
        frame[time].to_numpy(),
        frame["_event_flag"].to_numpy(),
        frame[entry].to_numpy() if entry else None,
    )
    return Prepared(frame.drop("_event_flag"), surv, dropped, time, event, entry, unit)


def load_builtin(name: str) -> tuple[Spec, Prepared]:
    spec = DATASETS.get(name)
    if spec is None:
        raise SurvivalError(
            f"unknown dataset {name!r}; choose from {', '.join(DATASETS)}"
        )
    path = data_dir() / spec.file
    if not path.exists():
        raise SurvivalError(f"{path} is missing; run `uv run survival fetch`")
    frame = _tidy(spec.name, read_table(path))
    prepared = prepare(frame, time=spec.time, event=spec.event, unit=spec.unit)
    return spec, prepared


def _tidy(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Per-dataset fixes that belong to the data, not to the estimators."""
    if name != "telco":
        return frame
    # 'No internet service' repeats InternetService == 'No' in these columns; keeping it as its own
    # level makes the dummy columns exactly collinear, so fold it into 'No'.
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    addons += ["StreamingTV", "StreamingMovies"]
    frame = frame.with_columns(
        [
            pl.col(c).replace("No internet service", "No")
            for c in addons
            if c in frame.columns
        ]
    )
    # the observation-window view: a customer seen for the last 1 (or 3) months only, so late entry
    return frame.with_columns(
        (pl.col("tenure") - 1).clip(lower_bound=0).cast(pl.Float64).alias("entry_1m"),
        (pl.col("tenure") - 3).clip(lower_bound=0).cast(pl.Float64).alias("entry_3m"),
    )


def source_notes() -> dict:
    path = data_dir() / "SOURCES.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
