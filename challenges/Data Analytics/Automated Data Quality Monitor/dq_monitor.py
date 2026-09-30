"""Automated Data Quality Monitor: learn a baseline from good CSV batches, check new ones.

Run with:
    uv run python dq_monitor.py profile batches/*.csv -o baseline.json
    uv run python dq_monitor.py check new_batch.csv -b baseline.json --html report.html

`profile` learns what "normal" looks like (schema, volume, null rates, numeric and
categorical distributions, key uniqueness, string formats, timestamp cadence).
`check` compares one incoming batch against it and exits non-zero on an alert.
"""

from __future__ import annotations

import difflib
import glob
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import tomllib
import typer
from rich.console import Console
from rich.table import Table
from scipy import stats
from scipy.spatial.distance import jensenshannon

BASELINE_VERSION = 1
SEVERITIES = ("info", "warn", "critical")
_RANK = {s: i for i, s in enumerate(SEVERITIES)}

KNOWN_CHECKS = (
    "schema.missing_column",
    "schema.new_column",
    "schema.possible_rename",
    "schema.type_change",
    "schema.column_order",
    "volume.empty",
    "volume.row_count",
    "nulls.all_null",
    "nulls.rate_shift",
    "values.cast_failures",
    "values.out_of_range",
    "values.unseen_categories",
    "values.missing_categories",
    "drift.numeric",
    "drift.categorical",
    "format.unexpected_shape",
    "keys.duplicates",
    "freshness.implausible",
    "freshness.future",
    "freshness.stale",
    "freshness.span",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    # Used by `profile` (the decisions are frozen into the baseline file).
    "profile": {
        "categorical_max_unique": 50,
        "categorical_max_ratio": 0.05,
        "categorical_hard_cap": 500,
        "quantile_points": 501,
        "psi_bins": 10,
        "range_pad_frac": 0.25,
        "max_shapes": 20,
        "min_shape_share": 0.005,
    },
    "schema": {"rename_similarity": 0.6},
    "volume": {"warn_z": 3.0, "critical_z": 6.0, "min_rel_change": 0.3},
    "nulls": {"alpha": 0.001, "min_abs_change": 0.02, "critical_abs_change": 0.1},
    "cast": {"warn_share": 0.005, "critical_share": 0.05},
    "ranges": {"warn_share": 0.005, "critical_share": 0.05},
    "numeric": {
        "alpha": 0.001,
        "min_rows": 30,
        "psi_warn": 0.1,
        "psi_critical": 0.25,
        "ks_min_effect": 0.1,
    },
    "categorical": {
        "alpha": 0.001,
        "min_rows": 30,
        "js_warn": 0.02,
        "js_critical": 0.1,
        "unseen_warn_share": 0.01,
        "unseen_critical_share": 0.1,
        "missing_min_share": 0.05,
        "missing_critical_share": 0.25,
        "missing_alpha": 0.0001,
    },
    "format": {"warn_share": 0.01, "critical_share": 0.1},
    "freshness": {
        "max_lag_seconds": 0.0,  # 0 = derive 3x the typical baseline batch span
        "span_ratio": 3.0,
        "implausible_slack_days": 365.0,
        "implausible_warn_share": 0.005,
        "implausible_critical_share": 0.05,
    },
}


class ConfigError(ValueError):
    """The TOML config has an unknown key, wrong type, or bad severity."""


@dataclass
class Config:
    values: dict[str, dict[str, Any]]
    severity_overrides: dict[str, str] = field(default_factory=dict)
    ignore_columns: frozenset[str] = frozenset()

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.values[section]


def default_config() -> Config:
    return Config({k: dict(v) for k, v in DEFAULT_CONFIG.items()})


def load_config(path: Path | None) -> Config:
    """Merge a TOML file over the defaults. Unknown keys are errors, never ignored."""
    cfg = default_config()
    if path is None:
        return cfg
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    for section, body in raw.items():
        if section == "ignore":
            if set(body) - {"columns"} or not isinstance(body.get("columns", []), list):
                raise ConfigError("[ignore] only accepts columns = [...]")
            cfg.ignore_columns = frozenset(str(c) for c in body.get("columns", []))
        elif section == "severity_overrides":
            for check, sev in body.items():
                if check not in KNOWN_CHECKS:
                    raise ConfigError(f"[severity_overrides] unknown check {check!r}")
                if sev not in (*SEVERITIES, "off"):
                    raise ConfigError(
                        f"[severity_overrides] {check}: {sev!r} is not off/info/warn/critical"
                    )
                cfg.severity_overrides[check] = sev
        elif section in DEFAULT_CONFIG:
            for key, val in body.items():
                if key not in DEFAULT_CONFIG[section]:
                    raise ConfigError(f"[{section}] unknown key {key!r}")
                want = DEFAULT_CONFIG[section][key]
                ok = isinstance(val, (int, float)) and not isinstance(val, bool)
                if not ok or (
                    isinstance(want, int)
                    and not isinstance(want, bool)
                    and not isinstance(val, int)
                ):
                    raise ConfigError(
                        f"[{section}] {key} must be a number, got {val!r}"
                    )
                cfg.values[section][key] = type(want)(val)
        else:
            raise ConfigError(f"unknown config section [{section}]")
    return cfg


# ---------------------------------------------------------------------------
# Reading and typing
# ---------------------------------------------------------------------------


class BatchReadError(Exception):
    """A CSV batch (or a baseline file) could not be read."""


def read_batch(path: Path) -> pl.DataFrame:
    try:
        df = pl.read_csv(path, infer_schema_length=None, try_parse_dates=True)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise BatchReadError(f"{path}: {exc}") from exc
    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.Datetime) and dtype.time_zone:
            df = df.with_columns(pl.col(name).dt.replace_time_zone(None))
    return df


def logical_type(s: pl.Series) -> str:
    """One of int, float, bool, string, datetime, date, null, other."""
    if s.null_count() == s.len():
        return "null"
    dt = s.dtype
    if dt.is_integer():
        return "int"
    if dt.is_float():
        return "float"
    if dt == pl.Boolean:
        return "bool"
    if isinstance(dt, pl.Datetime):
        return "datetime"
    if dt == pl.Date:
        return "date"
    if dt in (pl.String, pl.Categorical):
        return "string"
    return "other"


_NUMERIC = ("int", "float")
_TEMPORAL = ("datetime", "date")
_TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)


def _non_null(s: pl.Series) -> int:
    n = s.len() - s.null_count()
    if s.dtype.is_float():
        n -= int(s.is_nan().sum())
    return n


def _parse_datetime(s: pl.Series) -> pl.Series:
    exprs = [
        pl.col("x").str.to_datetime(f, strict=False, time_unit="us")
        for f in _TS_FORMATS
    ]
    return s.to_frame("x").select(pl.coalesce(exprs)).to_series()


def coerce(s: pl.Series, ltype: str) -> tuple[pl.Series, int]:
    """Cast a batch column to the baseline's logical type: (series, cast_failures).

    A failure is a value that was present in the source and became null in the cast.
    NaN counts as null throughout, in the baseline and in batches, so the two agree.
    """
    src = _non_null(s)
    if ltype in _NUMERIC:
        if s.dtype.is_numeric() or s.dtype in (pl.String, pl.Boolean):
            out = s.cast(pl.Float64, strict=False).fill_nan(None)
        else:
            out = pl.Series(s.name, [None] * s.len(), dtype=pl.Float64)
    elif ltype in _TEMPORAL:
        if s.dtype == pl.String:
            out = _parse_datetime(s)
        elif isinstance(s.dtype, pl.Datetime) or s.dtype == pl.Date:
            out = s.cast(pl.Datetime("us"))
        else:
            out = pl.Series(s.name, [None] * s.len(), dtype=pl.Datetime("us"))
    else:  # string / bool / other: compare as text
        out = s.cast(pl.String, strict=False)
    return out.alias(s.name), max(0, src - _non_null(out))


def _merge_types(types: list[str]) -> str:
    kinds = {t for t in types if t != "null"}
    if not kinds:
        return "string"
    if len(kinds) == 1:
        return next(iter(kinds))
    if kinds <= set(_NUMERIC):
        return "float"
    if kinds <= set(_TEMPORAL):
        return "datetime"
    return "string"


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------


def _num(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def bin_counts(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Counts in (-inf, e0], (e0, e1], ..., (e_last, inf)."""
    return np.bincount(np.searchsorted(edges, x, side="left"), minlength=len(edges) + 1)


def psi(ref_props: np.ndarray, counts: np.ndarray, eps: float = 1e-4) -> float:
    """Population Stability Index of a batch's bin counts against reference proportions."""
    a = np.clip(counts / max(counts.sum(), 1), eps, None)
    e = np.clip(np.asarray(ref_props, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_against_reference(
    batch: np.ndarray, ref_quantiles: np.ndarray, n_ref: int
) -> tuple[float, float]:
    """KS statistic D and p-value of a batch against a baseline's stored quantile grid.

    D comes from `ks_2samp` against the grid. The p-value uses the *true* baseline
    size `n_ref` rather than the grid size, so the baseline's real sample size counts.
    """
    d = float(stats.ks_2samp(batch, ref_quantiles).statistic)
    n1, n2 = len(batch), max(n_ref, 1)
    en = max(1, round(n1 * n2 / (n1 + n2)))
    return d, float(stats.kstwo.sf(d, en))


def _char_class(ch: str) -> str:
    if ch.isdigit():
        return "9"
    if ch.isupper():
        return "A"
    if ch.islower():
        return "a"
    return ch


def shape_of(value: str) -> str:
    """Run-length character-class signature: 'TX-004217' -> 'A{2}-9{6}'."""
    parts = []
    for cls, grp in itertools.groupby(value, key=_char_class):
        n = sum(1 for _ in grp)
        parts.append(cls if n == 1 else f"{cls}{{{n}}}")
    return "".join(parts)


def tier(share: float, warn: float, critical: float) -> str | None:
    if share <= 0:
        return None
    if share >= critical:
        return "critical"
    return "warn" if share >= warn else "info"


def _epoch_seconds(s: pl.Series) -> np.ndarray:
    return s.drop_nulls().dt.epoch("us").to_numpy().astype(float) / 1e6


def _iso(seconds: float) -> str:
    return (
        datetime.fromtimestamp(seconds, tz=timezone.utc)
        .replace(tzinfo=None)
        .isoformat(sep=" ")
    )


# ---------------------------------------------------------------------------
# Profiling: learn a baseline
# ---------------------------------------------------------------------------


def _is_categorical(n_unique: int, n_nonnull: int, p: dict[str, Any]) -> bool:
    if n_unique == 0:
        return False
    if n_unique <= p["categorical_max_unique"]:
        return True
    return (
        n_unique <= p["categorical_hard_cap"]
        and n_unique / n_nonnull <= p["categorical_max_ratio"]
    )


def _profile_column(
    name: str,
    ltype: str,
    parts: list[pl.Series],
    force_key: bool,
    auto_keys: bool,
    p: dict[str, Any],
) -> dict[str, Any]:
    pooled = pl.concat(parts)
    n = pooled.len()
    nn = pooled.drop_nulls()
    if pooled.dtype.is_float():
        nn = nn.filter(~nn.is_nan())
    n_null = n - nn.len()
    spec: dict[str, Any] = {
        "name": name,
        "type": ltype,
        "present_batches": len(parts),
        "n": n,
        "n_null": n_null,
        "null_rate": n_null / n if n else 0.0,
        "unique": False,
    }
    if nn.len() == 0:
        spec["kind"] = "empty"
        return spec

    per_batch_unique = all(
        pt.drop_nulls().n_unique() == pt.drop_nulls().len() for pt in parts
    )
    if ltype in ("int", "string", "bool") and nn.len() >= 20:
        pooled_unique = nn.n_unique() == nn.len()
        spec["unique"] = force_key or (
            auto_keys and len(parts) >= 2 and per_batch_unique and pooled_unique
        )
    spec["forced_key"] = force_key

    if ltype in _NUMERIC:
        x = nn.cast(pl.Float64).to_numpy()
        fin = x[np.isfinite(x)]
        if fin.size == 0:
            spec["kind"] = "empty"
            return spec
        mn, mx = float(fin.min()), float(fin.max())
        span = mx - mn
        pad = p["range_pad_frac"] * (span if span > 0 else max(abs(mx), 1.0))
        lo, hi = mn - pad, mx + pad
        if mn >= 0:
            lo = max(lo, 0.0)
        if mx <= 0:
            hi = min(hi, 0.0)
        edges = np.unique(np.quantile(fin, np.linspace(0, 1, p["psi_bins"] + 1)[1:-1]))
        if edges.size == 0:
            edges = np.array([mn])
        counts = bin_counts(fin, edges)
        spec.update(
            kind="numeric",
            n_finite=int(fin.size),
            min=mn,
            max=mx,
            mean=float(fin.mean()),
            std=float(fin.std()),
            lower_bound=lo,
            upper_bound=hi,
            quantiles=np.quantile(
                fin, np.linspace(0, 1, p["quantile_points"])
            ).tolist(),
            psi_edges=edges.tolist(),
            psi_ref=(counts / counts.sum()).tolist(),
        )
    elif ltype in _TEMPORAL:
        secs = _epoch_seconds(nn)
        spans = [
            float(np.ptp(_epoch_seconds(pt)))
            for pt in parts
            if pt.drop_nulls().len() > 1
        ]
        spec.update(
            kind="temporal",
            min=float(secs.min()),
            max=float(secs.max()),
            min_iso=_iso(float(secs.min())),
            max_iso=_iso(float(secs.max())),
            median_batch_span=float(np.median(spans)) if spans else 0.0,
        )
    else:
        n_unique = nn.n_unique()
        if not spec["unique"] and _is_categorical(n_unique, nn.len(), p):
            vc = nn.value_counts()
            spec["kind"] = "categorical"
            spec["categories"] = {
                str(v): int(c)
                for v, c in zip(
                    vc[nn.name].to_list(), vc["count"].to_list(), strict=True
                )
            }
            # Good-Turing: the share of *rows* expected to carry a never-seen category
            # is (categories seen exactly once) / (rows). Long-tailed columns such as a
            # zone name keep producing new values forever; that is normal, not drift.
            spec["unseen_mass"] = (
                sum(1 for c in spec["categories"].values() if c == 1) / nn.len()
            )
        else:
            vc = nn.value_counts().head(200_000)
            shape_counts: dict[str, int] = {}
            for v, c in zip(vc[nn.name].to_list(), vc["count"].to_list(), strict=True):
                shape_counts[shape_of(str(v))] = shape_counts.get(
                    shape_of(str(v)), 0
                ) + int(c)
            total = sum(shape_counts.values())
            keep = sorted(shape_counts.items(), key=lambda kv: -kv[1])[
                : p["max_shapes"]
            ]
            spec["kind"] = "text"
            spec["shapes"] = {
                s: c / total for s, c in keep if c / total >= p["min_shape_share"]
            }
    return spec


def build_baseline(
    frames: list[tuple[str, pl.DataFrame]],
    cfg: Config | None = None,
    keys: tuple[str, ...] = (),
    auto_keys: bool = True,
) -> dict[str, Any]:
    """Learn a baseline from known-good batches given as (name, frame) pairs."""
    cfg = cfg or default_config()
    p = cfg["profile"]
    if not frames:
        raise ValueError("need at least one baseline batch")
    order: list[str] = []
    for _, df in frames:
        order.extend(c for c in df.columns if c not in order)

    notes: list[str] = []
    columns = []
    for name in order:
        present = [(bn, df[name]) for bn, df in frames if name in df.columns]
        ltype = _merge_types([logical_type(s) for _, s in present])
        parts = [coerce(s, ltype)[0] for _, s in present]
        columns.append(_profile_column(name, ltype, parts, name in keys, auto_keys, p))
        if len(present) < len(frames):
            notes.append(
                f"column {name!r} is missing from {len(frames) - len(present)} baseline batch(es)"
            )
    for k in keys:
        if k not in order:
            notes.append(f"--key {k!r} is not a column in any baseline batch")
    if len(frames) < 5:
        notes.append(
            "fewer than 5 baseline batches: volume and cadence estimates are weak"
        )
    if len(frames) == 1:
        notes.append("single baseline batch: automatic key detection is disabled")

    counts = [df.height for _, df in frames]
    return {
        "version": BASELINE_VERSION,
        "batches": [{"name": bn, "rows": df.height} for bn, df in frames],
        "row_counts": counts,
        "columns": columns,
        "profile_config": p,
        "notes": notes,
    }


def save_baseline(baseline: dict[str, Any], path: Path) -> None:
    Path(path).write_text(
        json.dumps(baseline, indent=1, sort_keys=True), encoding="utf-8"
    )


def load_baseline(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchReadError(f"{path}: {exc}") from exc
    if (
        not isinstance(data, dict)
        or data.get("version") != BASELINE_VERSION
        or "columns" not in data
    ):
        raise BatchReadError(f"{path}: not a version-{BASELINE_VERSION} baseline file")
    return data


# ---------------------------------------------------------------------------
# Checking a batch
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    check: str
    severity: str
    column: str | None
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "column": self.column,
            "message": self.message,
            "metrics": self.metrics,
        }


@dataclass
class CheckResult:
    batch: str
    rows: int
    alerts: list[Alert] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    frame: pl.DataFrame | None = field(default=None, repr=False, compare=False)

    @property
    def status(self) -> str:
        if not self.alerts:
            return "ok"
        return max((a.severity for a in self.alerts), key=_RANK.__getitem__)

    def count(self, severity: str) -> int:
        return sum(1 for a in self.alerts if a.severity == severity)

    def by_check(self, check: str) -> list[Alert]:
        return [a for a in self.alerts if a.check == check]

    def fired(self, min_severity: str = "warn") -> set[str]:
        return {
            a.check for a in self.alerts if _RANK[a.severity] >= _RANK[min_severity]
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch,
            "rows": self.rows,
            "status": self.status,
            "counts": {s: self.count(s) for s in SEVERITIES},
            "alerts": [a.to_dict() for a in self.alerts],
            "notes": self.notes,
        }


_NUMERIC_FAMILY = set(_NUMERIC)
_TEMPORAL_FAMILY = set(_TEMPORAL)


def _same_family(a: str, b: str) -> bool:
    return (
        a == b
        or (a in _NUMERIC_FAMILY and b in _NUMERIC_FAMILY)
        or (a in _TEMPORAL_FAMILY and b in _TEMPORAL_FAMILY)
    )


def _name_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if a and b and (a in b or b in a):
        ratio = max(ratio, 0.85)
    return ratio


def _type_change_severity(base: str, batch: str) -> str:
    if (
        base != batch
        and {base, batch} <= _NUMERIC_FAMILY | _TEMPORAL_FAMILY
        and _same_family(base, batch)
    ):
        return "info"
    if batch == "string" and base != "string":
        return "critical"
    return "warn"


def check_schema(
    baseline: dict[str, Any], df: pl.DataFrame, cfg: Config
) -> list[Alert]:
    alerts: list[Alert] = []
    specs = {c["name"]: c for c in baseline["columns"]}
    base_order = [c["name"] for c in baseline["columns"]]
    n_batches = len(baseline["batches"])
    missing = [c for c in base_order if c not in df.columns]
    new = [c for c in df.columns if c not in specs]

    thr = cfg["schema"]["rename_similarity"]
    pairs: list[tuple[str, str, float]] = []
    for m in missing:
        best = None
        for n in new:
            ltype_new = logical_type(df[n])
            if not _same_family(specs[m]["type"], ltype_new) and ltype_new != "null":
                continue
            sim = _name_similarity(m, n)
            same_pos = base_order.index(m) == df.columns.index(n)
            score = sim if sim >= thr else (thr if same_pos else 0.0)
            if score >= thr and (best is None or score > best[1]):
                best = (n, score)
        if best and all(best[0] != p[1] for p in pairs):
            pairs.append((m, best[0], best[1]))
    renamed_from = {p[0] for p in pairs}
    renamed_to = {p[1] for p in pairs}

    for old, newname, score in pairs:
        alerts.append(
            Alert(
                "schema.possible_rename",
                "critical",
                old,
                f"column {old!r} is gone and {newname!r} has the same type in a similar place: likely a rename",
                {"old": old, "new": newname, "confidence": round(score, 3)},
            )
        )
    for m in missing:
        if m in renamed_from:
            continue
        optional = specs[m]["present_batches"] < n_batches
        alerts.append(
            Alert(
                "schema.missing_column",
                "info" if optional else "critical",
                m,
                f"column {m!r} is missing from the batch"
                + (" (it was optional in the baseline)" if optional else ""),
            )
        )
    for n in new:
        if n in renamed_to:
            continue
        alerts.append(
            Alert(
                "schema.new_column",
                "warn",
                n,
                f"column {n!r} is not in the baseline schema",
            )
        )

    for name, spec in specs.items():
        if name not in df.columns:
            continue
        lt = logical_type(df[name])
        if lt == "null" or lt == spec["type"]:
            continue
        alerts.append(
            Alert(
                "schema.type_change",
                _type_change_severity(spec["type"], lt),
                name,
                f"column {name!r} was {spec['type']} in the baseline and is {lt} in this batch",
                {"baseline": spec["type"], "batch": lt},
            )
        )

    shared_base = [c for c in base_order if c in df.columns]
    shared_batch = [c for c in df.columns if c in specs]
    if shared_base != shared_batch:
        alerts.append(
            Alert(
                "schema.column_order",
                "info",
                None,
                "columns are in a different order than in the baseline",
            )
        )
    return alerts


def check_volume(baseline: dict[str, Any], rows: int, cfg: Config) -> list[Alert]:
    if rows == 0:
        return [
            Alert(
                "volume.empty", "critical", None, "the batch has no rows", {"rows": 0}
            )
        ]
    counts = np.asarray(baseline["row_counts"], dtype=float)
    k = len(counts)
    mean = float(counts.mean())
    std = float(counts.std(ddof=1)) if k > 1 else 0.0
    # Poisson floor: even a perfectly steady source varies by about sqrt(mean).
    scale = max(std * math.sqrt(1 + 1 / k), math.sqrt(mean), 1.0)
    z = (rows - mean) / scale
    rel = (rows - mean) / mean if mean else 0.0
    v = cfg["volume"]
    if abs(rel) < v["min_rel_change"] or abs(z) < v["warn_z"]:
        return []
    sev = "critical" if abs(z) >= v["critical_z"] else "warn"
    word = "more" if rel > 0 else "fewer"
    return [
        Alert(
            "volume.row_count",
            sev,
            None,
            f"{rows} rows vs a baseline of {mean:.0f} +/- {std:.0f} per batch ({abs(rel):.0%} {word}, z = {z:+.1f})",
            {
                "rows": rows,
                "baseline_mean": mean,
                "baseline_std": std,
                "z": _num(z),
                "rel_change": _num(rel),
            },
        )
    ]


def _check_nulls(
    spec: dict[str, Any], src: pl.Series, rows: int, cfg: Config
) -> list[Alert]:
    name = spec["name"]
    n_null = rows - _non_null(src)
    if rows == 0:
        return []
    c = cfg["nulls"]
    p0 = spec["null_rate"]
    rate = n_null / rows
    if n_null == rows and p0 < 1:
        return [
            Alert(
                "nulls.all_null",
                "critical",
                name,
                f"column {name!r} is entirely null",
                {"baseline_rate": p0},
            )
        ]
    smoothed = (spec["n_null"] + 0.5) / (spec["n"] + 1)
    pval = float(stats.binomtest(n_null, rows, smoothed).pvalue)
    delta = rate - p0
    if pval >= c["alpha"] or abs(delta) < c["min_abs_change"]:
        return []
    if delta < 0:
        sev = "info"
    else:
        sev = "critical" if delta >= c["critical_abs_change"] else "warn"
    return [
        Alert(
            "nulls.rate_shift",
            sev,
            name,
            f"null rate of {name!r} is {rate:.1%} vs {p0:.1%} in the baseline (p = {pval:.1e})",
            {"rate": rate, "baseline_rate": p0, "p_value": pval},
        )
    ]


def _check_numeric(spec: dict[str, Any], col: pl.Series, cfg: Config) -> list[Alert]:
    name = spec["name"]
    alerts: list[Alert] = []
    x = col.drop_nulls().to_numpy().astype(float)
    if x.size == 0:
        return alerts
    r = cfg["ranges"]
    out = (x < spec["lower_bound"]) | (x > spec["upper_bound"])
    share = float(out.mean())
    sev = tier(share, r["warn_share"], r["critical_share"])
    if sev:
        bad = x[out]
        alerts.append(
            Alert(
                "values.out_of_range",
                sev,
                name,
                f"{int(out.sum())} of {x.size} values of {name!r} fall outside the learned range "
                f"[{spec['lower_bound']:.6g}, {spec['upper_bound']:.6g}] "
                f"(baseline observed {spec['min']:.6g} to {spec['max']:.6g}); "
                f"e.g. {', '.join(f'{v:.6g}' for v in bad[:3])}",
                {"count": int(out.sum()), "share": share},
            )
        )
    fin = x[np.isfinite(x)]
    n = cfg["numeric"]
    if fin.size < n["min_rows"]:
        return alerts
    d, pval = ks_against_reference(fin, np.asarray(spec["quantiles"]), spec["n_finite"])
    edges = np.asarray(spec["psi_edges"])
    score = psi(np.asarray(spec["psi_ref"]), bin_counts(fin, edges))
    # Significance and effect size must both clear their bar: with enough rows KS flags
    # differences nobody cares about, and with few rows PSI is inflated by pure noise.
    if pval < n["alpha"] and d >= n["ks_min_effect"] and score >= n["psi_warn"]:
        sev = "critical" if score >= n["psi_critical"] else "warn"
        alerts.append(
            Alert(
                "drift.numeric",
                sev,
                name,
                f"distribution of {name!r} shifted: PSI = {score:.3f}, KS D = {d:.3f} (p = {pval:.1e}); "
                f"median {float(np.median(fin)):.6g} vs {float(np.median(spec['quantiles'])):.6g} in the baseline",
                {
                    "psi": score,
                    "ks_d": d,
                    "p_value": pval,
                    "batch_median": float(np.median(fin)),
                },
            )
        )
    return alerts


def _check_categorical(
    spec: dict[str, Any], col: pl.Series, cfg: Config
) -> list[Alert]:
    name = spec["name"]
    alerts: list[Alert] = []
    vals = col.drop_nulls()
    n = vals.len()
    if n == 0:
        return alerts
    c = cfg["categorical"]
    vc = vals.value_counts()
    obs = dict(zip(vc[vals.name].to_list(), vc["count"].to_list(), strict=True))
    base = spec["categories"]
    base_total = sum(base.values())
    ref = {k: v / base_total for k, v in base.items()}

    unseen = {k: v for k, v in obs.items() if k not in base}
    n_unseen = sum(unseen.values())
    share = n_unseen / n
    mass = max(spec.get("unseen_mass", 0.0), 0.5 / (base_total + 1))
    excess = share - spec.get("unseen_mass", 0.0)
    sev = tier(excess, c["unseen_warn_share"], c["unseen_critical_share"])
    if (
        sev
        and n_unseen
        and float(stats.binomtest(n_unseen, n, mass, alternative="greater").pvalue)
        < c["alpha"]
    ):
        ex = ", ".join(repr(k) for k in sorted(unseen, key=lambda k: -unseen[k])[:3])
        alerts.append(
            Alert(
                "values.unseen_categories",
                sev,
                name,
                f"{n_unseen} of {n} values of {name!r} ({share:.1%}) are categories never seen in the baseline "
                f"(about {spec.get('unseen_mass', 0.0):.1%} expected from its long tail): {ex}",
                {
                    "share": share,
                    "expected_share": spec.get("unseen_mass", 0.0),
                    "categories": len(unseen),
                },
            )
        )

    for cat, share_ref in ref.items():
        if (
            share_ref >= c["missing_min_share"]
            and cat not in obs
            and (1 - share_ref) ** n < c["missing_alpha"]
        ):
            alerts.append(
                Alert(
                    "values.missing_categories",
                    "critical" if share_ref >= c["missing_critical_share"] else "warn",
                    name,
                    f"category {cat!r} of {name!r} made up {share_ref:.1%} of the baseline and is absent from all {n} rows",
                    {"category": cat, "baseline_share": share_ref},
                )
            )

    if n >= c["min_rows"]:
        big = [k for k, s in ref.items() if s * n >= 5]
        if big:
            p_ref = np.array(
                [ref[k] for k in big] + [max(0.0, 1.0 - sum(ref[k] for k in big))]
            )
            f_obs = np.array(
                [obs.get(k, 0) for k in big] + [n - sum(obs.get(k, 0) for k in big)],
                dtype=float,
            )
            if (
                p_ref[-1] <= 0
            ):  # no long tail in the baseline: unseen values are reported above
                p_ref, f_obs = p_ref[:-1], f_obs[:-1]
            if p_ref.size >= 2 and f_obs.sum() > 0:
                p_ref = p_ref / p_ref.sum()
                n_eff = float(f_obs.sum())
                pval = float(stats.chisquare(f_obs, p_ref * n_eff).pvalue)
                jsd = float(jensenshannon(p_ref, f_obs / n_eff, base=2) ** 2)
                if pval < c["alpha"] and jsd >= c["js_warn"]:
                    alerts.append(
                        Alert(
                            "drift.categorical",
                            "critical" if jsd >= c["js_critical"] else "warn",
                            name,
                            f"category mix of {name!r} shifted: Jensen-Shannon divergence = {jsd:.3f} bits, "
                            f"chi-square p = {pval:.1e}",
                            {"js_divergence": jsd, "p_value": pval},
                        )
                    )
    return alerts


def _check_text(spec: dict[str, Any], col: pl.Series, cfg: Config) -> list[Alert]:
    name = spec["name"]
    vals = col.drop_nulls()
    if vals.len() == 0 or not spec.get("shapes"):
        return []
    vc = vals.value_counts()
    bad_total = 0
    examples: list[str] = []
    for v, cnt in zip(vc[vals.name].to_list(), vc["count"].to_list(), strict=True):
        if shape_of(str(v)) not in spec["shapes"]:
            bad_total += int(cnt)
            if len(examples) < 3:
                examples.append(str(v))
    share = bad_total / vals.len()
    f = cfg["format"]
    sev = tier(share, f["warn_share"], f["critical_share"])
    if not sev:
        return []
    known = ", ".join(list(spec["shapes"])[:3])
    return [
        Alert(
            "format.unexpected_shape",
            sev,
            name,
            f"{bad_total} of {vals.len()} values of {name!r} do not match a learned format ({known}); "
            f"e.g. {', '.join(repr(e) for e in examples)}",
            {"count": bad_total, "share": share, "examples": examples},
        )
    ]


def _check_key(spec: dict[str, Any], col: pl.Series) -> list[Alert]:
    vals = col.drop_nulls()
    dups = vals.len() - vals.n_unique()
    if dups <= 0:
        return []
    return [
        Alert(
            "keys.duplicates",
            "critical",
            spec["name"],
            f"{spec['name']!r} is a unique key in the baseline but {dups} of {vals.len()} values repeat",
            {"duplicates": dups},
        )
    ]


def _check_temporal(
    spec: dict[str, Any], col: pl.Series, cfg: Config, as_of: float | None
) -> list[Alert]:
    name = spec["name"]
    secs = _epoch_seconds(col)
    if secs.size == 0:
        return []
    f = cfg["freshness"]
    alerts: list[Alert] = []
    slack = f["implausible_slack_days"] * 86400
    lo = spec["min"] - slack
    hi = max(spec["max"], as_of if as_of is not None else spec["max"]) + slack
    out = (secs < lo) | (secs > hi)
    share = float(out.mean())
    sev = tier(share, f["implausible_warn_share"], f["implausible_critical_share"])
    if sev:
        alerts.append(
            Alert(
                "freshness.implausible",
                sev,
                name,
                f"{int(out.sum())} of {secs.size} timestamps in {name!r} are more than {f['implausible_slack_days']:g} "
                f"days outside the baseline's {spec['min_iso']} to {spec['max_iso']}; e.g. {_iso(float(secs[out][0]))}",
                {"count": int(out.sum()), "share": share},
            )
        )
    good = secs[~out]
    if good.size > 1 and spec["median_batch_span"] > 0:
        span = float(np.ptp(good))
        ratio = span / spec["median_batch_span"]
        if ratio > f["span_ratio"] or ratio < 1 / f["span_ratio"]:
            alerts.append(
                Alert(
                    "freshness.span",
                    "warn",
                    name,
                    f"{name!r} covers {span / 86400:.2f} days; baseline batches cover {spec['median_batch_span'] / 86400:.2f} "
                    f"({ratio:.1f}x): the batch is not the same slice of time",
                    {"span_seconds": span, "ratio": ratio},
                )
            )
    if as_of is not None and good.size:
        newest = float(good.max())
        future = float((good > as_of).mean())
        sev = tier(future, f["implausible_warn_share"], f["implausible_critical_share"])
        if sev:
            alerts.append(
                Alert(
                    "freshness.future",
                    sev,
                    name,
                    f"{name!r} has timestamps after the as-of time ({_iso(as_of)}); newest {_iso(newest)}",
                    {"share": future},
                )
            )
        lag = as_of - newest
        limit = f["max_lag_seconds"] or 3 * spec["median_batch_span"]
        if limit > 0 and lag > limit:
            alerts.append(
                Alert(
                    "freshness.stale",
                    "critical" if lag > 3 * limit else "warn",
                    name,
                    f"newest {name!r} value is {lag / 3600:.1f} h before the as-of time; allowed lag is {limit / 3600:.1f} h",
                    {"lag_seconds": lag, "limit_seconds": limit},
                )
            )
    return alerts


def check_batch(
    baseline: dict[str, Any],
    df: pl.DataFrame,
    cfg: Config | None = None,
    as_of: datetime | None = None,
    name: str = "batch",
) -> CheckResult:
    """Compare one batch against a baseline and return every alert."""
    cfg = cfg or default_config()
    result = CheckResult(batch=name, rows=df.height, frame=df)
    alerts = result.alerts
    alerts.extend(check_schema(baseline, df, cfg))
    alerts.extend(check_volume(baseline, df.height, cfg))

    as_of_s = None
    if as_of is not None:
        if as_of.tzinfo is not None:
            as_of = as_of.astimezone(timezone.utc).replace(tzinfo=None)
        as_of_s = as_of.replace(tzinfo=timezone.utc).timestamp()

    if df.height > 0:
        for spec in baseline["columns"]:
            cname = spec["name"]
            if cname not in df.columns or cname in cfg.ignore_columns:
                continue
            src = df[cname]
            lt = logical_type(src)
            alerts.extend(_check_nulls(spec, src, df.height, cfg))
            if lt == "null" or spec["kind"] == "empty":
                continue
            col, failures = coerce(src, spec["type"])
            if lt != spec["type"] and failures:
                share = failures / max(_non_null(src), 1)
                sev = tier(
                    share, cfg["cast"]["warn_share"], cfg["cast"]["critical_share"]
                )
                if sev:
                    alerts.append(
                        Alert(
                            "values.cast_failures",
                            sev,
                            cname,
                            f"{failures} of {_non_null(src)} values of {cname!r} cannot be read as {spec['type']}",
                            {"count": failures, "share": share},
                        )
                    )
            if spec["unique"]:
                alerts.extend(_check_key(spec, col))
                if spec["kind"] == "text":
                    alerts.extend(_check_text(spec, col, cfg))
                continue
            kind = spec["kind"]
            if kind == "numeric":
                alerts.extend(_check_numeric(spec, col, cfg))
            elif kind == "categorical":
                alerts.extend(_check_categorical(spec, col, cfg))
            elif kind == "text":
                alerts.extend(_check_text(spec, col, cfg))
            elif kind == "temporal":
                alerts.extend(_check_temporal(spec, col, cfg, as_of_s))
    else:
        result.notes.append("empty batch: value-level checks were skipped")

    kept = []
    for a in alerts:
        if a.column in cfg.ignore_columns:
            continue
        override = cfg.severity_overrides.get(a.check)
        if override == "off":
            continue
        if override:
            a.severity = override
        kept.append(a)
    result.alerts = sorted(
        kept, key=lambda a: (-_RANK[a.severity], a.check, a.column or "")
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help=__doc__.split("\n\n")[0]
)
_err = Console(stderr=True)
_out = Console()
EXIT_USAGE = 2


def expand_inputs(items: list[Path]) -> list[Path]:
    """Files, directories (all *.csv inside) and shell-style globs, in a stable order."""
    found: list[Path] = []
    for item in items:
        if item.is_dir():
            found.extend(sorted(item.glob("*.csv")))
        elif item.exists():
            found.append(item)
        else:
            matches = sorted(Path(m) for m in glob.glob(str(item)))
            if not matches:
                raise BatchReadError(f"{item}: no such file")
            found.extend(matches)
    seen: dict[Path, None] = {}
    for f in found:
        seen.setdefault(f, None)
    return list(seen)


def _parse_as_of(text: str | None) -> datetime | None:
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise typer.BadParameter(
            f"--as-of must be ISO-8601, e.g. 2019-04-01T00:00:00 ({exc})"
        ) from exc


@app.command()
def profile(
    batches: Annotated[
        list[Path],
        typer.Argument(help="Known-good CSV files, directories of CSVs, or globs."),
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Where to write the baseline JSON.")
    ] = Path("baseline.json"),
    config: Annotated[
        Path | None, typer.Option("-c", "--config", help="TOML config.")
    ] = None,
    key: Annotated[
        list[str] | None,
        typer.Option("--key", help="Force a column to be treated as a unique key."),
    ] = None,
    no_auto_keys: Annotated[
        bool, typer.Option("--no-auto-keys", help="Do not infer unique keys.")
    ] = False,
) -> None:
    """Learn a baseline from known-good batches."""
    try:
        cfg = load_config(config)
        files = expand_inputs(batches)
        if not files:
            raise BatchReadError("no CSV files found")
        frames = [(f.name, read_batch(f)) for f in files]
        baseline = build_baseline(
            frames, cfg, tuple(key or ()), auto_keys=not no_auto_keys
        )
    except (ConfigError, BatchReadError) as exc:
        _err.print(f"error: {exc}", markup=False)
        raise typer.Exit(EXIT_USAGE) from exc
    save_baseline(baseline, output)
    _out.print(
        f"Profiled {len(frames)} batches ({sum(baseline['row_counts'])} rows, {len(baseline['columns'])} columns) -> {output}"
    )
    table = Table("column", "type", "kind", "null rate", "key")
    for c in baseline["columns"]:
        table.add_row(
            c["name"],
            c["type"],
            c["kind"],
            f"{c['null_rate']:.1%}",
            "yes" if c["unique"] else "",
        )
    _out.print(table)
    for note in baseline["notes"]:
        _out.print(f"note: {note}", markup=False)


@app.command()
def check(
    batch: Annotated[Path, typer.Argument(help="The incoming CSV batch.")],
    baseline: Annotated[
        Path, typer.Option("-b", "--baseline", help="Baseline JSON from `profile`.")
    ],
    config: Annotated[
        Path | None, typer.Option("-c", "--config", help="TOML config.")
    ] = None,
    html_out: Annotated[
        Path | None, typer.Option("--html", help="Write a self-contained HTML report.")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write a JSON report.")
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="Reference time (ISO-8601) for freshness checks."),
    ] = None,
    fail_on: Annotated[
        str,
        typer.Option(
            "--fail-on",
            help="Exit 1 at this severity or above: info, warn or critical.",
        ),
    ] = "warn",
) -> None:
    """Check one batch against a baseline. Exit 1 on an alert, 2 on bad input."""
    if fail_on not in SEVERITIES:
        raise typer.BadParameter(f"--fail-on must be one of {', '.join(SEVERITIES)}")
    parsed_as_of = _parse_as_of(as_of)
    try:
        cfg = load_config(config)
        base = load_baseline(baseline)
        df = read_batch(batch)
    except (ConfigError, BatchReadError) as exc:
        _err.print(f"error: {exc}", markup=False)
        raise typer.Exit(EXIT_USAGE) from exc
    result = check_batch(base, df, cfg, parsed_as_of, name=batch.name)

    if json_out:
        json_out.write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")
    if html_out:
        from dq_report import render_report

        html_out.write_text(render_report(result, base), encoding="utf-8")

    style = {"critical": "bold red", "warn": "yellow", "info": "cyan"}
    _out.print(
        f"{batch.name}: {result.rows} rows, status {result.status.upper()} "
        f"({result.count('critical')} critical, {result.count('warn')} warn, {result.count('info')} info)",
        markup=False,
    )
    if result.alerts:
        table = Table("severity", "check", "column", "message")
        for a in result.alerts:
            table.add_row(
                a.severity, a.check, a.column or "-", a.message, style=style[a.severity]
            )
        _out.print(table)
    for note in result.notes:
        _out.print(f"note: {note}", markup=False)
    worst = _RANK[result.status] if result.alerts else -1
    if worst >= _RANK[fail_on]:
        raise typer.Exit(1)


if __name__ == "__main__":
    sys.exit(app())
