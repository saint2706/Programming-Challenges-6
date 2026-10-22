"""Re-deriving every claim from the raw table with a second engine (DuckDB SQL plus scipy).

The detectors work on polars frames and my own code paths. This module takes only a finding's
*claim* (its facts, and the bucket dates or group names it names), recomputes the numbers with SQL
aggregation and library statistics (``scipy.stats``, ``statsmodels`` called directly), and compares.
A finding whose numbers cannot be reproduced is dropped from the report, and the report says so.
What it does not do: it cannot catch a wrong *choice* (say, a threshold), only a wrong number.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import STL

from report_generator.findings import Finding
from report_generator.profile import Profile
from report_generator.series import ROWS
from report_generator.stats_ts import block_size

REL_TOL = 1e-6


@dataclass
class Check:
    name: str
    claimed: float
    recomputed: float
    ok: bool


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def close(a: float, b: float, tol: float = REL_TOL) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=1e-9)


class Verifier:
    def __init__(self, profile: Profile):
        self.profile = profile
        self.dir = tempfile.TemporaryDirectory()
        path = Path(self.dir.name) / "data.parquet"
        profile.frame.write_parquet(path)
        self.con = duckdb.connect()
        self.con.execute(
            f"CREATE VIEW data AS SELECT * FROM read_parquet('{path.as_posix()}')"
        )
        self.time = profile.time

    def close(self) -> None:
        self.con.close()
        self.dir.cleanup()

    def one(self, sql: str, params: list | None = None):
        return self.con.execute(sql, params or []).fetchone()[0]

    def all(self, sql: str, params: list | None = None) -> list[tuple]:
        return self.con.execute(sql, params or []).fetchall()

    # ---- time series -------------------------------------------------------------------------
    def series_values(self, f: Finding) -> np.ndarray:
        """The finding's own buckets, aggregated again in SQL (bucketing and aggregation re-checked)."""
        key, grain = f.columns[0], f.stats["grain"]
        t = f"CAST({q(self.time)} AS DATE)"
        anchor = self.one(f"SELECT min({t}) FROM data")
        if grain == "day":
            bucket = t
        elif grain == "week":
            bucket = f"(DATE '{anchor}' + CAST(7 * (date_diff('day', DATE '{anchor}', {t}) // 7) AS INTEGER))"
        else:
            bucket = f"CAST(date_trunc('{grain}', {t}) AS DATE)"
        if key == ROWS:
            agg = "count(*)"
        else:
            agg = f"{'sum' if self.profile.columns[key].agg == 'sum' else 'avg'}({q(key)})"
        rows = dict(
            self.all(
                f"SELECT {bucket} AS b, {agg} AS v FROM data WHERE {t} IS NOT NULL GROUP BY 1"
            )
        )
        out = []
        dates = f.chart["dates"] if f.chart and "dates" in f.chart else f.stats["dates"]
        for d in dates:
            v = rows.get(np.datetime64(d, "D").astype(object))
            out.append(np.nan if v is None else float(v))
        return np.array(out)

    def _decompose(
        self, y: np.ndarray, period: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(seasonal, remainder) the way the detectors choose: STL with 3+ cycles, a fixed profile with 2."""
        n = y.size
        t = np.arange(n, dtype=float)
        if not period or n < 2 * period:
            return np.zeros(n), y - np.polyval(np.polyfit(t, y, 1), t)
        if n // period >= 3:
            r = STL(
                y,
                period=period,
                seasonal=2 * n + 1,
                seasonal_deg=0,
                robust=n // period >= 5,
            ).fit()
            return np.asarray(r.seasonal), np.asarray(r.resid)
        slope, intercept, _, _ = stats.theilslopes(y, t)
        trend = intercept + slope * t
        prof = np.array([(y - trend)[k::period].mean() for k in range(period)])
        if period >= 24:
            prof = np.convolve(
                np.concatenate([prof[-2:], prof, prof[:2]]),
                np.ones(5) / 5,
                mode="valid",
            )
        seasonal = np.tile(prof - prof.mean(), n // period + 1)[:n]
        return seasonal, y - trend - seasonal

    def check_series_match(self, f: Finding) -> tuple[np.ndarray, list[Check]]:
        y = self.series_values(f)
        claimed = np.array(
            f.chart["values"] if "dates" in f.chart else f.stats["values"], dtype=float
        )
        ok = np.isfinite(y)
        match = bool(ok.any()) and np.allclose(y[ok], claimed[ok], rtol=1e-9, atol=1e-9)
        # interpolated mean buckets have no SQL counterpart; everything that exists must agree
        return np.where(ok, y, claimed), [
            Check(
                "bucket values",
                float(np.nansum(claimed)),
                float(np.nansum(y[ok])) + float(np.nansum(claimed[~ok])),
                match,
            )
        ]

    def verify_trend(self, f: Finding) -> list[Check]:
        y, checks = self.check_series_match(f)
        period = {"day": 7, "week": 52, "month": 12, "quarter": 4}.get(f.stats["grain"])
        adj = y
        if f.stats["seasonally_adjusted"]:
            seasonal, _ = self._decompose(y, period)
            adj = y - seasonal
        n = len(adj)
        slope, intercept, _, _ = stats.theilslopes(adj, np.arange(n))
        start, end = intercept, intercept + slope * (n - 1)
        mean_level = float(np.mean(np.abs(adj)))
        base = (
            start
            if abs(start) >= 0.25 * mean_level
            else math.copysign(max(mean_level, 1e-12), start or 1.0)
        )
        change = (end - start) / abs(base)
        checks += [
            Check(
                "start level",
                f.facts["start_level"].value,
                start,
                close(f.facts["start_level"].value, start),
            ),
            Check(
                "end level",
                f.facts["end_level"].value,
                end,
                close(f.facts["end_level"].value, end),
            ),
            Check(
                "change",
                f.facts["change"].value,
                abs(change),
                close(f.facts["change"].value, abs(change)),
            ),
            Check(
                "direction",
                float(f.stats["direction"] == "up"),
                float(change > 0),
                (change > 0) == (f.stats["direction"] == "up"),
            ),
        ]
        return checks

    def verify_seasonality(self, f: Finding) -> list[Check]:
        y, checks = self.check_series_match(f)
        period, cycles = f.stats["period"], f.stats["cycles"]
        seasonal, _ = self._decompose(y, period)
        profile = np.array([seasonal[k::period].mean() for k in range(period)])
        swing = float(profile.max() - profile.min()) / float(np.mean(np.abs(y)))
        block = block_size(
            period
        )  # the same averaging of neighbouring buckets as the detector
        yb = (
            y[: y.size // block * block].reshape(-1, block).mean(axis=1)
            if block > 1
            else y
        )
        pb = period // block
        d = np.diff(yb[: cycles * pb].reshape(cycles, pb), axis=1)
        rhos = [
            float(stats.spearmanr(d[i], d[j])[0])
            for i in range(cycles)
            for j in range(i + 1, cycles)
        ]
        rho = float(np.mean(rhos))
        checks += [
            Check(
                "cycle repeatability",
                f.stats["repeatability"],
                rho,
                close(f.stats["repeatability"], rho, 1e-9),
            ),
            Check(
                "swing",
                f.facts["swing"].value,
                swing,
                close(f.facts["swing"].value, swing),
            ),
            Check(
                "peak of profile",
                f.stats["profile_max"],
                float(profile.max()),
                close(f.stats["profile_max"], profile.max()),
            ),
        ]
        return checks

    def verify_shift(self, f: Finding) -> list[Check]:
        y, checks = self.check_series_match(f)
        period = {"day": 7, "week": 52, "month": 12, "quarter": 4}.get(f.stats["grain"])
        adj = y
        if f.chart.get("adjusted") is not None:
            seasonal, _ = self._decompose(y, period)
            adj = y - seasonal
        cut, w = f.stats["cut_index"], f.facts["window"].value
        before, after = (
            float(adj[cut - w : cut].mean()),
            float(adj[cut : cut + w].mean()),
        )
        checks += [
            Check(
                "before level",
                f.facts["before"].value,
                before,
                close(f.facts["before"].value, before),
            ),
            Check(
                "after level",
                f.facts["after"].value,
                after,
                close(f.facts["after"].value, after),
            ),
        ]
        return checks

    def verify_anomaly(self, f: Finding) -> list[Check]:
        y, checks = self.check_series_match(f)
        period = {"day": 7, "week": 52, "month": 12, "quarter": 4}.get(f.stats["grain"])
        _, resid = self._decompose(y, period)
        med = float(np.median(resid))
        mad = float(stats.median_abs_deviation(resid, scale="normal"))
        mad = max(mad, f.stats["noise"])
        z = (resid - med) / mad
        idx = max(f.stats["flagged"], key=lambda i: abs(z[i]))
        checks += [
            Check(
                "value",
                f.facts["value"].value,
                float(y[idx]),
                close(f.facts["value"].value, y[idx]),
            ),
            Check(
                "robust z",
                abs(f.stats["robust_z"]),
                abs(float(z[idx])),
                close(abs(f.stats["robust_z"]), abs(z[idx]), 1e-5),
            ),
            Check(
                "expected",
                f.facts["expected"].value,
                float(y[idx] - resid[idx] + med),
                close(f.facts["expected"].value, y[idx] - resid[idx] + med),
            ),
        ]
        return checks

    def verify_latest(self, f: Finding) -> list[Check]:
        y, checks = self.check_series_match(f)
        v, prev = float(y[-1]), float(y[-2])
        checks += [
            Check(
                "latest value",
                f.facts["value"].value,
                v,
                close(f.facts["value"].value, v),
            ),
            Check(
                "previous value",
                f.stats["previous"],
                prev,
                close(f.stats["previous"], prev),
            ),
        ]
        if f.stats.get("year_ago") is not None:
            back = {"week": 52, "month": 12, "quarter": 4, "year": 1}[f.stats["grain"]]
            checks.append(
                Check(
                    "year-ago value",
                    f.stats["year_ago"],
                    float(y[-1 - back]),
                    close(f.stats["year_ago"], y[-1 - back]),
                )
            )
        return checks

    # ---- cross-sectional ---------------------------------------------------------------------
    def _totals(self, by: str, measure: str) -> list[tuple]:
        total = "count(*)" if measure == "rows" else f"sum({q(measure)})"
        return self.all(
            f"SELECT {q(by)}, {total} AS t FROM data WHERE {q(by)} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC, 1"
        )

    def verify_concentration(self, f: Finding) -> list[Check]:
        by, measure = f.columns
        vals = np.clip(
            np.array([float(t) for _, t in self._totals(by, measure)]), 0, None
        )
        shares = vals / vals.sum()
        n = shares.size
        if f.template == "concentration.entity":
            top_n = max(1, math.ceil(0.2 * n))
            share = float(shares[:top_n].sum())
            return [
                Check("groups", f.facts["n"].value, n, int(f.facts["n"].value) == n),
                Check(
                    "top share",
                    f.facts["share"].value,
                    share,
                    close(f.facts["share"].value, share),
                ),
            ]
        k = min(3, n - 1)
        return [
            Check(
                "top share",
                f.facts["share"].value,
                float(shares[0]),
                close(f.facts["share"].value, shares[0]),
            ),
            Check(
                "top-k share",
                f.facts["share_k"].value,
                float(shares[:k].sum()),
                close(f.facts["share_k"].value, shares[:k].sum()),
            ),
        ]

    def verify_group(self, f: Finding) -> list[Check]:
        dim, measure = f.columns
        rows = self.all(
            f"SELECT CAST({q(dim)} AS VARCHAR), CAST({q(measure)} AS DOUBLE) FROM data "
            f"WHERE {q(dim)} IS NOT NULL AND {q(measure)} IS NOT NULL"
        )
        by: dict[str, list[float]] = {}
        for g, v in rows:
            if math.isfinite(v):
                by.setdefault(g, []).append(v)
        by = {
            g: np.array(v)
            for g, v in by.items()
            if len(v) >= f.stats.get("min_rows", 20)
        }
        med = {g: float(np.median(v)) for g, v in by.items()}
        hi_med, lo_med = med[f.facts["hi"].value], med[f.facts["lo"].value]
        h, _ = stats.kruskal(*by.values())
        n, k = sum(v.size for v in by.values()), len(by)
        eps = max(0.0, (h - k + 1) / (n - k))
        return [
            Check(
                "high median",
                f.facts["hi_med"].value,
                hi_med,
                close(f.facts["hi_med"].value, hi_med),
            ),
            Check(
                "low median",
                f.facts["lo_med"].value,
                lo_med,
                close(f.facts["lo_med"].value, lo_med),
            ),
            Check(
                "highest is highest",
                hi_med,
                max(med.values()),
                hi_med >= max(med.values()),
            ),
            Check(
                "effect size",
                f.facts["share"].value,
                eps,
                close(f.facts["share"].value, eps, 1e-5),
            ),
        ]

    def verify_correlation(self, f: Finding) -> list[Check]:
        a, b = f.columns
        rows = self.all(
            f"SELECT CAST({q(a)} AS DOUBLE), CAST({q(b)} AS DOUBLE) FROM data WHERE {q(a)} IS NOT NULL AND {q(b)} IS NOT NULL"
        )
        xy = np.array(rows, dtype=float)
        rho = float(stats.spearmanr(xy[:, 0], xy[:, 1])[0])
        return [
            Check(
                "rank correlation",
                abs(f.stats["rho"]),
                abs(rho),
                close(abs(f.stats["rho"]), abs(rho), 1e-9),
            ),
            Check(
                "sign",
                float(f.stats["rho"] > 0),
                float(rho > 0),
                (rho > 0) == (f.stats["rho"] > 0),
            ),
            Check(
                "rows",
                f.facts["n"].value,
                len(rows),
                int(f.facts["n"].value) == len(rows),
            ),
        ]

    def verify_distribution(self, f: Finding) -> list[Check]:
        c = f.columns[0]
        med = float(self.one(f"SELECT median({q(c)}) FROM data"))
        mean = float(self.one(f"SELECT avg({q(c)}) FROM data"))
        checks = [
            Check(
                "median",
                f.facts["median"].value,
                med,
                close(f.facts["median"].value, med),
            ),
            Check(
                "mean", f.facts["mean"].value, mean, close(f.facts["mean"].value, mean)
            ),
        ]
        if "top_share" in f.facts:
            n = int(self.one(f"SELECT count({q(c)}) FROM data"))
            take = max(1, n // 100)
            top = float(
                self.one(
                    f"SELECT sum(v) FROM (SELECT {q(c)} AS v FROM data WHERE {q(c)} IS NOT NULL ORDER BY 1 DESC LIMIT {take})"
                )
            )
            total = float(self.one(f"SELECT sum({q(c)}) FROM data"))
            checks.append(
                Check(
                    "top 1% share",
                    f.facts["top_share"].value,
                    top / total,
                    close(f.facts["top_share"].value, top / total),
                )
            )
        return checks

    def verify_quality(self, f: Finding) -> list[Check]:
        n = int(self.one("SELECT count(*) FROM data"))
        if f.id == "quality:duplicates":
            dup = n - int(
                self.one("SELECT count(*) FROM (SELECT DISTINCT * FROM data)")
            )
            return [
                Check(
                    "duplicates",
                    f.facts["n"].value,
                    dup,
                    int(f.facts["n"].value) == dup,
                )
            ]
        if f.id == "quality:missing":
            worst = max(f.stats["columns"], key=f.stats["columns"].get)
            share = (
                int(self.one(f"SELECT count(*) FROM data WHERE {q(worst)} IS NULL")) / n
            )
            return [
                Check(
                    "worst missing share",
                    f.facts["share"].value,
                    share,
                    close(f.facts["share"].value, share),
                )
            ]
        if f.id == "quality:gaps":
            t = f"CAST({q(self.time)} AS DATE)"
            span = int(
                self.one(f"SELECT date_diff('day', min({t}), max({t})) + 1 FROM data")
            )
            gaps = span - int(self.one(f"SELECT count(DISTINCT {t}) FROM data"))
            return [
                Check(
                    "gap days",
                    f.facts["n"].value,
                    gaps,
                    int(f.facts["n"].value) == gaps,
                )
            ]
        if f.id.startswith("quality:negative"):
            c = f.columns[0]
            neg = int(self.one(f"SELECT count(*) FROM data WHERE {q(c)} < 0"))
            return [
                Check(
                    "negative rows",
                    f.facts["n"].value,
                    neg,
                    int(f.facts["n"].value) == neg,
                )
            ]
        if f.id.startswith("quality:identity"):
            a, b, c2 = f.columns
            bad = int(
                self.one(
                    f"SELECT count(*) FROM data WHERE {q(a)} IS NOT NULL AND {q(b)} IS NOT NULL AND {q(c2)} IS NOT NULL AND {q(a)} <> {q(b)} + {q(c2)}"
                )
            )
            return [Check("rows breaking the identity", 0, bad, bad == 0)]
        return []

    def verify_overview(self, f: Finding) -> list[Check]:
        rows = int(self.one("SELECT count(*) FROM data"))
        checks = [
            Check(
                "rows", f.facts["rows"].value, rows, int(f.facts["rows"].value) == rows
            )
        ]
        if self.time:
            t = f"CAST({q(self.time)} AS DATE)"
            lo, hi = (
                str(x) for x in self.all(f"SELECT min({t}), max({t}) FROM data")[0]
            )
            checks.append(
                Check(
                    "date range",
                    1.0,
                    float(lo == f.facts["start"].value and hi == f.facts["end"].value),
                    lo == f.facts["start"].value and hi == f.facts["end"].value,
                )
            )
        return checks

    def verify(self, f: Finding) -> list[Check]:
        kind = {"trend": "trend", "seasonality": "seasonality", "shift": "shift", "anomaly": "anomaly", "latest": "latest",
                "concentration": "concentration", "group": "group", "correlation": "correlation",
                "distribution": "distribution", "quality": "quality", "overview": "overview"}[f.type]  # fmt: skip
        return getattr(self, f"verify_{kind}")(f)


def verify_findings(
    profile: Profile, findings: list[Finding]
) -> dict[str, list[Check]]:
    """Checks for each finding id; a failing check means the claim does not reproduce."""
    v = Verifier(profile)
    try:
        out = {}
        for f in findings:
            try:
                out[f.id] = v.verify(f)
            except Exception as exc:  # noqa: BLE001 (a claim that cannot even be recomputed counts as failed)
                out[f.id] = [
                    Check(
                        f"recompute failed: {type(exc).__name__}: {exc}",
                        0.0,
                        1.0,
                        False,
                    )
                ]
        return out
    finally:
        v.close()


def failed(checks: list[Check]) -> list[Check]:
    return [c for c in checks if not c.ok]
