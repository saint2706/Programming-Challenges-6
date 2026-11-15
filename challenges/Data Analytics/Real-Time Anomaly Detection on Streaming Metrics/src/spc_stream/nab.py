"""Replay the vendored Numenta Anomaly Benchmark streams through each chart and score them."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from spc_stream.charts import make_chart
from spc_stream.monitor import Monitor
from spc_stream.paths import project_root

# name -> (labels.json key, season in points: one day at the stream's sampling rate, 0 = none)
STREAMS = {
    "nyc_taxi": ("realKnownCause/nyc_taxi.csv", 48),
    "ambient_temperature_system_failure": (
        "realKnownCause/ambient_temperature_system_failure.csv",
        24,
    ),
    "ec2_cpu_utilization_24ae8d": (
        "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
        0,
    ),
    "art_daily_jumpsdown": ("artificialWithAnomaly/art_daily_jumpsdown.csv", 288),
}
PROBATION = (
    0.15  # NAB's probationary share: the detector may learn on it, it is never scored
)
COOLDOWN = 12


def data_dir() -> Path:
    return project_root() / "sample_data"


def load(name: str) -> tuple[list[datetime], list[float], list[tuple[int, int]]]:
    """Timestamps, values and labelled windows (as inclusive index ranges) of one stream."""
    if name not in STREAMS:
        raise ValueError(f"unknown stream {name!r}; choose from {sorted(STREAMS)}")
    key, _ = STREAMS[name]
    with (data_dir() / f"{name}.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    ts = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    xs = [float(r["value"]) for r in rows]
    labels = json.loads((data_dir() / "labels.json").read_text())[key]
    windows = []
    for a, b in labels:
        lo, hi = datetime.fromisoformat(a), datetime.fromisoformat(b)
        idx = [i for i, t in enumerate(ts) if lo <= t <= hi]
        windows.append((idx[0], idx[-1]))
    return ts, xs, windows


def monitor_for(name: str, chart: str, prewhiten: bool, n: int) -> Monitor:
    season = STREAMS[name][1] if prewhiten else 0
    return Monitor(
        make_chart(chart),
        baseline_n=int(n * PROBATION),
        season=season,
        ar1=prewhiten,
        cooldown=COOLDOWN,
    )


@dataclass(frozen=True, slots=True)
class Score:
    stream: str
    chart: str
    prewhiten: bool
    windows_hit: int
    windows: int
    false_alarms: int
    false_per_1k: float
    mean_delay: float | None  # points from window start to the first alarm inside it


def score(name: str, chart: str, prewhiten: bool) -> Score:
    _, xs, windows = load(name)
    mon = monitor_for(name, chart, prewhiten, len(xs))
    alarms = [e.i for e in mon.run(xs) if e.alarm]
    inside = lambda i: any(a <= i <= b for a, b in windows)
    delays = []
    for a, b in windows:
        hits = [i for i in alarms if a <= i <= b]
        if hits:
            delays.append(hits[0] - a)
    false = sum(not inside(i) for i in alarms)
    scored = len(xs) - mon.baseline_n
    return Score(
        name,
        chart,
        prewhiten,
        len(delays),
        len(windows),
        false,
        round(1000 * false / scored, 2),
        round(sum(delays) / len(delays), 1) if delays else None,
    )


def benchmark() -> list[Score]:
    return [
        score(s, c, p)
        for s in STREAMS
        for c in ("shewhart", "ewma", "cusum")
        for p in (False, True)
    ]
