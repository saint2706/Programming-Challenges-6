"""Benchmarks and failure-mode demonstrations for the detectors.

    uv run python benchmark.py                       # everything except MSTL (~2 min)
    uv run python benchmark.py --only masking fpr    # chosen sections
    uv run python benchmark.py --mstl --write benchmark_results.md

Sections: samuelson, masking, injected, wrong-period, rolling, fpr, stl-calibration,
period-detection, nab.
Every section is a function returning a markdown table so tests can assert on the
numbers that the README quotes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from anomaly_detector import analyze, load_series, load_windows
from detectors import (
    METHODS,
    DetectorError,
    Params,
    detect_period,
    generalized_esd,
    modified_z_scores,
    run_method,
    stl_detect,
    z_scores,
)
from evaluate import event_counts, point_counts
from scipy import stats
from synth import (
    base_series,
    clean_noise,
    inject_level_shift,
    inject_spikes,
    make_scenarios,
)

HERE = Path(__file__).parent
SAMPLE = HERE / "sample_data"
TOL = 2  # events count as found if a flag lands within 2 samples of the truth


def md_table(header: list[str], rows: list[list[object]]) -> str:
    def cell(v: object) -> str:
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def _run(method: str, x, period: int | None, params: Params | None = None):
    """Run a method, returning None when it can't apply (STL without a period)."""
    try:
        return run_method(
            method, x, periods=[period] if period else None, params=params
        )
    except DetectorError:
        return None


# ------------------------------------------------------------------ samuelson


def samuelson_table() -> str:
    """Z-score with the classic threshold of 3 cannot fire at all for n <= 10.

    Samuelson's inequality: for a sample of n points, max |z| <= (n-1)/sqrt(n)
    where z uses the sample standard deviation, however extreme the outlier.
    """
    rows = []
    for n in (5, 8, 10, 11, 15, 30):
        x = np.zeros(n)
        x[0] = 1e9  # about as extreme as an outlier gets
        bound = (n - 1) / np.sqrt(n)
        x = x + np.random.default_rng(n).normal(0, 1, n)
        z = float(z_scores(x).max())
        mad_flag = bool(modified_z_scores(x).max() > 3.5)
        rows.append(
            [n, bound, z, "yes" if z > 3 else "**no**", "yes" if mad_flag else "no"]
        )
    return md_table(
        [
            "n",
            "max possible abs(z)",
            "abs(z) of a 1e9 outlier",
            "Z-score (3) flags it",
            "MAD (3.5) flags it",
        ],
        rows,
    )


# -------------------------------------------------------------------- masking


def masking_table(seeds: int = 20) -> str:
    """10 outliers near +8 among 100 N(0,1) points: they inflate the sd that judges them."""
    methods = ["zscore", "iqr", "mad", "gesd (k=5)", "gesd (k=20)", "rolling-mad"]
    acc: dict[str, list[tuple[float, float, float]]] = {m: [] for m in methods}
    sd_ratio = []
    for seed in range(seeds):
        sc = make_scenarios(seed)[4]
        x = sc.values
        clean = x[~sc.truth]
        sd_ratio.append(x.std(ddof=1) / clean.std(ddof=1))
        for m in methods:
            if m.startswith("gesd"):
                k = int(m.split("=")[1].rstrip(")"))
                det = run_method("gesd", x, params=Params(max_outliers=k))
            elif m == "rolling-mad":
                det = run_method(m, x, params=Params(window=30))
            else:
                det = run_method(m, x)
            c = point_counts(det.flags, sc.truth)
            acc[m].append((c.precision, c.recall, c.f1))
    rows = []
    for m in methods:
        a = np.mean(acc[m], axis=0)
        rows.append([m, a[0], a[1], a[2]])
    table = md_table(["method", "precision", "recall", "F1"], rows)
    return (
        table
        + f"\n\nThe 10% outlier cluster inflates the sample sd by x{np.mean(sd_ratio):.2f} on average."
    )


# -------------------------------------------------------------------- injected


INJECTED_METHODS = (
    "zscore",
    "iqr",
    "mad",
    "gesd",
    "rolling-z",
    "rolling-mad",
    "stl-mad",
    "stl-z",
    "stl-iqr",
    "stl-gesd",
)


def injected_scores(
    seeds: int = 10, methods: tuple[str, ...] = INJECTED_METHODS
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """scenario -> method -> mean (event precision, event recall, event F1)."""
    out: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
    for seed in range(seeds):
        for sc in make_scenarios(seed):
            for m in methods:
                det = _run(m, sc.values, sc.period, Params(window=48))
                if det is None:
                    continue
                c = event_counts(det.flags, sc.truth, TOL)
                out.setdefault(sc.name, {}).setdefault(m, []).append(
                    (c.precision, c.recall, c.f1)
                )
    return {
        s: {m: tuple(np.mean(v, axis=0)) for m, v in d.items()} for s, d in out.items()
    }


def injected_table(seeds: int = 10) -> str:
    scores = injected_scores(seeds)
    names = list(scores)
    rows = []
    for m in INJECTED_METHODS:
        row: list[object] = [m]
        for s in names:
            v = scores[s].get(m)
            row.append("-" if v is None else f"{v[2]:.2f} ({v[0]:.2f}/{v[1]:.2f})")
        rows.append(row)
    return md_table(["method (event F1, precision/recall)", *names], rows)


# ----------------------------------------------------------------- wrong period


def wrong_period_table(seeds: int = 8) -> str:
    """STL on the 'seasonal spikes' scenario (true period 24) with the wrong period."""
    rows = []
    for period in (24, 48, 12, 18, 23, 25, 30, 36, 7):
        f1s, flags, recalls = [], [], []
        for seed in range(seeds):
            sc = make_scenarios(seed)[0]
            det = stl_detect(sc.values, period, "z")
            c = event_counts(det.flags, sc.truth, TOL)
            f1s.append(c.f1)
            recalls.append(c.recall)
            flags.append(det.count)
        note = {24: "true period", 48: "a multiple of the truth"}.get(period, "")
        rows.append(
            [period, note, np.mean(f1s), np.mean(recalls), float(np.mean(flags))]
        )
    return md_table(
        [
            "period given to STL",
            "",
            "event F1 (stl-z)",
            "event recall",
            "samples flagged (10 true spikes)",
        ],
        rows,
    )


# --------------------------------------------------------------------- rolling


def rolling_table(seeds: int = 20) -> str:
    """A burst of 10 outliers and a sustained 60-sample level shift, window 50."""
    rows = []
    for label, burst, shift in (
        ("burst of 10 outliers (+8)", 10, 0),
        ("level shift of 60 samples (+8)", 0, 60),
    ):
        acc: dict[str, list[float]] = {"rolling-z": [], "rolling-mad": [], "mad": []}
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            x = base_series(600, None, rng=rng, level=0.0)
            if burst:
                truth = inject_level_shift(x, 300, burst, 8.0)
            else:
                truth = inject_level_shift(x, 300, shift, 8.0)
            for m, hits in acc.items():
                det = run_method(m, x, params=Params(window=50))
                hits.append(float((det.flags & truth).sum() / truth.sum()))
        rows.append(
            [
                label,
                *[float(np.mean(acc[m])) for m in ("rolling-z", "rolling-mad", "mad")],
            ]
        )
    return md_table(
        [
            "scenario",
            "rolling-z: share of anomalous samples flagged",
            "rolling-mad",
            "global mad",
        ],
        rows,
    )


# ------------------------------------------------------------------------- fpr


def fpr_table(trials: int = 400, n: int = 1000) -> str:
    """False-positive rate on i.i.d. N(0,1) (no anomalies exist), vs theory."""
    window = 50
    theory = {
        "zscore": 2 * stats.norm.sf(3.0),
        "iqr": 2 * stats.norm.sf(stats.norm.ppf(0.75) + 1.5 * 2 * stats.norm.ppf(0.75)),
        "mad": 2 * stats.norm.sf(3.5),
        "rolling-z": 2 * stats.t.sf(3.0 / np.sqrt(1 + 1 / window), window - 1),
    }
    flags = {m: 0 for m in ("zscore", "iqr", "mad", "rolling-z", "rolling-mad")}
    for seed in range(trials):
        x = clean_noise(n, seed)
        for m in flags:
            det = run_method(m, x, params=Params(window=window))
            flags[m] += int(
                det.flags[window:].sum() if m.startswith("rolling") else det.flags.sum()
            )
    rows = []
    for m, count in flags.items():
        denom = trials * (n - window) if m.startswith("rolling") else trials * n
        rows.append(
            [
                m,
                f"{count / denom:.5f}",
                f"{theory[m]:.5f}" if m in theory else "no closed form",
            ]
        )
    # Generalized ESD: the guarantee is per *series*, not per point.
    alarms = 0
    esd_trials = 2000
    for seed in range(esd_trials):
        x = clean_noise(200, 10_000 + seed)
        alarms += int(generalized_esd(x, 10, 0.05).n_outliers > 0)
    rows.append(
        [
            "gesd (alpha=0.05, n=200): series with >=1 flag",
            f"{alarms / esd_trials:.4f}",
            "<= 0.05",
        ]
    )
    return md_table(
        [
            "method",
            f"measured rate on clean N(0,1) ({trials} series of {n})",
            "theoretical",
        ],
        rows,
    )


# --------------------------------------------------------------- stl calibration


def stl_calibration_table(seeds: int = 6) -> str:
    """STL detectors on a clean seasonal series: how many false flags per 1200 samples?"""
    rows = []
    for robust in (True, False):
        for seasonal in (7, 13, 25):
            kurt, ratio = [], []
            flags = {"mad": [], "z": [], "iqr": [], "gesd": []}
            for seed in range(seeds):
                x = base_series(1200, 24, rng=np.random.default_rng(seed))
                p = Params(stl_robust=robust, stl_seasonal=seasonal)
                for s, counts in flags.items():
                    det = stl_detect(x, 24, s, p)
                    counts.append(det.count)
                resid = det.components["resid"]
                kurt.append(stats.kurtosis(resid))
                ratio.append(
                    np.median(np.abs(resid - np.median(resid)))
                    / 0.6745
                    / resid.std(ddof=1)
                )
            rows.append(
                [
                    "robust" if robust else "plain",
                    seasonal,
                    float(np.mean(kurt)),
                    float(np.mean(ratio)),
                    *[float(np.mean(flags[s])) for s in ("mad", "z", "iqr", "gesd")],
                ]
            )
    theory = [
        "Gaussian: 0",
        "1",
        f"{1200 * 2 * stats.norm.sf(3.5):.1f}",
        f"{1200 * 2 * stats.norm.sf(3.0):.1f}",
        f"{1200 * 2 * stats.norm.sf(2.698):.1f}",
        "<0.05/series",
    ]
    rows.append(["theory (clean Gaussian)", "", *theory[:2], *theory[2:]])
    return md_table(
        [
            "STL fits",
            "seasonal window",
            "residual excess kurtosis",
            "MAD-sigma / sd",
            "stl-mad flags",
            "stl-z",
            "stl-iqr",
            "stl-gesd",
        ],
        rows,
    )


# --------------------------------------------------------------- period detection


def period_detection_table(cases: int = 60) -> str:
    """Recover the period of seasonal series with noise, trend and spikes; reject noise."""
    rng = np.random.default_rng(123)
    exact = off_by_one = multiple = wrong = 0
    for _ in range(cases):
        period = int(rng.integers(6, 120))
        n = int(period * rng.integers(8, 25))
        x = base_series(
            n,
            period,
            amplitude=float(rng.uniform(3, 12)),
            noise_sd=float(rng.uniform(0.3, 2.0)),
            trend_per_step=float(rng.uniform(-30, 30)) / n,
            rng=rng,
        )
        inject_spikes(x, rng.integers(0, n, max(1, n // 200)), 15.0, rng)
        est = detect_period(x)
        if est is None:
            wrong += 1
        elif est.period == period:
            exact += 1
        elif abs(est.period - period) == 1:
            off_by_one += 1
        elif est.period % period == 0:
            multiple += 1
        else:
            wrong += 1
    noise_hits = sum(
        detect_period(clean_noise(int(rng.integers(200, 3000)), 5000 + i)) is not None
        for i in range(200)
    )
    return (
        md_table(
            ["outcome", f"count (of {cases} seasonal series)"],
            [
                ["exact period", exact],
                ["off by one", off_by_one],
                ["multiple of the true period", multiple],
                ["wrong / none", wrong],
            ],
        )
        + f"\n\nWhite noise called seasonal: {noise_hits} of 200 series."
    )


# -------------------------------------------------------------------------- nab

NAB_FILES = (
    "nyc_taxi",
    "ambient_temperature_system_failure",
    "ec2_cpu_utilization_24ae8d",
    "art_daily_jumpsdown",
)


def nab_table(mstl: bool = False) -> str:
    from anomaly_detector import evaluate_detection

    rows = []
    for name in NAB_FILES:
        path = SAMPLE / f"{name}.csv"
        ts = load_series(path)
        windows = load_windows(SAMPLE / "labels.json", path.name)
        t0 = time.time()
        an = analyze(ts, list(METHODS), "auto", Params())
        runs_: list[tuple[str, object]] = list(an.detections.items())
        if mstl and name == "nyc_taxi":
            an2 = analyze(ts, ["stl-mad", "stl-z", "stl-gesd"], [48, 336], Params())
            runs_ += [(f"{m} (MSTL 48+336)", d) for m, d in an2.detections.items()]
        for m, det in runs_:
            ev = evaluate_detection(ts, det, windows)
            rows.append(
                [
                    name,
                    m,
                    f"{ev.nab.windows_hit}/{ev.nab.windows_total}",
                    det.count,
                    ev.nab.false_positive_events,
                    ev.point_f1,
                    f"{ev.nab.normalized:.1f}",
                ]
            )
        for m, why in an.skipped.items():
            rows.append(
                [name, m, "-", "-", "-", "-", f"skipped ({why.split(' (')[0]})"]
            )
        print(
            f"  [{name}: {time.time() - t0:.0f}s, period {an.periods or 'none'}]",
            file=sys.stderr,
        )
    return md_table(
        [
            "series",
            "method",
            "windows hit",
            "samples flagged",
            "false-alarm events",
            "point F1",
            "NAB score",
        ],
        rows,
    )


SECTIONS = {
    "samuelson": (
        "Z-score cannot fire for n <= 10 (Samuelson's inequality)",
        lambda a: samuelson_table(),
    ),
    "masking": (
        "Masking: outliers hide themselves from the Z-score",
        lambda a: masking_table(),
    ),
    "injected": (
        "Injected anomalies: event-level scores by scenario",
        lambda a: injected_table(),
    ),
    "wrong-period": (
        "STL with the wrong seasonal period",
        lambda a: wrong_period_table(),
    ),
    "rolling": (
        "Rolling windows adapt to what they should flag",
        lambda a: rolling_table(),
    ),
    "fpr": ("False-positive rate on clean noise vs theory", lambda a: fpr_table()),
    "stl-calibration": (
        "How calibrated is the STL residual?",
        lambda a: stl_calibration_table(),
    ),
    "period-detection": (
        "Automatic seasonal-period detection",
        lambda a: period_detection_table(),
    ),
    "nab": (
        "Real data: labelled windows from the Numenta Anomaly Benchmark",
        lambda a: nab_table(a.mstl),
    ),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--only", nargs="+", choices=list(SECTIONS), help="run just these sections"
    )
    p.add_argument(
        "--mstl",
        action="store_true",
        help="add MSTL rows for nyc_taxi in the nab section (~30 s each)",
    )
    p.add_argument(
        "--write", type=Path, help="write the markdown output to this file too"
    )
    args = p.parse_args(argv)

    out: list[str] = []
    for key in args.only or list(SECTIONS):
        title, fn = SECTIONS[key]
        t0 = time.time()
        body = fn(args)
        block = f"## {title}\n\n{body}\n"
        print(block)
        print(f"  ({key}: {time.time() - t0:.1f}s)\n", file=sys.stderr)
        out.append(block)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text("\n".join(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
