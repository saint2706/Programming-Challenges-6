"""The experiment: tune two settings on the validation window, then report on the test window.

Stage 1 (validation): fit on sales before ``val_start``, score the validation window, and choose the
market-index extrapolation and the way comps are combined. Stage 2 (test): refit on everything
before ``test_start`` with those choices frozen and score the test window once.

Headline targets are *first sales* of a property (``n_prior == 0``): a house's own earlier sale is
a near-perfect comp that would flatter the similarity measures, so it is analysed separately.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import product

import lightgbm as lgb
import numpy as np
import polars as pl

from comp_finder import data, evaluate
from comp_finder.engine import METHODS, Engine, Settings
from comp_finder.hedonic import EXTRAPOLATIONS
from comp_finder.paths import project_root
from comp_finder.trees import N_TREES, PARAMS

HEADLINE_METHODS = ("knn", "hedonic", "leaf")  # ``ridge`` is an ablation of the weights
KS = (3, 5, 8, 12)
KMAX = 15  # enough for k=12 and for the 3k candidates the bracket-aware selection looks through
DISTANCE_BANDS_KM = (8.0, 20.0)


@dataclass
class Result:
    windows: data.Windows
    settings: Settings
    metrics: dict
    engine: Engine
    targets: np.ndarray
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


def first_sales(engine: Engine, lo: int, hi: int) -> np.ndarray:
    rows = np.arange(lo, hi)
    return rows[engine.sales["n_prior"].to_numpy()[rows] == 0]


def mdape(est: np.ndarray, actual: np.ndarray) -> float:
    return float(np.median(np.abs(est - actual) / actual))


def tune(sales: pl.DataFrame, n_fit: int, n_val: int) -> tuple[Settings, list[dict]]:
    """Pick (extrapolate, combine) by mean MdAPE over the headline methods on the validation window."""
    engine = Engine(sales, n_fit)
    rows = first_sales(engine, n_fit, n_fit + n_val)
    actual = engine.price[rows]
    comps = {m: engine.rank(m, rows, KMAX) for m in HEADLINE_METHODS}
    grid = []
    for extrapolate, combine in product(EXTRAPOLATIONS, ("median", "mean")):
        s = Settings(extrapolate=extrapolate, combine=combine)
        scores = {
            m: mdape(engine.estimate(rows, comps[m], s).price, actual) for m in comps
        }
        grid.append(
            {
                "extrapolate": extrapolate,
                "combine": combine,
                "mean_mdape": float(np.mean(list(scores.values()))),
                **{f"mdape_{m}": v for m, v in scores.items()},
            }
        )
    best = min(grid, key=lambda g: g["mean_mdape"])
    return Settings(extrapolate=best["extrapolate"], combine=best["combine"]), grid


def avm_with_zipcode(
    engine: Engine, rows: np.ndarray, settings: Settings
) -> np.ndarray:
    """The GBM reference with zipcode as a categorical feature next to lat/long."""
    codes = {z: i for i, z in enumerate(engine.hedonic.zips)}
    zc = np.array([codes.get(z, -1) for z in engine.zip], dtype=np.float64)
    x = np.column_stack([engine.z, zc])
    fit = slice(0, engine.n_fit)
    y = engine.log_price[fit] - engine.hedonic.index(engine.month[fit])
    ds = lgb.Dataset(x[fit], y, categorical_feature=[x.shape[1] - 1])
    booster = lgb.train({**PARAMS}, ds, num_boost_round=N_TREES)
    return np.exp(
        booster.predict(x[rows])
        + engine.hedonic.index(engine.month[rows], settings.extrapolate)
    )


def run(windows: data.Windows | None = None, reps: int = 1000, log=print) -> Result:
    windows = windows or data.Windows()
    sales = data.load()
    n_fit, n_val, n_test = data.window_counts(sales, windows)
    t0 = time.time()

    settings, grid = tune(sales, n_fit, n_val)
    log(
        f"validation picked extrapolate={settings.extrapolate} combine={settings.combine}"
    )

    n_train = n_fit + n_val
    engine = Engine(sales, n_train)
    rows = first_sales(engine, n_train, sales.height)
    actual = engine.price[rows]
    idx = evaluate.boot_indices(len(rows), reps)
    out: dict = {
        "data": {
            "sales": sales.height,
            "first": str(sales["sold"].min()),
            "last": str(sales["sold"].max()),
            "repeat_sales": int((sales["n_prior"] > 0).sum()),
            "fit": n_fit,
            "validation": n_val,
            "test": n_test,
            "test_first_sales": len(rows),
        },
        "windows": {
            "val_start": str(windows.val_start),
            "test_start": str(windows.test_start),
        },
        "settings": {
            "k": settings.k,
            "extrapolate": settings.extrapolate,
            "combine": settings.combine,
        },
        "validation_grid": grid,
    }
    arrays: dict[str, np.ndarray] = {"actual": actual, "rows": rows}

    comps = {m: engine.rank(m, rows, KMAX) for m in METHODS}
    estimates = {m: engine.estimate(rows, comps[m], settings) for m in METHODS}
    preds = {m: estimates[m].price for m in METHODS}
    preds["avm"] = engine.avm(rows, settings.extrapolate)
    arrays |= {f"pred_{m}": p for m, p in preds.items()}
    assert not any(np.isnan(p).any() for p in preds.values()), (
        "a subject has no earlier comps"
    )

    out["main"] = {
        m: {
            **evaluate.summarize(p, actual),
            "ci": evaluate.interval(p, actual, idx),
        }
        for m, p in preds.items()
    }
    pairs = [
        ("hedonic", "knn"),
        ("leaf", "knn"),
        ("leaf", "hedonic"),
        ("leaf", "avm"),
        ("hedonic", "avm"),
        ("knn", "avm"),
    ]
    out["paired"] = {
        f"{a}-{b}": evaluate.paired_difference(preds[a], preds[b], actual, idx)
        for a, b in pairs
    }

    out["adjust_ablation"] = {
        m: {
            mode: mdape(
                engine.estimate(
                    rows, comps[m], Settings(**{**vars(settings), "adjust": mode})
                ).price,
                actual,
            )
            for mode in ("none", "time", "full", "gbm")
        }
        for m in METHODS
    }
    out["k_sweep"] = {
        m: {
            str(k): evaluate.summarize(
                engine.estimate(
                    rows, comps[m], Settings(**{**vars(settings), "k": k})
                ).price,
                actual,
            )["mdape"]
            for k in KS
        }
        for m in METHODS
    }
    out["extrapolation"] = {
        e: {
            m: mdape(
                engine.estimate(
                    rows, comps[m], Settings(**{**vars(settings), "extrapolate": e})
                ).price,
                actual,
            )
            for m in HEADLINE_METHODS
        }
        | {"avm": mdape(engine.avm(rows, e), actual)}
        for e in EXTRAPOLATIONS
    }

    pool_fit = {m: engine.rank(m, rows, KMAX, pool="fit") for m in HEADLINE_METHODS}
    out["pool"] = {
        m: {
            "prior": mdape(preds[m], actual),
            "fit": mdape(engine.estimate(rows, pool_fit[m], settings).price, actual),
        }
        for m in HEADLINE_METHODS
    }

    bracket = Settings(**{**vars(settings), "selection": "bracket"})
    out["bracketing"] = {}
    for m in HEADLINE_METHODS:
        top, br = estimates[m], engine.estimate(rows, comps[m], bracket)
        out["bracketing"][m] = {
            "bracketed_share_top": float(top.bracketed.mean()),
            "bracketed_share_selected": float(br.bracketed.mean()),
            "mdape_top": mdape(top.price, actual),
            "mdape_selected": mdape(br.price, actual),
            "mdape_when_bracketed": mdape(
                top.price[top.bracketed], actual[top.bracketed]
            ),
            "mdape_when_not": mdape(top.price[~top.bracketed], actual[~top.bracketed]),
        }

    km = np.hypot(*(engine.xy[rows].T))
    bands = np.digitize(km, DISTANCE_BANDS_KM)
    tercile = np.digitize(preds["avm"], np.quantile(preds["avm"], [1 / 3, 2 / 3]))
    seg: dict[str, dict] = {}
    for name, labels, names in (
        ("distance to downtown", bands, ("< 8 km", "8-20 km", "> 20 km")),
        (
            "price tier (by AVM estimate)",
            tercile,
            ("low third", "middle third", "top third"),
        ),
    ):
        seg[name] = {
            label: {
                "n": int((labels == b).sum()),
                **{m: mdape(preds[m][labels == b], actual[labels == b]) for m in preds},
            }
            for b, label in enumerate(names)
        }
    out["segments"] = seg

    out["comp_quality"] = {}
    for m in METHODS:
        est = estimates[m]
        c = est.comps[:, : settings.k]
        dist = np.hypot(
            *(engine.xy[c] - engine.xy[rows][:, None, :]).transpose(2, 0, 1)
        )
        out["comp_quality"][m] = {
            "median_km": float(np.median(dist)),
            "same_zip": float((engine.zip[c] == engine.zip[rows][:, None]).mean()),
            "median_days_old": float(
                np.median(engine.sold_days[rows][:, None] - engine.sold_days[c])
            ),
            "median_gross_adjustment": float(np.median(est.gross[:, : settings.k])),
        }

    rep_all = np.arange(n_train, sales.height)
    rep = rep_all[engine.sales["n_prior"].to_numpy()[rep_all] > 0]
    if len(rep):
        act_r = engine.price[rep]
        out["repeat_sales"] = {
            "n": len(rep),
            "avm": mdape(engine.avm(rep, settings.extrapolate), act_r),
            **{
                f"{m}_{'with' if own else 'without'}_own_prior": mdape(
                    engine.predict(m, rep, settings, include_own=own).price, act_r
                )
                for m in ("knn", "leaf")
                for own in (False, True)
            },
        }

    out["zipcode"] = {
        "latlong_only": mdape(preds["avm"], actual),
        "with_zip_categorical": mdape(avm_with_zipcode(engine, rows, settings), actual),
    }

    avm_err = (preds["avm"] - actual) / actual
    months = engine.month[rows]
    out["monthly_bias"] = {
        int(mth): {
            "n": int((months == mth).sum()),
            "avm_median_error": float(np.median(avm_err[months == mth])),
        }
        for mth in np.unique(months)
    }
    out["market_index"] = {
        int(mth): {
            "fitted": bool(mth <= engine.hedonic.months[-1]),
            "flat": float(engine.hedonic.index(np.array([mth]), "flat")[0]),
            "trend": float(engine.hedonic.index(np.array([mth]), "trend")[0]),
        }
        for mth in range(int(engine.hedonic.months[0]), int(engine.month.max()) + 1)
    }

    sample = rows[:: max(1, len(rows) // 400)]
    timing = {}
    for m in METHODS:
        t = time.perf_counter()
        engine.predict(m, sample, settings)
        timing[m] = (time.perf_counter() - t) / len(sample) * 1000
    out["ms_per_subject"] = timing
    out["pool_size_at_test"] = {
        "first": int(engine.before[rows].min()),
        "last": int(engine.before[rows].max()),
    }
    out["gbm"] = {"trees": N_TREES, "num_leaves": PARAMS["num_leaves"]}
    out["seconds"] = time.time() - t0
    return Result(windows, settings, out, engine, rows, arrays)


def saved_settings() -> Settings:
    """The settings the last ``comps evaluate`` chose on validation, else the defaults."""
    try:
        saved = json.loads(
            (project_root() / "results" / "metrics.json").read_text("utf-8")
        )
        chosen = saved["settings"]
        return Settings(extrapolate=chosen["extrapolate"], combine=chosen["combine"])
    except (OSError, KeyError, ValueError):
        return Settings()


def build_engine(windows: data.Windows | None = None) -> tuple[Engine, Settings]:
    """The test-stage engine (fit on everything before ``test_start``) for the app and CLI."""
    windows = windows or data.Windows()
    sales = data.load()
    n_fit, n_val, _ = data.window_counts(sales, windows)
    return Engine(sales, n_fit + n_val), saved_settings()
