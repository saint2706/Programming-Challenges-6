"""The experiment: split by time, fit and compare models, and measure everything the report shows.

``run`` returns one JSON-able dict (``result``) plus the arrays the charts need. It writes nothing
unless asked to (``save``), so tests can run it on a synthetic table.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import date

import numpy as np
import polars as pl

from stockout_risk import data, evaluate, features, models
from stockout_risk.features import FEATURES, GROUPS, HEAVY_HOURS
from stockout_risk.paths import data_dir, results_dir

Log = Callable[[str], None]
LABEL_HOURS = (1, HEAVY_HOURS, 8)
DRIFT_FEATURES = [
    "oos_mean14",
    "product_heavy_rate",
    "oos_last4",
    "demand16",
    "discount_next",
]


def load_features(*, refresh: bool = False) -> pl.DataFrame:
    """The feature table of the real data, cached in ``data/features.parquet``."""
    path = data_dir() / "features.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    frame = features.build(data.load())
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return frame


def baselines(frame: pl.DataFrame, hours: int) -> dict[str, np.ndarray]:
    """Scores that need no model: yesterday, the same weekday last week, and two recent rates."""
    return {
        "yesterday": (frame["oos_t"] >= hours).to_numpy().astype(float),
        "same_weekday_last_week": (frame["oos_lag6"] >= hours).to_numpy().astype(float),
        "heavy_rate_7d": frame["heavy_rate7"].to_numpy(),
        "mean_oos_hours_14d": frame["oos_mean14"].to_numpy(),
    }


def _censoring(train: pl.DataFrame) -> list[dict]:
    """Sales on stockout days against the sales rate of the in-stock hours, scaled to a full day."""
    bins = [(0, 0), (1, 3), (4, 7), (8, 11), (12, 15)]
    rows = []
    for lo, hi in bins:
        part = train.filter(
            (pl.col("oos_t") >= lo) & (pl.col("oos_t") <= hi)
        ).drop_nulls("demand_rate")
        rows.append(
            {
                "oos_hours": f"{lo}" if lo == hi else f"{lo}-{hi}",
                "rows": part.height,
                "mean_sale": float(part["sale_t"].mean()),
                "mean_demand_16h": float(
                    (part["demand_rate"] * data.WINDOW_HOURS).mean()
                ),
            }
        )
    return rows


def _weekly(frames: dict[str, pl.DataFrame]) -> list[dict]:
    both = pl.concat(
        [
            f.select("label_day", "y").with_columns(pl.lit(n).alias("part"))
            for n, f in frames.items()
        ]
    )
    weekly = (
        both.with_columns(pl.col("label_day").dt.truncate("1w").alias("week"))
        .group_by("week", "part")
        .agg(pl.col("y").mean().alias("rate"), pl.len().alias("rows"))
        .sort("week", "part")
    )
    return [
        {
            "week": str(r["week"]),
            "part": r["part"],
            "rate": r["rate"],
            "rows": r["rows"],
        }
        for r in weekly.iter_rows(named=True)
    ]


def run(
    feat: pl.DataFrame,
    windows: models.Windows | None = None,
    *,
    hours: int = HEAVY_HOURS,
    ablate: bool = True,
    other_labels: bool = True,
    rounds: int = 600,
    log: Log = lambda *_: None,
) -> dict:
    """Fit, calibrate and evaluate on the time split; return ``result`` and the objects behind it."""
    windows = windows or models.Windows()
    train, val, test = models.split(feat, windows, hours)
    log(
        f"rows: train {train.height:,}, validation {val.height:,}, test {test.height:,}"
    )
    y_val, y_test = val["y"].to_numpy(), test["y"].to_numpy()
    days_test = test["label_day"].to_numpy()

    result: dict = {
        "windows": {k: str(v) for k, v in asdict(windows).items()},
        "label_hours": hours,
        "rows": {"train": train.height, "validation": val.height, "test": test.height},
        "label_days": {
            "train": [str(train["label_day"].min()), str(train["label_day"].max())],
            "validation": [str(val["label_day"].min()), str(val["label_day"].max())],
            "test": [str(test["label_day"].min()), str(test["label_day"].max())],
        },
        "base_rate": {
            "train": float(train["y"].mean()),
            "validation": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
    }

    base_val, base_test = baselines(val, hours), baselines(test, hours)
    result["baselines"] = {
        k: evaluate.ranking_scores(y_test, p) for k, p in base_test.items()
    }

    log("fitting logistic regression")
    logit, logit_cols = models.fit_logistic(train)
    p_logit = logit.predict_proba(models.matrix(test, logit_cols))[:, 1]
    result["logistic"] = evaluate.scores(y_test, p_logit)

    log("fitting LightGBM")
    booster = models.fit_lgbm(train, val, rounds=rounds)
    p_val_raw, p_test_raw = models.predict(booster, val), models.predict(booster, test)
    calibrator = models.Calibrator.fit(p_val_raw, y_val)
    p_val, p_test = calibrator(p_val_raw), calibrator(p_test_raw)
    result["best_iteration"] = int(booster.best_iteration)
    result["lightgbm_raw"] = evaluate.scores(y_test, p_test_raw)
    result["lightgbm"] = evaluate.scores(y_test, p_test)
    result["validation"] = {
        "lightgbm": evaluate.scores(y_val, p_val_raw),
        "mean_oos_hours_14d": evaluate.ranking_scores(
            y_val, base_val["mean_oos_hours_14d"]
        ),
    }
    result["reliability"] = {
        "raw": evaluate.reliability(y_test, p_test_raw),
        "calibrated": evaluate.reliability(y_test, p_test),
    }

    result["worklist"] = {
        "lightgbm": evaluate.worklist(y_test, p_test, days_test),
        "mean_oos_hours_14d": evaluate.worklist(
            y_test, base_test["mean_oos_hours_14d"], days_test
        ),
        "yesterday": evaluate.worklist(y_test, base_test["yesterday"], days_test),
    }
    result["cost"] = {
        "lightgbm": evaluate.cost_table(y_val, p_val, y_test, p_test),
        "mean_oos_hours_14d": evaluate.cost_table(
            y_val,
            base_val["mean_oos_hours_14d"],
            y_test,
            base_test["mean_oos_hours_14d"],
        ),
    }

    # The harder, more useful question: products that were not stocked out today.
    fresh = (test["oos_t"] < hours).to_numpy()
    result["newly_at_risk"] = {
        "rows": int(fresh.sum()),
        "base_rate": float(y_test[fresh].mean()),
        "lightgbm": evaluate.scores(y_test[fresh], p_test[fresh]),
        "mean_oos_hours_14d": evaluate.ranking_scores(
            y_test[fresh], base_test["mean_oos_hours_14d"][fresh]
        ),
    }

    cat = test["cat1"].to_numpy()
    ours = {r["group"]: r for r in evaluate.group_scores(y_test, p_test, cat)}
    ref = {
        r["group"]: r
        for r in evaluate.group_scores(y_test, base_test["mean_oos_hours_14d"], cat)
    }
    result["per_category"] = [
        {**ours[g], "baseline_pr_auc": ref[g]["pr_auc"]}
        for g in sorted(ours)
        if g in ref
    ]

    result["drift"] = {
        "weekly": _weekly({"train": train, "validation": val, "test": test}),
        "psi_train_to_test": {
            f: evaluate.psi(
                train[f].to_numpy().astype(float), test[f].to_numpy().astype(float)
            )
            for f in DRIFT_FEATURES
        },
    }
    result["censoring"] = _censoring(train)

    log("feature importance")
    sample = test.sample(min(100_000, test.height), seed=0)
    shap = np.abs(models.contributions(booster, sample)).mean(axis=0)
    gain = booster.feature_importance("gain")
    result["importance"] = sorted(
        (
            {
                "feature": f,
                "group": next(g for g, fs in GROUPS.items() if f in fs),
                "gain_share": float(gain[i] / gain.sum()),
                "mean_abs_shap": float(shap[i]),
            }
            for i, f in enumerate(FEATURES)
        ),
        key=lambda r: -r["mean_abs_shap"],
    )

    if ablate:
        result["ablation"] = []
        for group in GROUPS:
            log(f"ablation: without {group}")
            cols = models.without(group)
            b = models.fit_lgbm(train, val, cols, rounds=rounds)
            p = models.predict(b, test, cols)
            sc = evaluate.scores(y_test, p)
            result["ablation"].append(
                {
                    "dropped": group,
                    "pr_auc": sc["pr_auc"],
                    "roc_auc": sc["roc_auc"],
                    "delta_pr_auc": sc["pr_auc"] - result["lightgbm_raw"]["pr_auc"],
                }
            )

    if other_labels:
        result["label_definitions"] = []
        for h in LABEL_HOURS:
            log(f"label definition: {h}+ hours")
            tr, va, te = models.split(feat, windows, h)
            if h == hours:
                p_h, booster_h = p_test_raw, booster
            else:
                booster_h = models.fit_lgbm(tr, va, rounds=rounds)
                p_h = models.predict(booster_h, te)
            yh = te["y"].to_numpy()
            bl = baselines(te, h)
            result["label_definitions"].append(
                {
                    "hours": h,
                    "base_rate": float(yh.mean()),
                    "yesterday": evaluate.ranking_scores(yh, bl["yesterday"]),
                    "mean_oos_hours_14d": evaluate.ranking_scores(
                        yh, bl["mean_oos_hours_14d"]
                    ),
                    "lightgbm": evaluate.scores(yh, p_h),
                }
            )

    meta = {
        "features": FEATURES,
        "label_hours": hours,
        "trained_label_days": result["label_days"]["train"],
        "best_iteration": result["best_iteration"],
        "calibrator": calibrator.to_dict(),
        "thresholds": {
            str(r["ratio"]): r["threshold"] for r in result["cost"]["lightgbm"]
        },
    }
    predictions = pl.concat(
        [
            part.select(
                "store_id",
                "product_id",
                "dt",
                "label_day",
                "y",
                "cat1",
                "cat2",
                "city_id",
            ).with_columns(
                pl.lit(name).alias("part"), pl.Series("p", p), pl.Series("p_raw", raw)
            )
            for name, part, p, raw in (
                ("validation", val, p_val, p_val_raw),
                ("test", test, p_test, p_test_raw),
            )
        ]
    )
    arrays = {
        "y_test": y_test,
        "p": {
            "LightGBM (calibrated)": p_test,
            "LightGBM (raw)": p_test_raw,
            "Logistic regression": p_logit,
            "14-day mean out-of-stock hours": base_test["mean_oos_hours_14d"],
            "Yesterday": base_test["yesterday"],
        },
    }
    return {
        "result": result,
        "booster": booster,
        "meta": meta,
        "predictions": predictions,
        "arrays": arrays,
    }


def save(out: dict, *, directory=None) -> None:
    """``results/metrics.json`` (plus report and charts, see ``report``), the model and predictions."""
    from stockout_risk import report

    target = directory or results_dir()
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(
        json.dumps(out["result"], indent=4) + "\n", encoding="utf-8", newline="\n"
    )
    models.save_model(data_dir() / "model", out["booster"], out["meta"])
    out["predictions"].write_parquet(data_dir() / "predictions.parquet")
    report.write(out, target)


def last_day(feat: pl.DataFrame) -> date:
    """The last day whose next day is also in the data, so the prediction can be checked."""
    return feat.filter(pl.col("next_oos").is_not_null())["dt"].max()
