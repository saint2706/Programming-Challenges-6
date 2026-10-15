"""The experiment: models, calibration, operating points and how sure we can be of them.

``run`` returns a JSON-able ``result`` plus the objects the charts and the CLI need; ``save`` writes
them. Nothing is written by ``run``, so tests can use it on a synthetic table.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_flagging import data, evaluate, features, models
from fraud_flagging.features import FEATURES, GROUPS
from fraud_flagging.paths import data_dir, results_dir

Log = Callable[[str], None]
SWEEP = tuple(float(x) for x in np.geomspace(0.5, 200, 14))
DRIFT_FEATURES = ["log_amount", "V4", "V10", "V12", "V14", "V17"]


def load_frame() -> pl.DataFrame:
    """The cleaned, featurised real data."""
    return features.build(data.load())


def _val_test_scores(booster, val, test):
    return models.predict(booster, val), models.predict(booster, test)


def _decision_row(y, amount, flag, review_cost=None, n_boot=2000):
    c = evaluate.confusion(y, flag)
    ci = evaluate.bootstrap_decision(y, flag, n_boot)
    row = {
        **c,
        "precision_ci": ci["precision"],
        "recall_ci": ci["recall"],
        "fraud_amount_caught": float(amount[(y == 1) & flag].sum()),
        "fraud_amount_missed": float(amount[(y == 1) & ~flag].sum()),
    }
    return row


def run(
    frame: pl.DataFrame,
    windows: models.Windows | None = None,
    *,
    rounds: int = 1000,
    n_boot: int = 300,
    rolling: bool = True,
    ablate: bool = True,
    log: Log = lambda *_: None,
) -> dict:
    """Fit, calibrate and evaluate on the time split; return ``result`` and the objects behind it."""
    windows = windows or models.Windows()
    train, val, test = models.split(frame, windows)
    y_tr, y_va, y = train["y"].to_numpy(), val["y"].to_numpy(), test["y"].to_numpy()
    amount_va, amount = val["Amount"].to_numpy(), test["Amount"].to_numpy()
    hour = test["hour"].to_numpy()
    log(
        f"rows: train {train.height:,}, validation {val.height:,}, test {test.height:,}"
    )

    result: dict = {
        "windows": {
            "validation_from_hour": windows.val_hour,
            "test_from_hour": windows.test_hour,
        },
        "rows": {"train": train.height, "validation": val.height, "test": test.height},
        "frauds": {
            "train": int(y_tr.sum()),
            "validation": int(y_va.sum()),
            "test": int(y.sum()),
        },
        "fraud_rate": {
            "train": float(y_tr.mean()),
            "validation": float(y_va.mean()),
            "test": float(y.mean()),
        },
        "fraud_amount": {
            "train": float(train.filter(pl.col("y") == 1)["Amount"].sum()),
            "validation": float(amount_va[y_va == 1].sum()),
            "test": float(amount[y == 1].sum()),
        },
    }

    # --- candidates -------------------------------------------------------------------------
    cands: dict[str, dict] = {}
    log("isolation forest")
    forest = models.fit_isolation_forest(train)
    cands["isolation_forest"] = {
        "val": models.anomaly_score(forest, val),
        "test": models.anomaly_score(forest, test),
        "kind": "unsupervised",
    }
    for balanced in (False, True):
        name = "logistic_balanced" if balanced else "logistic"
        log(name)
        model, c = models.fit_logistic(train, val, balanced=balanced)
        cands[name] = {
            "val": model.predict_proba(models.matrix(val))[:, 1],
            "test": model.predict_proba(models.matrix(test))[:, 1],
            "kind": "logistic",
            "C": c,
        }
    boosters = {}
    for strategy in models.STRATEGIES:
        log(f"LightGBM ({strategy})")
        b = models.fit_lgbm(train, val, strategy=strategy, rounds=rounds)
        boosters[strategy] = b
        pv, pt = _val_test_scores(b, val, test)
        cands[f"lightgbm_{strategy}"] = {
            "val": pv,
            "test": pt,
            "kind": "lightgbm",
            "best_iteration": int(b.best_iteration),
        }

    result["candidates"] = {}
    for name, c in cands.items():
        row = {
            "kind": c["kind"],
            "validation": evaluate.ranking_scores(y_va, c["val"]),
            "test": evaluate.ranking_scores(y, c["test"]),
        }
        row["test_pr_auc_ci"] = evaluate.bootstrap_ap(y, c["test"], n_boot)
        row.update({k: c[k] for k in ("C", "best_iteration") if k in c})
        result["candidates"][name] = row

    # The main model is chosen on validation PR-AUC among the LightGBM variants, never on test.
    main_strategy = max(
        models.STRATEGIES,
        key=lambda s: result["candidates"][f"lightgbm_{s}"]["validation"]["pr_auc"],
    )
    booster = boosters[main_strategy]
    result["main_strategy"] = main_strategy
    log(f"main model: LightGBM ({main_strategy})")
    raw_val, raw_test = (
        cands[f"lightgbm_{main_strategy}"]["val"],
        cands[f"lightgbm_{main_strategy}"]["test"],
    )
    platt = models.Platt.fit(raw_val, y_va)
    p_val, p_test = platt(raw_val), platt(raw_test)
    result["calibration"] = {
        "platt": platt.to_dict(),
        "raw": evaluate.scores(y, raw_test),
        "platt_scaled": evaluate.scores(y, p_test),
        "reliability_raw": evaluate.reliability(y, raw_test),
        "reliability_platt": evaluate.reliability(y, p_test),
        "by_strategy": {
            s: {
                "raw": {
                    k: v
                    for k, v in evaluate.scores(
                        y, cands[f"lightgbm_{s}"]["test"]
                    ).items()
                    if k in ("log_loss", "brier")
                },
                "platt": {
                    k: v
                    for k, v in evaluate.scores(
                        y,
                        models.Platt.fit(cands[f"lightgbm_{s}"]["val"], y_va)(
                            cands[f"lightgbm_{s}"]["test"]
                        ),
                    ).items()
                    if k in ("log_loss", "brier")
                },
            }
            for s in models.STRATEGIES
        },
    }
    result["baselines"] = {
        "amount_only": evaluate.ranking_scores(y, amount),
        "random": {"pr_auc": float(y.mean()), "roc_auc": 0.5},
    }

    # --- operating points: fixed precision / recall ----------------------------------------
    log("operating points")
    precision_points, recall_points = {}, {}
    thresholds = {"precision": {}, "recall": {}, "cost_static": {}}
    for target in evaluate.PRECISIONS:
        thr = evaluate.threshold_for_precision(y_va, p_val, target)
        thresholds["precision"][str(target)] = thr
        if thr is None:
            precision_points[str(target)] = None
            continue
        flag = p_test >= thr
        val_c = evaluate.confusion(y_va, p_val >= thr)
        precision_points[str(target)] = {
            "threshold": thr,
            "validation": {"precision": val_c["precision"], "recall": val_c["recall"]},
            **_decision_row(y, amount, flag),
        }
    for target in evaluate.RECALLS:
        thr = evaluate.threshold_for_recall(y_va, p_val, target)
        thresholds["recall"][str(target)] = thr
        if thr is None:
            recall_points[str(target)] = None
            continue
        val_c = evaluate.confusion(y_va, p_val >= thr)
        recall_points[str(target)] = {
            "threshold": thr,
            "validation": {"precision": val_c["precision"], "recall": val_c["recall"]},
            **_decision_row(y, amount, p_test >= thr),
        }
    result["fixed_precision"], result["fixed_recall"] = precision_points, recall_points

    # --- operating points: cost ------------------------------------------------------------
    cost_rows = []
    for c in evaluate.REVIEW_COSTS:
        thr = evaluate.best_cost_threshold(y_va, amount_va, p_val, c)
        thresholds["cost_static"][str(c)] = thr
        static = np.zeros(len(y), bool) if thr is None else p_test >= thr
        aware = evaluate.amount_aware_flag(p_test, amount, c)
        triv = evaluate.trivial_costs(y, amount, c)
        row = {
            "review_cost": c,
            "static_threshold": thr,
            **{f"trivial_{k}": v for k, v in triv.items()},
        }
        for name, flag in (("static", static), ("amount_aware", aware)):
            cost = evaluate.policy_cost(y, amount, flag, c)
            row[name] = {
                **cost,
                "cost_ci": evaluate.bootstrap_cost(
                    y, amount, flag, c, n_boot=n_boot * 4
                ),
                **{
                    k: v
                    for k, v in evaluate.confusion(y, flag).items()
                    if k in ("tp", "fp", "fn", "flagged", "precision", "recall")
                },
            }
        cost_rows.append(row)
    result["cost"] = cost_rows
    sweep = []
    for c in SWEEP:
        thr = evaluate.best_cost_threshold(y_va, amount_va, p_val, c)
        static = np.zeros(len(y), bool) if thr is None else p_test >= thr
        triv = evaluate.trivial_costs(y, amount, c)
        sweep.append(
            {
                "review_cost": c,
                "static": evaluate.policy_cost(y, amount, static, c)["total"],
                "amount_aware": evaluate.policy_cost(
                    y, amount, evaluate.amount_aware_flag(p_test, amount, c), c
                )["total"],
                **triv,
            }
        )
    result["cost_sweep"] = sweep

    # --- operating points: review capacity -------------------------------------------------
    capacity = []
    for k in evaluate.CAPACITIES:
        row = {"per_hour": k, "hours": len(np.unique(hour))}
        for name, score in (("by_risk", p_test), ("by_expected_loss", p_test * amount)):
            flag = evaluate.capacity_flag(score, hour, k)
            row[name] = {
                **evaluate.confusion(y, flag),
                "fraud_amount_caught": float(amount[(y == 1) & flag].sum()),
            }
        capacity.append(row)
    result["capacity"] = capacity

    # --- how sure can we be of a threshold -------------------------------------------------
    log("threshold stability")
    result["threshold_stability"] = [
        evaluate.threshold_stability(y_va, p_val, y, p_test, t, n_boot)
        for t in (0.8, 0.9, 0.95)
    ]

    # --- rank metrics ----------------------------------------------------------------------
    result["precision_at_k"] = {
        str(k): evaluate.precision_at_k(y, p_test, k) for k in (20, 50, 100)
    }
    result["recall_at_precision"] = {
        str(t): evaluate.recall_at_precision(y, p_test, t)
        for t in (0.5, 0.8, 0.9, 0.95)
    }

    # --- errors at the 90%-precision operating point ---------------------------------------
    thr90 = thresholds["precision"].get("0.9")
    if thr90 is not None:
        flag = p_test >= thr90
        caught, missed = amount[(y == 1) & flag], amount[(y == 1) & ~flag]
        result["errors_at_precision_0.9"] = {
            "threshold": thr90,
            "caught": len(caught),
            "missed": len(missed),
            "median_amount_caught": float(np.median(caught)) if len(caught) else None,
            "median_amount_missed": float(np.median(missed)) if len(missed) else None,
            "median_score_missed": float(np.median(p_test[(y == 1) & ~flag]))
            if len(missed)
            else None,
            "false_alarm_median_amount": float(np.median(amount[(y == 0) & flag]))
            if ((y == 0) & flag).any()
            else None,
        }

    # --- importance, drift -----------------------------------------------------------------
    log("importance")
    shap = np.abs(models.contributions(booster, test)).mean(axis=0)
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
    result["drift"] = {
        "psi_train_to_test": {
            f: evaluate.psi(train[f].to_numpy(), test[f].to_numpy())
            for f in DRIFT_FEATURES
        },
        "fraud_per_hour": [
            {"hour": int(h), "n": n, "frauds": int(f)}
            for h, n, f in frame.group_by("hour")
            .agg(pl.len(), pl.col("y").sum())
            .sort("hour")
            .iter_rows()
        ],
    }

    # --- feature-group ablation, rolling origin ---------------------------------------------
    if ablate:
        result["ablation"] = []
        for group in GROUPS:
            log(f"ablation: without {group}")
            cols = features.without(group)
            b = models.fit_lgbm(train, val, cols, strategy=main_strategy, rounds=rounds)
            sc = evaluate.ranking_scores(y, models.predict(b, test, cols))
            result["ablation"].append(
                {
                    "dropped": group,
                    "pr_auc": sc["pr_auc"],
                    "roc_auc": sc["roc_auc"],
                    "delta_pr_auc": sc["pr_auc"]
                    - result["candidates"][f"lightgbm_{main_strategy}"]["test"][
                        "pr_auc"
                    ],
                }
            )
    if rolling:
        log("rolling origin")
        result["rolling_origin"] = _rolling(
            frame,
            windows,
            main_strategy,
            int(booster.best_iteration),
            result["candidates"]["logistic"].get("C", 0.1),
        )

    meta = {
        "features": FEATURES,
        "strategy": main_strategy,
        "best_iteration": int(booster.best_iteration),
        "platt": platt.to_dict(),
        "thresholds": thresholds,
        "fraud_rate": result["fraud_rate"],
    }
    predictions = pl.concat(
        [
            part.select("tx_id", "Time", "Amount", "y", "hour").with_columns(
                pl.lit(name).alias("part"), pl.Series("p", p), pl.Series("p_raw", raw)
            )
            for name, part, p, raw in (
                ("validation", val, p_val, raw_val),
                ("test", test, p_test, raw_test),
            )
        ]
    )
    arrays = {
        "y_test": y,
        "amount": amount,
        "p": p_test,
        "scores": {k: c["test"] for k, c in cands.items()},
    }
    return {
        "result": result,
        "booster": booster,
        "meta": meta,
        "predictions": predictions,
        "arrays": arrays,
    }


def _rolling(frame, windows, strategy, rounds, c) -> list[dict]:
    """Train on everything before each 6-hour block and score the block: the same model, four times.

    No validation window is carved out here, so LightGBM gets the number of trees the main model
    stopped at, and the logistic regression the ``C`` it chose.
    """
    start = windows.val_hour
    end = float(frame["Time"].max() / 3600)
    rows = []
    for lo in np.arange(start, end, 6.0):
        hi = lo + 6.0
        tr = frame.filter(pl.col("Time") < lo * 3600)
        te = frame.filter((pl.col("Time") >= lo * 3600) & (pl.col("Time") < hi * 3600))
        if te.height == 0 or te["y"].sum() == 0 or tr["y"].sum() == 0:
            continue
        y_tr, y_te = tr["y"].to_numpy(), te["y"].to_numpy()
        booster = models.fit_lgbm_fixed(tr, strategy=strategy, rounds=max(rounds, 20))
        logit = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        logit.fit(models.matrix(tr), y_tr)
        rows.append(
            {
                "from_hour": float(lo),
                "to_hour": float(hi),
                "rows": te.height,
                "frauds": int(y_te.sum()),
                "lightgbm": evaluate.ranking_scores(
                    y_te, booster.predict(models.matrix(te))
                ),
                "logistic": evaluate.ranking_scores(
                    y_te, logit.predict_proba(models.matrix(te))[:, 1]
                ),
            }
        )
    return rows


def save(out: dict, *, directory=None) -> None:
    """``results/metrics.json`` (plus report and charts), the model and the predictions."""
    from fraud_flagging import report

    target = directory or results_dir()
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(
        json.dumps(out["result"], indent=4) + "\n", encoding="utf-8", newline="\n"
    )
    models.save_model(data_dir() / "model", out["booster"], out["meta"])
    out["predictions"].write_parquet(data_dir() / "predictions.parquet")
    report.write(out, target)
