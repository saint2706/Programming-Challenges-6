"""prepare -> train -> evaluate -> ``results/report.json`` plus reloadable models.

``prepare`` turns the raw corpus into a labelled, featured dataset (written to the
gitignored ``data/``). ``run_all`` splits by time, fits and calibrates the models,
evaluates them on the held-out test period and writes aggregates only: the report
never contains message text or addresses.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from data import CSV_NAME, DATA_DIR, dedupe, load_mailboxes, received
from evaluate import (
    ablation,
    bootstrap_ci,
    daily_inbox_metrics,
    ece,
    pr_auc,
    reliability,
    summarize,
)
from explain import contributions, group_contributions
from features import (
    FEATURE_GROUPS,
    build_sent_index,
    metadata_features,
    received_frame,
)
from models import MODEL_NAMES, Calibrator, Models, fit_models, time_split
from thread import censor_cutoff, label_received

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

# The six mailboxes with the most sent mail: enough replies to learn from.
DEFAULT_USERS = [
    "mann-k",
    "kaminski-v",
    "dasovich-j",
    "germany-c",
    "shackleton-s",
    "jones-t",
]
CALIBRATED = ["tfidf_lr", "lgbm_meta", "lgbm_meta_text"]
SHAP_SAMPLE = 2000
DATASET_FILE = "dataset.parquet"
ARTIFACTS_FILE = "models.joblib"


def prepare(
    csv: Path, users: list[str], out_dir: Path
) -> tuple[pl.DataFrame, dict[str, dict]]:
    """Labelled, featured dataset for the given mailboxes, plus drop counts and label rates."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    stats: dict[str, dict] = {}
    for user, box in load_mailboxes(Path(csv), users).items():
        owner = box.owner
        # Mail the owner sent can sit in any folder (e.g. all_documents), so a
        # reply filed away from the sent folder still counts as a reply.
        owner_sent = dedupe(box.sent + [m for m in box.others if m.sender == owner])
        raw_addressed = [
            m for m in box.others if m.sender != owner and owner in m.to + m.cc
        ]
        addressed = dedupe(raw_addressed)
        rec = received(box.others, owner)
        cutoff = censor_cutoff(owner_sent)
        kept = [m for m in rec if m.date <= cutoff]
        labels = label_received(kept, owner_sent, owner)
        frame = received_frame(kept, labels, user, owner)
        feats = metadata_features(frame, build_sent_index(owner_sent, owner))
        frames.append(frame.join(feats, on="message_id", how="left"))
        n = max(len(kept), 1)
        stats[user] = {
            "copies_in_folders": len(raw_addressed),
            "addressed_to_owner": len(addressed),
            "dropped_bad_or_missing_date": len(addressed) - len(rec),
            "received": len(rec),
            "censored": len(rec) - len(kept),
            "kept": len(kept),
            "acted_rate": float(labels["acted"].sum()) / n,
            "reply_rate": float((labels["kind"] == "reply").sum()) / n,
            "forward_rate": float((labels["kind"] == "forward").sum()) / n,
        }
    df = pl.concat(frames).sort(["mailbox", "date", "message_id"])
    df.write_parquet(out_dir / DATASET_FILE)
    (out_dir / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    return df, stats


@dataclass
class Artifacts:
    models: Models
    calibrators: dict[str, Calibrator]
    seed: int


def load_artifacts(data_dir: Path = DATA_DIR) -> Artifacts:
    """Reload what ``run_all`` saved.

    joblib is pickle: only ever load the file this pipeline wrote into the local,
    gitignored ``data/`` directory, never one from an untrusted source.
    """
    return joblib.load(Path(data_dir) / ARTIFACTS_FILE)


def _clean(obj):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _per_mailbox(test: pl.DataFrame, scores: np.ndarray) -> dict[str, dict]:
    out = {}
    for box in sorted(set(test["mailbox"])):
        mask = (test["mailbox"] == box).to_numpy()
        s = summarize(test["acted"].to_numpy()[mask], scores[mask])
        out[box] = {
            k: s[k] for k in ("n", "positives", "pos_rate", "pr_auc", "roc_auc")
        }
    return out


def run_all(
    data_dir: Path,
    out_dir: Path,
    users: list[str],
    seed: int = 0,
    csv: Path | None = None,
) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, stats = prepare(csv or data_dir / CSV_NAME, users, data_dir)
    train, val, test = time_split(df)
    models = fit_models(train, val, seed=seed)

    y_test = test["acted"].to_numpy().astype(int)
    y_val = val["acted"].to_numpy().astype(int)
    scores = {name: models.predict(name, test) for name in MODEL_NAMES}

    model_report = {}
    for name in MODEL_NAMES:
        daily = daily_inbox_metrics(test, scores[name], seed=seed)
        daily_summary = {"n_days": daily.height}
        for metric in ("precision_at_3", "ndcg_at_5", "recall_top20"):
            mean, lo, hi = bootstrap_ci(daily[metric].to_numpy(), seed=seed)
            daily_summary[metric] = {"mean": mean, "lo": lo, "hi": hi}
        per_box = _per_mailbox(test, scores[name])
        model_report[name] = {
            "overall": summarize(y_test, scores[name]),
            "per_mailbox": per_box,
            "macro_pr_auc": float(np.nanmean([b["pr_auc"] for b in per_box.values()])),
            "daily": daily_summary,
        }

    calibrators: dict[str, Calibrator] = {}
    calibration = {}
    for name in CALIBRATED:
        cal = Calibrator().fit(models.predict(name, val), y_val)
        calibrators[name] = cal
        raw = scores[name]
        fixed = cal.predict(raw)
        calibration[name] = {
            "before": {
                "ece": ece(y_test, raw),
                "reliability": reliability(y_test, raw),
            },
            "after": {
                "ece": ece(y_test, fixed),
                "reliability": reliability(y_test, fixed),
            },
        }

    drops = ablation(train, val, test, FEATURE_GROUPS, seed=seed)
    drops["text"] = pr_auc(y_test, scores["lgbm_meta_text"]) - pr_auc(
        y_test, scores["lgbm_meta"]
    )

    rng = np.random.default_rng(seed)
    pick = rng.choice(test.height, size=min(SHAP_SAMPLE, test.height), replace=False)
    sample = test[sorted(pick.tolist())]
    contribs, _ = contributions(
        models.lgbm_meta_text, models.matrix(sample, with_text=True)
    )
    grouped, labels = group_contributions(contribs, models.meta_text_names)
    mean_abs = np.abs(grouped).mean(axis=0)
    top = sorted(zip(labels, mean_abs, strict=True), key=lambda t: -t[1])[:12]

    report = {
        "seed": seed,
        "data": {
            "mailboxes": stats,
            "splits": {"train": train.height, "val": val.height, "test": test.height},
            "test_period": {
                "start": test["date"].min(),
                "end": test["date"].max(),
            },
        },
        "models": model_report,
        "calibration": calibration,
        "ablation": drops,
        "shap": {
            "n": sample.height,
            "top": [{"feature": f, "mean_abs": float(v)} for f, v in top],
        },
    }
    report = _clean(report)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    joblib.dump(Artifacts(models, calibrators, seed), data_dir / ARTIFACTS_FILE)
    return report
