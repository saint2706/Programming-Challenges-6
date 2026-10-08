"""The benchmark as resumable stages, each cached on disk under a key that says what it depends on.



Stages (``STAGES``): ``train`` fits every model on the train split; ``calibrate`` fits every

post-hoc calibrator on the *calibration* split only; ``evaluate`` scores everything on the in-time

test and the out-of-domain test with cluster-bootstrap intervals; ``conformal``, ``shift`` and

``selective`` are the three extras; ``ablation`` re-trains linear heads under the other input-text

settings. A stage file stores the key it was computed under (its own config fields, a fingerprint

of the data, and the keys of the stages it consumed) and is reused only if the key still matches,

so changing a setting recomputes exactly what depends on it and an interrupted run resumes. The

JSON marker is written last, so a crash never leaves a stage that looks finished.



``--stage X`` runs X (and whatever it needs that is missing) but never drops other finished stages

from ``report.json``; stages that are stale or missing are listed under ``"missing"``.

"""

from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from review_stars import (
    calibrate,
    conformal,
    data,
    embed,
    heads,
    metrics,
    selective,
    shift,
    stats,
)
from review_stars import text as text_mod
from review_stars.config import FRAMINGS, Config
from review_stars.probs import Predictions

STAGE_VERSION = 2  # bump when a stage's payload format or meaning changes

STAGES = (
    "train",
    "calibrate",
    "evaluate",
    "conformal",
    "shift",
    "selective",
    "ablation",
)

DEPENDS = {
    "train": (),
    "calibrate": ("train",),
    "evaluate": ("calibrate",),
    "conformal": ("calibrate",),
    "shift": ("train",),
    "selective": ("calibrate",),
    "ablation": ("train",),
}

STAGE_FIELDS = {
    "train": [
        "l2_grid",
        "mlp_hidden",
        "mlp_dropouts",
        "mlp_seeds",
        "mlp_epochs",
        "mlp_patience",
        "mlp_lr",
        "mlp_batch",
        "mlp_weight_decay",
        "tfidf_features",
        "tfidf_c_grid",
        "ridge_alpha_grid",
        "head_seed",
    ],
    "calibrate": [],
    "evaluate": ["n_boot", "ece_bins", "levels", "stat_seed"],
    "conformal": ["alpha", "n_boot", "stat_seed"],
    "shift": ["recal_ns", "recal_draws", "ece_bins", "stat_seed"],
    "selective": [],
    "ablation": ["ablation_train", "ece_bins", "text_mode", "stat_seed"],
}

EVAL_SPLITS = ("cal", "test", "ood_test", "ood_pool")

SCORED = ("test", "ood_test")

ABLATION_MODES = ("title_text", "text_only", "raw_title")

RISK_TARGETS = (0.1, 0.2, 0.3)

FIXED_TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)

RELIABILITY_BINS = 10


def reliability_bands(conf, correct, rep_n, rep_k) -> dict:
    """A reliability diagram with a confidence histogram and bootstrap bands.

    Equal-width bins have the same edges in every bootstrap replicate, so each bin's accuracy has a
    band (2.5th and 97.5th percentile over replicates; empty in replicates where the bin is empty);
    ``n`` is the confidence histogram, which shows how sharp the model is."""
    n, k = metrics.fixed_bin_counts(conf, correct, RELIABILITY_BINS)
    acc = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    reps = np.where(rep_n > 0, rep_k / np.maximum(rep_n, 1), np.nan)
    lo, hi = np.full(RELIABILITY_BINS, np.nan), np.full(RELIABILITY_BINS, np.nan)
    for b in range(RELIABILITY_BINS):
        col = reps[:, b][~np.isnan(reps[:, b])]
        if len(col) >= 2:
            lo[b], hi[b] = np.percentile(col, [2.5, 97.5])
    return {
        "edges": np.linspace(0.0, 1.0, RELIABILITY_BINS + 1),
        "n": n,
        "acc": acc,
        "lo": lo,
        "hi": hi,
    }


COMPARE_METRICS = ("ece", "nll", "rps", "mae_expected")


def model_names(cfg: Config) -> list[str]:

    names = ["tfidf-classification", "tfidf-regression"]
    names += [f"linear-{f}" for f in FRAMINGS]
    names += [f"mlp-{f}" for f in FRAMINGS]
    names += [f"ens-{f}" for f in FRAMINGS] if cfg.mlp_seeds > 1 else []
    return names


def framing_of(model: str) -> str:

    return model.split("-", 1)[1]


def capacity_of(model: str) -> str:

    return model.split("-", 1)[0]


# ---------------------------------------------------------------- features


@dataclass
class SplitData:
    X: np.ndarray  # float32 [n, d] unit-norm embeddings
    y: np.ndarray  # int64 stars 1..5
    groups: np.ndarray  # parent_asin: the bootstrap cluster
    texts: list[str]  # what was embedded (the TF-IDF baselines read it too)
    n_tok: np.ndarray  # int32 token counts
    verified: np.ndarray  # bool
    ids: list[str]


@dataclass
class Features:
    splits: dict[str, SplitData]
    meta: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for name in sorted(self.splits):
            s = self.splits[name]
            h.update(f"{name}|{s.X.shape}".encode())
            for arr in (s.X, s.y, s.n_tok, s.verified):
                h.update(np.ascontiguousarray(arr).tobytes())
            h.update("\0".join(s.groups.astype(str)).encode("utf-8", "surrogatepass"))
            h.update("\0".join(s.texts).encode("utf-8", "surrogatepass"))
        return h.hexdigest()[:16]


def load_features(cfg: Config, data_dir, get_choice, progress=None) -> Features:
    """Prepared reviews -> ``Features``: build the model text, embed everything once (shard cache)."""
    data_dir = Path(data_dir)
    df = data.load_prepared(data_dir / "prepared")
    texts_all, order = [], []
    for split in data.SPLITS:
        sub = df.filter(pl.col("split") == split)
        if sub.is_empty():
            raise ValueError(
                f"the prepared data has no {split!r} reviews; run `review-stars prepare`"
            )
        order.append((split, sub))
        texts_all += [
            text_mod.build_text(t, x, cfg.text_mode)
            for t, x in zip(sub["title"], sub["text"], strict=True)
        ]
    X, n_tok, info = embed.embed_texts(
        texts_all, data_dir / "embeddings", get_choice, progress=progress
    )
    splits, at = {}, 0
    for split, sub in order:
        n = sub.height
        splits[split] = SplitData(
            X=X[at : at + n],
            y=sub["rating"].to_numpy().astype(np.int64),
            groups=sub["parent_asin"].to_numpy(),
            texts=texts_all[at : at + n],
            n_tok=n_tok[at : at + n],
            verified=sub["verified"].to_numpy().astype(bool),
            ids=sub["id"].to_list(),
        )
        at += n
    windows_json = data_dir / "prepared" / "windows.json"
    meta = {
        "embedding": info,
        "prepared": json.loads(windows_json.read_text(encoding="utf-8"))
        if windows_json.exists()
        else {},
        "empty_texts": int(sum(not t for t in texts_all)),
    }
    return Features(splits=splits, meta=meta)


def make_ablation_provider(cfg: Config, data_dir, get_choice) -> Callable:
    """``provider(mode, subsets) -> {split: embeddings}`` for the input-text ablations: embeds only
    the requested rows of each split under another ``text_mode`` (shard-cached like everything)."""
    df = data.load_prepared(Path(data_dir) / "prepared")

    def provider(mode: str, subsets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for split, idx in subsets.items():
            sub = df.filter(pl.col("split") == split)
            titles, texts = sub["title"].to_list(), sub["text"].to_list()
            built = [text_mod.build_text(titles[i], texts[i], mode) for i in idx]
            X, _, _ = embed.embed_texts(
                built, Path(data_dir) / "embeddings", get_choice
            )
            out[split] = X
        return out

    return provider


# ---------------------------------------------------------------- JSON cleaning, stage files


def clean(obj):
    """Strict-JSON-safe copy: NaN/inf -> None, numpy scalars/arrays -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    return obj


def stage_key(stage: str, cfg: Config, fp: str, upstream: list[str]) -> str:

    cfg_all = cfg.to_dict()
    payload = {
        "version": STAGE_VERSION,
        "stage": stage,
        "data": fp,
        "cfg": {k: cfg_all[k] for k in STAGE_FIELDS[stage]},
        "upstream": upstream,
    }
    if stage == "ablation":
        payload["embed"] = [embed.MODEL_ID, embed.EMBED_VERSION]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _paths(results_dir: Path, stage: str):

    base = results_dir / "stages"
    return base / f"{stage}.json", base / f"{stage}.npz"


def _read_stage(results_dir: Path, stage: str, key: str):
    """``(payload, arrays or None)`` if a valid stage file exists for ``key``, else ``None``."""
    js, npz = _paths(results_dir, stage)
    try:
        stored = json.loads(js.read_text(encoding="utf-8"))
        if stored.get("key") != key:
            return None
        arrays = None
        if stored.get("has_arrays"):
            with np.load(npz, allow_pickle=False) as z:
                arrays = dict(z)
        return stored["payload"], arrays
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        # missing, torn, or not what we wrote
        return None


def _write_stage(results_dir: Path, stage: str, key: str, payload, arrays=None) -> None:

    js, npz = _paths(results_dir, stage)
    js.parent.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        part = npz.with_name(npz.name + ".part")
        with part.open("wb") as fh:
            np.savez(fh, **arrays)
        part.replace(npz)
    tmp = js.with_name(js.name + ".part")
    tmp.write_text(
        json.dumps(
            {"key": key, "has_arrays": arrays is not None, "payload": clean(payload)}
        ),
        encoding="utf-8",
    )
    tmp.replace(js)  # last: it marks the stage complete


# ---------------------------------------------------------------- stage context


class Ctx:
    """Everything a stage needs: features, config, and the upstream results (loaded lazily)."""

    def __init__(self, features: Features, cfg: Config, results_dir: Path, progress):
        self.features, self.cfg, self.results_dir, self.say = (
            features,
            cfg,
            results_dir,
            progress,
        )
        self.models = model_names(cfg)
        self.arrays: dict[str, dict] = {}
        self.payloads: dict[str, dict] = {}
        self._preds = None
        self._cals = None

    def split(self, name: str) -> SplitData:
        return self.features.splits[name]

    def preds(self) -> dict[str, dict[str, Predictions]]:
        if self._preds is None:
            arrays = self.arrays["train"]
            self._preds = {
                m: {
                    s: Predictions.from_arrays(arrays, f"{m}/{s}/") for s in EVAL_SPLITS
                }
                for m in self.models
            }
        return self._preds

    def calibrators(self) -> dict[str, dict]:
        if self._cals is None:
            arrays = self.arrays["calibrate"]
            self._cals = {}
            for m in self.models:
                self._cals[m] = {}
                for kind in calibrate.KINDS:
                    if f"cal/{m}/{kind}/kind" in arrays:
                        self._cals[m][kind] = calibrate.from_arrays(
                            arrays, f"cal/{m}/{kind}/"
                        )
        return self._cals

    def proba(self, model: str, kind: str, split: str) -> np.ndarray:
        return self.calibrators()[model][kind].proba(self.preds()[model][split])


# ---------------------------------------------------------------- stage: train


def _train_stage(ctx: Ctx):

    cfg = ctx.cfg
    tr, va = ctx.split("train"), ctx.split("val")
    arrays: dict[str, np.ndarray] = {}
    summary: dict = {"models": {}, "n_train": len(tr.y), "n_val": len(va.y)}
    models_dir = ctx.results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    def store(name: str, per_split: dict[str, Predictions]):
        for split, pred in per_split.items():
            arrays.update(pred.to_arrays(f"{name}/{split}/"))

    t0 = time.time()
    ctx.say("train: TF-IDF baselines")
    base = heads.fit_tfidf(tr.texts, tr.y, va.texts, va.y, cfg)
    outs = {s: base.predict(ctx.split(s).texts) for s in EVAL_SPLITS}
    for framing in ("classification", "regression"):
        store(
            f"tfidf-{framing}",
            {s: Predictions([outs[s][framing]]) for s in EVAL_SPLITS},
        )
        summary["models"][f"tfidf-{framing}"] = {"hyper": base.report}
    for framing in FRAMINGS:
        ctx.say(f"train: linear {framing}")
        head, table = heads.select_linear(framing, tr.X, tr.y, va.X, va.y, cfg.l2_grid)
        head.save(models_dir / f"linear-{framing}.pt")
        store(
            f"linear-{framing}",
            {s: Predictions([head.predict(ctx.split(s).X)]) for s in EVAL_SPLITS},
        )
        summary["models"][f"linear-{framing}"] = {
            "l2": head.l2,
            "val_nll": heads.val_nll(head, va.X, va.y),
            "grid": table,
            "history": head.history,
        }
        ctx.say(f"train: MLP {framing} (seed 0 + dropout choice)")
        kw = {
            "epochs": cfg.mlp_epochs,
            "patience": cfg.mlp_patience,
            "lr": cfg.mlp_lr,
            "batch": cfg.mlp_batch,
            "weight_decay": cfg.mlp_weight_decay,
        }
        first, dtable = heads.select_mlp(
            framing, tr.X, tr.y, va.X, va.y,
            dropouts=cfg.mlp_dropouts, hidden=cfg.mlp_hidden, seed=cfg.head_seed, **kw,
        )  # fmt: skip
        members = [first]
        for i in range(1, cfg.mlp_seeds):
            ctx.say(f"train: MLP {framing} seed {i}")
            members.append(
                heads.fit_mlp(
                    framing,
                    tr.X,
                    tr.y,
                    va.X,
                    va.y,
                    hidden=cfg.mlp_hidden,
                    dropout=first.dropout,
                    seed=cfg.head_seed + i,
                    **kw,
                )
            )
        for i, m in enumerate(members):
            m.save(models_dir / f"mlp-{framing}-s{i}.pt")
        preds = {s: [m.predict(ctx.split(s).X) for m in members] for s in EVAL_SPLITS}
        store(f"mlp-{framing}", {s: Predictions([preds[s][0]]) for s in EVAL_SPLITS})
        summary["models"][f"mlp-{framing}"] = {
            "dropout": first.dropout,
            "dropout_grid": dtable,
            "hidden": cfg.mlp_hidden,
            "epochs": [len(m.history["val_nll"]) for m in members],
            "val_nll": [min(m.history["val_nll"]) for m in members],
        }
        if cfg.mlp_seeds > 1:
            store(f"ens-{framing}", {s: Predictions(preds[s]) for s in EVAL_SPLITS})
            summary["models"][f"ens-{framing}"] = {"members": len(members)}
    summary["seconds"] = round(time.time() - t0, 1)
    summary["models_list"] = ctx.models
    return summary, arrays


# ---------------------------------------------------------------- stage: calibrate


def _calibrate_stage(ctx: Ctx):

    cal = ctx.split("cal")
    arrays: dict[str, np.ndarray] = {}
    summary: dict = {"n_cal": len(cal.y), "models": {}}
    for m in ctx.models:
        ctx.say(f"calibrate: {m}")

        # fit on the calibration split only: the one place any calibrator sees labels
        fitted = calibrate.fit_all(ctx.preds()[m]["cal"], cal.y)
        row: dict = {}
        for kind, c in fitted.items():
            arrays.update(c.to_arrays(f"cal/{m}/{kind}/"))
            if kind == "temperature":
                row["T"] = c.T
            elif kind == "vector":
                row["a"], row["b"] = c.a.tolist(), c.b.tolist()
            elif kind == "isotonic":
                row["degenerate_stars"] = [k + 1 for k in c.degenerate]
        summary["models"][m] = row
    return summary, arrays


# ---------------------------------------------------------------- stage: evaluate


def comparison_pairs(models: list[str], kinds_of: dict[str, list[str]]):
    """``(label, key_a, key_b, metrics)``: differences worth a paired interval."""
    pairs = []
    key = lambda m, k: f"{m}|{k}"
    for m in models:
        if "temperature" in kinds_of[m]:
            pairs.append(
                (
                    f"{m}: temperature - none",
                    key(m, "temperature"),
                    key(m, "none"),
                    ("ece", "smece", "nll", "brier", "rps"),
                )
            )
        if "isotonic" in kinds_of[m] and "temperature" in kinds_of[m]:
            pairs.append(
                (
                    f"{m}: isotonic - temperature",
                    key(m, "isotonic"),
                    key(m, "temperature"),
                    ("ece", "nll"),
                )
            )
    have = set(models)
    for cap in ("linear", "mlp", "ens"):
        for a, b in (
            ("classification", "regression"),
            ("classification", "ordinal"),
            ("regression", "ordinal"),
        ):
            ma, mb = f"{cap}-{a}", f"{cap}-{b}"
            if ma in have and mb in have:
                for kind in ("none", "temperature"):
                    pairs.append(
                        (
                            f"{cap}: {a} - {b} [{kind}]",
                            key(ma, kind),
                            key(mb, kind),
                            COMPARE_METRICS,
                        )
                    )
    for f in FRAMINGS:
        for hi, lo in (("mlp", "linear"), ("ens", "mlp")):
            if f"{hi}-{f}" in have and f"{lo}-{f}" in have:
                pairs.append(
                    (
                        f"{f}: {hi} - {lo} [temperature]",
                        key(f"{hi}-{f}", "temperature"),
                        key(f"{lo}-{f}", "temperature"),
                        COMPARE_METRICS,
                    )
                )
    return pairs  # fmt: skip


def _evaluate_stage(ctx: Ctx):

    cfg = ctx.cfg
    cals = ctx.calibrators()
    kinds_of = {m: list(cals[m]) for m in ctx.models}
    seen = set(ctx.split("train").groups.tolist())
    out: dict = {"splits": {}}
    for split in SCORED:
        ctx.say(f"evaluate: {split}")
        sd = ctx.split(split)
        probs = {
            f"{m}|{k}": ctx.proba(m, k, split) for m in ctx.models for k in kinds_of[m]
        }
        point = {key: metrics.bundle(P, sd.y, cfg.ece_bins) for key, P in probs.items()}
        reps = stats.boot_metrics(
            probs,
            sd.y,
            sd.groups,
            cfg.n_boot,
            cfg.stat_seed,
            cfg.ece_bins,
            cfg.n_jobs,
            curve_bins=RELIABILITY_BINS,
        )
        rows: dict = {}
        for key, P in probs.items():
            conf, correct = metrics.top_label(P, sd.y)
            rows[key] = {
                "point": point[key],
                "ci": {
                    k: stats.ci(reps[key][k], point[key][k])
                    for k in metrics.BUNDLE_KEYS
                },
                "reliability": {
                    k: v.tolist()
                    for k, v in metrics.reliability_curve(
                        conf, correct, cfg.ece_bins
                    ).items()
                },
                "reliability_fixed": reliability_bands(
                    conf,
                    correct,
                    reps[key]["reliability_n"],
                    reps[key]["reliability_k"],
                ),
                "coverage": metrics.coverage_curve(P, sd.y, cfg.levels),
            }
        diffs = []
        for label, ka, kb, names in comparison_pairs(ctx.models, kinds_of):
            for name in names:
                est = point[ka][name] - point[kb][name]
                diffs.append(
                    {
                        "label": label,
                        "metric": name,
                        **stats.ci(reps[ka][name] - reps[kb][name], est),
                    }
                )
        masks = shift.slice_masks(
            pl.DataFrame({"parent_asin": sd.groups, "verified": sd.verified}),
            seen,
            sd.n_tok,
        )
        slices = {
            f"{m}|{k}": shift.slice_table(probs[f"{m}|{k}"], sd.y, masks, cfg.ece_bins)
            for m in ctx.models
            for k in ("none", "temperature")
        }
        intervals = {}
        for m in ctx.models:
            pred = ctx.preds()[m][split]
            if pred.framing == "regression" and pred.is_single:
                mem = pred.members[0]
                T = cals[m]["temperature"].T
                intervals[m] = {
                    "none": metrics.interval_coverage(
                        mem.mu, mem.sigma, sd.y, cfg.levels
                    ),
                    "temperature": metrics.interval_coverage(
                        mem.mu, mem.sigma * T, sd.y, cfg.levels
                    ),
                }
        out["splits"][split] = {
            "n": len(sd.y),
            "rows": rows,
            "diffs": diffs,
            "slices": slices,
            "intervals": intervals,
            "class_share": [float(np.mean(sd.y == k)) for k in range(1, 6)],
        }
    return out, None


# ---------------------------------------------------------------- stage: conformal


def _hit_ci(hit: np.ndarray, groups, n_boot: int, seed: int) -> dict:

    reps = np.array(
        [hit[idx].mean() for idx in stats.boot_indices(groups, n_boot, seed)]
    )
    return stats.ci(reps, float(hit.mean()))


def _conformal_stage(ctx: Ctx):

    cfg = ctx.cfg
    cal = ctx.split("cal")
    n_cal = len(cal.y)

    def draws(code: int, n: int) -> np.ndarray:
        return np.random.default_rng((cfg.stat_seed, 17, code)).uniform(size=n)

    u = {
        "cal": draws(0, n_cal),
        "test": draws(1, len(ctx.split("test").y)),
        "ood_test": draws(2, len(ctx.split("ood_test").y)),
    }
    out: dict = {"alpha": cfg.alpha, "models": {}}
    for m in ctx.models:
        ctx.say(f"conformal: {m}")
        entry: dict = {}
        for kind in ("none", "temperature"):
            P_cal = ctx.proba(m, kind, "cal")
            per_variant = {}
            for variant, randomized in (("aps", True), ("aps_det", False)):
                model = conformal.ApsConformal(cfg.alpha, randomized).fit(
                    P_cal, cal.y, u["cal"] if randomized else None
                )
                row: dict = {"qhat": model.qhat}
                for split in SCORED:
                    sd = ctx.split(split)
                    P, draws_s = (
                        ctx.proba(m, kind, split),
                        (u[split] if randomized else None),
                    )
                    mask = model.sets(P, draws_s)
                    raw = model.sets(P, draws_s, nonempty=False)
                    rows = np.arange(len(sd.y))
                    hit, raw_hit = mask[rows, sd.y - 1], raw[rows, sd.y - 1]
                    row[split] = {
                        **conformal.set_stats(mask, sd.y),
                        "coverage_ci": _hit_ci(
                            hit, sd.groups, cfg.n_boot, cfg.stat_seed
                        ),
                        # the sets the guarantee covers, before empty ones get their top star
                        "coverage_raw": float(raw_hit.mean()),
                        "coverage_raw_ci": _hit_ci(
                            raw_hit, sd.groups, cfg.n_boot, cfg.stat_seed
                        ),
                        "empty_share": float((~raw.any(axis=1)).mean()),
                    }
                per_variant[variant] = row
            entry[kind] = per_variant
        pred_cal = ctx.preds()[m]["cal"]
        if pred_cal.framing == "regression" and pred_cal.is_single:
            mem = pred_cal.members[0]
            T = ctx.calibrators()[m]["temperature"].T
            intervals = {}
            for kind, scale in (("none", 1.0), ("temperature", T)):
                ic = conformal.IntervalConformal(cfg.alpha).fit(
                    mem.mu, mem.sigma * scale, cal.y
                )
                row = {"qhat": ic.qhat}
                for split in SCORED:
                    sd, mp = ctx.split(split), ctx.preds()[m][split].members[0]
                    lo, hi = ic.interval(mp.mu, mp.sigma * scale)
                    hit = (sd.y >= lo) & (sd.y <= hi)
                    row[split] = {
                        "coverage": float(hit.mean()),
                        "mean_width": float(np.mean(np.minimum(hi - lo, 1e6))),
                        "coverage_ci": _hit_ci(
                            hit, sd.groups, cfg.n_boot, cfg.stat_seed
                        ),
                    }
                intervals[kind] = row
            entry["interval"] = intervals
        out["models"][m] = entry
    return out, None


# ---------------------------------------------------------------- stage: shift


def _shift_stage(ctx: Ctx):

    cfg = ctx.cfg
    pool, test = ctx.split("ood_pool"), ctx.split("ood_test")
    out: dict = {"n_pool": len(pool.y), "n_test": len(test.y), "models": {}}
    for m in ctx.models:
        ctx.say(f"shift: {m}")
        p = ctx.preds()[m]
        out["models"][m] = shift.recalibration_curve(
            p["ood_pool"], pool.y, p["ood_test"], test.y,
            ns=cfg.recal_ns, draws=cfg.recal_draws, seed=cfg.stat_seed, bins=cfg.ece_bins,
        )  # fmt: skip
    return out, None


# ---------------------------------------------------------------- stage: selective


def _selective_stage(ctx: Ctx):

    cal = ctx.split("cal")
    out: dict = {"models": {}}
    for m in ctx.models:
        ctx.say(f"selective: {m}")
        entry = {}
        for kind in ctx.calibrators()[m]:
            conf_c, ok_c = metrics.top_label(ctx.proba(m, kind, "cal"), cal.y)
            row = {}
            for split in SCORED:
                sd = ctx.split(split)
                P = ctx.proba(m, kind, split)
                conf, ok = metrics.top_label(P, sd.y)
                loss = (~ok).astype(float)
                row[split] = {
                    "summary": selective.summarize(P, sd.y),
                    "transfer": selective.transfer(
                        conf_c, (~ok_c).astype(float), conf, loss, RISK_TARGETS
                    ),
                    "fixed": [
                        selective.at_threshold(conf, loss, tau) for tau in FIXED_TAUS
                    ],
                }
            entry[kind] = row
        out["models"][m] = entry
    return out, None


# ---------------------------------------------------------------- stage: ablation


def _ablation_stage(ctx: Ctx, provider):

    cfg = ctx.cfg
    tr, te = ctx.split("train"), ctx.split("test")
    rng = np.random.default_rng((cfg.stat_seed, 5))
    n_sub = min(cfg.ablation_train, len(tr.y))
    sub = np.sort(rng.choice(len(tr.y), n_sub, replace=False))
    perm = rng.permutation(n_sub)
    n_hold = max(1, n_sub // 10)
    hold, fit = np.sort(sub[perm[:n_hold]]), np.sort(sub[perm[n_hold:]])
    train_payload = ctx.payloads["train"]
    out: dict = {
        "n_fit": len(fit),
        "n_holdout": len(hold),
        "n_test": len(te.y),
        "modes": {},
    }
    for mode in ABLATION_MODES:
        ctx.say(f"ablation: {mode}")
        if mode == cfg.text_mode:
            X_tr, X_te = tr.X, te.X
        else:
            if provider is None:
                out["modes"][mode] = {
                    "skipped": "no embedding provider for this input-text setting"
                }
                continue
            emb = provider(mode, {"train": sub, "test": np.arange(len(te.y))})
            X_tr = np.zeros((len(tr.y), emb["train"].shape[1]), dtype=np.float32)
            X_tr[sub] = emb["train"]
            X_te = emb["test"]
        row = {}
        for framing in FRAMINGS:
            l2 = train_payload["models"][f"linear-{framing}"]["l2"]

            # the 64 "validation" rows are required by the signature but unused by a linear fit
            head = heads.fit_linear(
                framing, X_tr[fit], tr.y[fit], X_tr[fit][:64], tr.y[fit][:64], l2=l2
            )
            cell = {}
            for name, X, y in (
                ("holdout", X_tr[hold], tr.y[hold]),
                ("test", X_te, te.y),
            ):
                b = metrics.bundle(head.predict(X).proba(), y, cfg.ece_bins)
                cell[name] = {
                    k: b[k] for k in ("acc", "mae_argmax", "nll", "ece", "qwk")
                }
            row[framing] = cell
        out["modes"][mode] = row
    return out, None


# ---------------------------------------------------------------- driver


def _wanted_closure(stages) -> set[str]:

    wanted = set(stages)
    for s in list(wanted):
        stack = list(DEPENDS[s])
        while stack:
            d = stack.pop()
            if d not in wanted:
                wanted.add(d)
                stack.extend(DEPENDS[d])
    return wanted


def run_all(
    features: Features,
    results_dir,
    cfg: Config | None = None,
    *,
    stages=STAGES,
    fresh: bool = False,
    ablation_provider=None,
    progress: Callable[[str], None] | None = None,
) -> dict:

    cfg = cfg or Config()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)}; choose from {STAGES}")
    ctx = Ctx(features, cfg, results_dir, progress or (lambda _msg: None))
    fp = features.fingerprint()
    asked = set(stages)
    wanted = _wanted_closure(asked)
    keys: dict[str, str] = {}
    ran: list[str] = []
    runners = {
        "train": _train_stage,
        "calibrate": _calibrate_stage,
        "evaluate": _evaluate_stage,
        "conformal": _conformal_stage,
        "shift": _shift_stage,
        "selective": _selective_stage,
        "ablation": lambda c: _ablation_stage(c, ablation_provider),
    }
    for stage in STAGES:
        keys[stage] = stage_key(stage, cfg, fp, [keys[d] for d in DEPENDS[stage]])
        cached = (
            None
            if (fresh and stage in asked)
            else _read_stage(results_dir, stage, keys[stage])
        )

        # a stage is only trustworthy if everything upstream of it is also current
        upstream_ok = all(d in ctx.payloads for d in DEPENDS[stage])
        if cached is not None and upstream_ok:
            ctx.payloads[stage], arrays = cached
            if arrays is not None:
                ctx.arrays[stage] = arrays
            continue
        if stage not in wanted or not upstream_ok:
            continue
        ctx.say(f"== stage {stage} ==")
        started = time.time()
        payload, arrays = runners[stage](ctx)
        payload = clean(payload)
        payload["stage_seconds"] = round(time.time() - started, 1)
        _write_stage(results_dir, stage, keys[stage], payload, arrays)
        ctx.payloads[stage] = payload
        if arrays is not None:
            ctx.arrays[stage] = arrays
        if stage == "train":
            ctx._preds = None
        if stage == "calibrate":
            ctx._cals = None
        ran.append(stage)
    report = {
        "version": STAGE_VERSION,
        "config": cfg.to_dict(),
        "data": {
            "fingerprint": fp,
            "sizes": {s: len(d.y) for s, d in features.splits.items()},
            **clean(features.meta),
        },
        "models": ctx.models,
        "stages": {s: ctx.payloads[s] for s in STAGES if s in ctx.payloads},
        "missing": [s for s in STAGES if s not in ctx.payloads],
        "ran": ran,
    }
    (results_dir / "report.json").write_text(
        json.dumps(clean(report), indent=1), encoding="utf-8"
    )
    return report
