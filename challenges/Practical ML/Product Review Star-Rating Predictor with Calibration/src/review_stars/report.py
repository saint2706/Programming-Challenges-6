"""Markdown tables and PNG figures generated from ``report.json``.

Every number in the README comes from here: nothing is typed in by hand, and a stage that has not
been computed yet is said to be missing instead of silently omitted. Figures use matplotlib's
Agg backend, so rendering needs no display and no browser.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENS = ("ens-classification", "ens-regression", "ens-ordinal")
KINDS = ("none", "temperature", "vector", "isotonic")
SPLIT_TITLES = {"test": "In-time test", "ood_test": "Out-of-domain test (Software)"}
MAIN_COLUMNS = (
    ("acc", "accuracy"),
    ("mae_argmax", "MAE (stars)"),
    ("qwk", "QWK"),
    ("nll", "NLL"),
    ("rps", "RPS"),
    ("ece", "ECE"),
    ("smece", "smooth ECE"),
)


# ---------------------------------------------------------------- formatting


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt(x, spec: str = ".3f") -> str:
    return "–" if _missing(x) else format(x, spec)


def fmt_ci(ci, spec: str = ".3f") -> str:
    """``est [lo, hi]``; just ``est`` if the interval is unavailable; a dash if there is nothing.

    Accepts both shapes the results use: a bootstrap interval (``est``) and the spread over
    random draws in the recalibration study (``mean``)."""
    est = None if not ci else ci.get("est", ci.get("mean"))
    if _missing(est):
        return "–"
    if _missing(ci.get("lo")) or _missing(ci.get("hi")):
        return format(est, spec)
    return f"{est:{spec}} [{ci['lo']:{spec}}, {ci['hi']:{spec}}]"


def fmt_pct(x) -> str:
    return "–" if _missing(x) else f"{100 * x:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _stage(rep: dict, name: str):
    return rep.get("stages", {}).get(name)


def _need(rep: dict, name: str) -> str | None:
    if _stage(rep, name) is None:
        return f"_not computed yet: run `review-stars benchmark --stage {name}`_"
    return None


# ---------------------------------------------------------------- tables


def _data_section(rep: dict) -> str:
    sizes = rep["data"].get("sizes", {})
    prepared = rep["data"].get("prepared", {})
    splits = prepared.get("splits", {})
    share = prepared.get("auto_title_share_by_split", {})
    rows = []
    for s, n in sizes.items():
        info = splits.get(s, {})
        rows.append(
            [
                s,
                f"{n:,}",
                f"{info.get('available', 0):,}" if info else "–",
                fmt_pct(share.get(s)),
            ]
        )
    out = [
        _table(
            ["split", "reviews", "available in window", "auto 'N stars' titles"], rows
        )
    ]
    w = prepared.get("windows")
    if w:
        out.append(
            f"\nWindows (counted back from {w['end']}): test {w['test'][0]} to {w['test'][1]}, "
            f"calibration {w['cal'][0]} to {w['cal'][1]} ({w['cal_months']} months), "
            f"validation {w['val'][0]} to {w['val'][1]} ({w['val_months']} months), "
            f"train before {w['train_before']}."
        )
        out += [f"\n- {n}" for n in w.get("notes", [])]
    dup = prepared.get("duplicates_removed")
    if dup:
        out.append(
            "\nExact duplicates removed: "
            + ", ".join(f"{k} {v:,}" for k, v in dup.items())
            + "."
        )
    emb = rep["data"].get("embedding", {})
    if emb.get("backend"):
        out.append(f"\nEmbeddings: {emb.get('model', '')} on {emb['backend']}.")
    return "\n".join(out)


def _main_section(rep: dict, split: str) -> str:
    if (msg := _need(rep, "evaluate")) is not None:
        return msg
    rows_by_key = _stage(rep, "evaluate")["splits"][split]["rows"]
    out = []
    for kind in ("none", "temperature"):
        out.append(
            f"\n**{'Uncalibrated' if kind == 'none' else 'After temperature scaling'}**\n"
        )
        rows = [
            [m, *(fmt_ci(rows_by_key[f"{m}|{kind}"]["ci"][k]) for k, _ in MAIN_COLUMNS)]
            for m in rep["models"]
            if f"{m}|{kind}" in rows_by_key
        ]
        out.append(_table(["model", *(name for _, name in MAIN_COLUMNS)], rows))
    return "\n".join(out)


def _calibrators_section(rep: dict) -> str:
    if (msg := _need(rep, "evaluate")) is not None:
        return msg
    out = []
    for split, title in SPLIT_TITLES.items():
        rows_by_key = _stage(rep, "evaluate")["splits"][split]["rows"]
        out.append(f"\n**{title}: ECE and NLL by calibrator**\n")
        rows = []
        for m in rep["models"]:
            cells = []
            for metric in ("ece", "nll"):
                for kind in KINDS:
                    row = rows_by_key.get(f"{m}|{kind}")
                    cells.append(fmt(row["point"][metric]) if row else "–")
            rows.append([m, *cells])
        header = ["model", *(f"ECE {k}" for k in KINDS), *(f"NLL {k}" for k in KINDS)]
        out.append(_table(header, rows))
    return "\n".join(out)


def _diffs_section(rep: dict) -> str:
    if (msg := _need(rep, "evaluate")) is not None:
        return msg
    out = []
    for split, title in SPLIT_TITLES.items():
        diffs = [
            d
            for d in _stage(rep, "evaluate")["splits"][split]["diffs"]
            if d["metric"] in ("ece", "nll")
        ]
        rows = []
        for d in diffs:
            excludes_zero = not _missing(d["lo"]) and (d["lo"] > 0 or d["hi"] < 0)
            rows.append(
                [
                    d["label"],
                    d["metric"],
                    fmt_ci(d, ".4f"),
                    "yes" if excludes_zero else "no",
                ]
            )
        out.append(f"\n**{title}** (paired cluster bootstrap; a minus b)\n")
        out.append(
            _table(["comparison", "metric", "difference", "interval excludes 0"], rows)
        )
    return "\n".join(out)


def _conformal_section(rep: dict) -> str:
    if (msg := _need(rep, "conformal")) is not None:
        return msg
    st = _stage(rep, "conformal")
    out = [
        f"\nTarget coverage {1 - st['alpha']:.0%}; sets built from temperature-scaled probabilities.\n"
    ]
    rows = []
    for m, entry in st["models"].items():
        v = entry["temperature"]
        rows.append(
            [
                m,
                fmt_ci(v["aps"]["test"]["coverage_ci"]), fmt(v["aps"]["test"]["mean_size"], ".2f"),
                fmt_ci(v["aps"]["ood_test"]["coverage_ci"]), fmt(v["aps"]["ood_test"]["mean_size"], ".2f"),
                fmt(v["aps_det"]["test"]["coverage"]), fmt(v["aps_det"]["test"]["mean_size"], ".2f"),
            ]
    )  # fmt: skip
    out.append(
        _table(
            [
                "model",
                "APS coverage (test)",
                "size",
                "APS coverage (Software)",
                "size",
                "deterministic APS coverage (test)",
                "size",
            ],
            rows,
        )
    )
    reg = [(m, e["interval"]) for m, e in st["models"].items() if "interval" in e]
    if reg:
        out.append("\n**Regression intervals** (normalized-residual conformal)\n")
        rows = [
            [m, kind, fmt_ci(iv[kind]["test"]["coverage_ci"]), fmt(iv[kind]["test"]["mean_width"], ".2f"),
             fmt_ci(iv[kind]["ood_test"]["coverage_ci"]), fmt(iv[kind]["ood_test"]["mean_width"], ".2f")]
            for m, iv in reg
            for kind in ("none", "temperature")
        ]  # fmt: skip
        out.append(
            _table(
                [
                    "model",
                    "sigma",
                    "coverage (test)",
                    "width",
                    "coverage (Software)",
                    "width",
                ],
                rows,
            )
        )
    return "\n".join(out)


def _recal_section(rep: dict) -> str:
    if (msg := _need(rep, "shift")) is not None:
        return msg
    models = [m for m in ENS if m in _stage(rep, "shift")["models"]] or list(
        _stage(rep, "shift")["models"]
    )[:3]
    data = _stage(rep, "shift")["models"]
    ns = sorted({r["n"] for m in models for r in data[m]["rows"]})
    header = [
        "Software labels",
        *(f"{m} ECE" for m in models),
        *(f"{m} T" for m in models),
    ]
    rows = []
    for n in ns:
        cells = []
        for key in ("ece", "T"):
            for m in models:
                row = next((r for r in data[m]["rows"] if r["n"] == n), None)
                cells.append(fmt_ci(row[key]) if row else "–")
        rows.append([f"{n:,}", *cells])
    rows.append(
        [
            "none (uncalibrated)",
            *(fmt(data[m]["uncalibrated"]["ece"]) for m in models),
            *["–"] * len(models),
        ]
    )
    rows.append(
        [
            "all pool labels",
            *(fmt(data[m]["full_pool"]["ece"]) for m in models),
            *(fmt(data[m]["full_pool"]["T"]) for m in models),
        ]
    )
    return _table(header, rows)  # fmt: skip


def _selective_section(rep: dict) -> str:
    if (msg := _need(rep, "selective")) is not None:
        return msg
    models = _stage(rep, "selective")["models"]
    rows = []
    for m in rep["models"]:
        cells = []
        for kind in ("none", "temperature"):
            for split in ("test", "ood_test"):
                cells.append(fmt(models[m][kind][split]["summary"]["error"]["aurc"]))
        rows.append([m, *cells])
    out = [
        "\nArea under the risk-coverage curve for the 0/1 error (lower is better).\n",
        _table(["model", "none (test)", "none (Software)", "temperature (test)", "temperature (Software)"], rows),
        "\n**Keeping only reviews whose stated confidence is at least tau** (accuracy of what is kept)\n",
    ]  # fmt: skip
    rows = []
    for m in [x for x in ENS if x in models]:
        for kind in ("none", "temperature"):
            for split in ("test", "ood_test"):
                for f in models[m][kind][split]["fixed"]:
                    if f["tau"] in (0.7, 0.9):
                        rows.append(
                            [
                                m,
                                kind,
                                split,
                                f"{f['tau']:.1f}",
                                fmt_pct(f["coverage"]),
                                fmt_pct(None if _missing(f["risk"]) else 1 - f["risk"]),
                            ]
                        )
    out.append(
        _table(
            ["model", "confidence", "split", "tau", "kept", "accuracy of kept"], rows
        )
    )
    return "\n".join(out)  # fmt: skip


def _ablation_section(rep: dict) -> str:
    if (msg := _need(rep, "ablation")) is not None:
        return msg
    st = _stage(rep, "ablation")
    out = [
        (
            f"\nLinear heads on {st['n_fit']:,} training reviews, scored on "
            f"{st['n_holdout']:,} held-out reviews from the same era and on the in-time "
            f"test ({st['n_test']:,}); no calibrator.\n"
        )
    ]
    rows = []
    for mode, row in st["modes"].items():
        if "skipped" in row:
            rows.append([mode, "–", "skipped: " + row["skipped"], *["–"] * 5])
            continue
        for framing, cell in row.items():
            values = [
                fmt(cell[s][k])
                for s in ("holdout", "test")
                for k in ("acc", "nll", "ece")
            ]
            rows.append([mode, framing, *values])
    out.append(
        _table(
            [
                "input text",
                "framing",
                "same-era acc",
                "NLL",
                "ECE",
                "test acc",
                "NLL",
                "ECE",
            ],
            rows,
        )
    )
    return "\n".join(out)


def _slices_section(rep: dict) -> str:
    if (msg := _need(rep, "evaluate")) is not None:
        return msg
    out = []
    for split, title in SPLIT_TITLES.items():
        slices = _stage(rep, "evaluate")["splits"][split]["slices"]
        for key in ("ens-classification|temperature", "ens-ordinal|temperature"):
            if key not in slices:
                continue
            out.append(f"\n**{title}: {key}**\n")
            rows = [
                [name, f"{s['n']:,}", *([fmt(s["acc"]), fmt(s["nll"]), fmt(s["ece"])] if "acc" in s else ["–", "–", "–"])]
                for name, s in slices[key].items()
            ]  # fmt: skip
            out.append(_table(["slice", "reviews", "accuracy", "NLL", "ECE"], rows))
    return "\n".join(out)


SECTIONS = (
    ("## Data and windows", _data_section),
    ("## In-time test", lambda r: _main_section(r, "test")),
    ("## Out-of-domain test (Software)", lambda r: _main_section(r, "ood_test")),
    ("## Calibrators", _calibrators_section),
    ("## Paired differences", _diffs_section),
    ("## Conformal prediction", _conformal_section),
    ("## Recalibration budget", _recal_section),
    ("## Selective prediction", _selective_section),
    ("## Input-text ablation", _ablation_section),
    ("## Slices", _slices_section),
)


def tables_markdown(rep: dict) -> str:
    parts = [
        "# Results tables\n",
        "Generated by `review-stars report` from `results/report.json`.\n",
    ]
    if rep.get("missing"):
        parts.append(f"Stages not computed yet: {', '.join(rep['missing'])}.\n")
    for heading, build in SECTIONS:
        parts.append(f"{heading}\n\n{build(rep)}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------- figures


def _models(rep: dict) -> list[str]:
    found = [m for m in ENS if m in rep["models"]]
    return found or rep["models"][:3]


def _save(fig, path: Path, tight: bool = True) -> Path:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _floats(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _fig_reliability(rep: dict, path: Path) -> Path:
    """Reliability diagrams with bootstrap bands; the strip under each is the confidence histogram
    (how sharp the model is)."""
    ev = _stage(rep, "evaluate")["splits"]
    models = _models(rep)
    fig = plt.figure(figsize=(4.2 * len(models), 8.8))
    grid = fig.add_gridspec(4, len(models), height_ratios=[3, 1, 3, 1], hspace=0.55)
    for i, split in enumerate(("test", "ood_test")):
        for j, m in enumerate(models):
            ax = fig.add_subplot(grid[2 * i, j])
            hist = fig.add_subplot(grid[2 * i + 1, j], sharex=ax)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            for shift_x, kind, color in (
                (-0.012, "none", "C0"),
                (0.012, "temperature", "C1"),
            ):
                rel = ev[split]["rows"][f"{m}|{kind}"]["reliability_fixed"]
                edges = _floats(rel["edges"])
                centers = (edges[:-1] + edges[1:]) / 2
                acc, lo, hi, n = (
                    _floats(rel["acc"]),
                    _floats(rel["lo"]),
                    _floats(rel["hi"]),
                    _floats(rel["n"]),
                )
                ok = ~np.isnan(acc)
                banded = ok & ~np.isnan(lo) & ~np.isnan(hi)
                ax.fill_between(
                    centers[banded], lo[banded], hi[banded], color=color, alpha=0.18
                )
                ax.plot(centers[ok], acc[ok], "o-", ms=4, color=color, label=kind)
                hist.bar(
                    centers + shift_x, n / max(n.sum(), 1), width=0.02, color=color
                )
            ax.set(
                title=f"{m}: {SPLIT_TITLES[split].split(' (')[0].lower()}",
                ylabel="accuracy",
                xlim=(0, 1),
                ylim=(0, 1.02),
            )
            ax.tick_params(labelbottom=False)
            ax.legend(loc="upper left", fontsize=8)
            hist.set(xlabel="confidence", ylabel="share")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.06, wspace=0.28)
    return _save(fig, path, tight=False)


def _fig_coverage(rep: dict, path: Path) -> Path:
    ev = _stage(rep, "evaluate")["splits"]
    models = _models(rep)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, split in zip(axes, ("test", "ood_test"), strict=True):
        ax.plot([0.4, 1], [0.4, 1], "k--", lw=1)
        for m in models:
            for kind, style in (("none", ":"), ("temperature", "-")):
                cov = ev[split]["rows"][f"{m}|{kind}"]["coverage"]
                ax.plot(
                    [c["level"] for c in cov],
                    [c["coverage"] for c in cov],
                    style,
                    marker="o",
                    ms=3,
                    label=f"{m} {kind}",
                )
        ax.set(
            title=SPLIT_TITLES[split],
            xlabel="nominal mass of the smallest set",
            ylabel="empirical coverage",
        )
    axes[0].legend(fontsize=6)
    return _save(fig, path)  # fmt: skip


def _fig_recal(rep: dict, path: Path) -> Path:
    data = _stage(rep, "shift")["models"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in _models(rep):
        rows = data[m]["rows"]
        ns = [r["n"] for r in rows]
        mean = [r["ece"]["mean"] for r in rows]
        ax.plot(ns, mean, "o-", label=m)
        ax.fill_between(
            ns,
            [r["ece"]["lo"] for r in rows],
            [r["ece"]["hi"] for r in rows],
            alpha=0.15,
        )
        ax.axhline(
            data[m]["uncalibrated"]["ece"], ls=":", lw=1, color=ax.lines[-1].get_color()
        )
    ax.set(xscale="log", xlabel="labeled Software reviews used to refit T", ylabel="ECE on Software test",
           title="Recalibration budget (dotted: uncalibrated)")  # fmt: skip
    ax.legend(fontsize=8)
    return _save(fig, path)


def _fig_risk(rep: dict, path: Path) -> Path:
    data = _stage(rep, "selective")["models"]
    models = _models(rep)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, split in zip(axes, ("test", "ood_test"), strict=True):
        for m in models:
            for kind, style in (("none", ":"), ("temperature", "-")):
                curve = data[m][kind][split]["summary"]["error"]["curve"]
                ax.plot(curve["coverage"], curve["risk"], style, label=f"{m} {kind}")
        ax.set(
            title=SPLIT_TITLES[split],
            xlabel="coverage (share of reviews kept)",
            ylabel="error rate of what is kept",
        )
    axes[0].legend(fontsize=6)
    return _save(fig, path)  # fmt: skip


def _fig_framings(rep: dict, path: Path) -> Path:
    ev = _stage(rep, "evaluate")["splits"]
    models = rep["models"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    x = range(len(models))
    for ax, split in zip(axes, ("test", "ood_test"), strict=True):
        for off, kind in ((-0.2, "none"), (0.2, "temperature")):
            vals = [ev[split]["rows"][f"{m}|{kind}"]["point"]["ece"] for m in models]
            ax.bar([i + off for i in x], vals, width=0.4, label=kind)
        ax.set(title=SPLIT_TITLES[split], ylabel="ECE")
        ax.set_xticks(list(x), models, rotation=70, ha="right", fontsize=8)
    axes[0].legend()
    return _save(fig, path)


FIGURES = (
    ("reliability.png", "evaluate", _fig_reliability),
    ("coverage.png", "evaluate", _fig_coverage),
    ("recalibration.png", "shift", _fig_recal),
    ("risk_coverage.png", "selective", _fig_risk),
    ("framings.png", "evaluate", _fig_framings),
)


def render(rep: dict, out_dir) -> list[Path]:
    """Write ``tables.md`` and every figure whose stage has been computed; return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "tables.md"
    md.write_text(tables_markdown(rep), encoding="utf-8")
    written = [md]
    for name, stage, draw in FIGURES:
        if _stage(rep, stage) is not None:
            written.append(draw(rep, out_dir / name))
    return written


def render_file(report_path, out_dir) -> list[Path]:
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} does not exist; run `review-stars benchmark` first"
        )
    return render(json.loads(report_path.read_text(encoding="utf-8")), out_dir)
