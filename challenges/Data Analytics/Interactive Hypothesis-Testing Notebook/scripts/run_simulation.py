"""Run the Monte Carlo study and write results/simulation.json and results/simulation.md.

Run from the challenge folder:  uv run python scripts/run_simulation.py
Defaults (4000 replicates, 1000 permutations for the permutation test, seed 20261027) are the
ones quoted in the README. Takes a few minutes.
"""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis_wizard import simulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPS = 4000
PERM_REPS = 1000
SEED = 20261027

PIPELINE_ORDER = (
    "wizard",
    "welch_always",
    "student_always",
    "mann_whitney_always",
    "brunner_munzel_always",
    "levene_then_student_or_welch",
    "shapiro_then_t_or_mwu",
    "permutation_mean",
)


def _cell(entry: dict | None) -> str:
    if entry is None:
        return "not run"
    return f"{entry['rate']:.3f} (±{entry['mc_se']:.3f})"


def to_markdown(study: dict) -> str:
    lines = [
        "# Simulation results",
        "",
        (
            f"Replicates per cell: {study['reps']}; permutation test: {study['perm_reps']} "
            f"relabellings on its scenarios only. Alpha = {study['alpha']}. Seed {study['seed']}."
        ),
        (
            "Each cell is the rejection rate with its Monte Carlo SE in brackets: "
            "sqrt(p (1 - p) / R). On null scenarios a rate is the type I error; on power "
            "scenarios it is power."
        ),
        "",
    ]
    for kind, title in (
        ("null", "Null scenarios (type I error)"),
        ("power", "Power scenarios"),
    ):
        keys = [
            k for k, v in study["scenarios"].items() if v["null"] == (kind == "null")
        ]
        lines += [f"## {title}", ""]
        header = (
            "| Pipeline | "
            + " | ".join(f"{study['scenarios'][k]['label']}" for k in keys)
            + " |"
        )
        lines += [header, "|" + "---|" * (len(keys) + 1)]
        for name in PIPELINE_ORDER:
            cells = [_cell(study["scenarios"][k]["rates"].get(name)) for k in keys]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    flip = study["flip_through"]
    if flip:
        lines += [
            "## Flip-through (null, normal, n 20/20)",
            "",
            f"Tests tried in order: {', '.join(flip['tests'])}.",
            "",
            "| Rule | Rejection rate (±MC SE) |",
            "|---|---|",
            f"| Reject if any test has p < 0.05 | {_cell(flip['naive_any_rejection'])} |",
            (
                f"| Reject if min p < 0.05 / 4 (counter rule) | "
                f"{_cell(flip['alpha_over_k_rejection'])} |"
            ),
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    study = simulate.run_study(reps=REPS, perm_reps=PERM_REPS, seed=SEED)
    study["seed"] = SEED
    (RESULTS / "simulation.json").write_text(
        json.dumps(study, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS / "simulation.md").write_text(to_markdown(study) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS / 'simulation.json'} and {RESULTS / 'simulation.md'}")


if __name__ == "__main__":
    main()
