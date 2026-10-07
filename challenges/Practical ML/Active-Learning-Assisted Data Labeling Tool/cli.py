"""Typer CLI: fetch Banking77, embed it, benchmark the query strategies, and run a labeling project."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import data
import embed
import explain
import pipeline
import project as project_mod
import strategies
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


def get_choice():
    """Indirection so tests can swap in a fake embedding backend."""
    return embed.default_choice()


def load_problem():
    """Indirection so tests can swap in a small problem."""
    return pipeline.load_problem(data.DATA_DIR, get_choice=lambda: get_choice())


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


def _open(path: Path) -> project_mod.Project:
    try:
        return project_mod.Project(path)
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc


@app.command()
def fetch() -> None:
    """Download Banking77 (CSV files from PolyAI's GitHub, ~1 MB) into data/."""
    try:
        typer.echo(f"data in {data.fetch()}")
    except OSError as exc:
        raise _fail(str(exc)) from exc


@app.command("embed")
def embed_cmd() -> None:
    """Split and embed Banking77 (cached): shows the verified backend and the chosen head C."""
    try:
        problem, _, info = load_problem()
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(
        f"pool {problem.X.shape[0]:,} x {problem.X.shape[1]}, validation {len(problem.X_val):,}, "
        f"test {len(problem.X_eval):,}, {problem.n_classes} intents"
    )
    typer.echo(f"embedding backend: {info['backend']} ({info.get('model', '')})")
    for name, verdict in info.get("tried", []):
        typer.echo(f"  {name}: {verdict}")
    typer.echo(f"head regularization C = {problem.C}")


def _ci(c, scale=1.0, spec=".3f") -> str:
    if not c or c.get("est") is None or c["est"] != c["est"]:
        return "-"
    est, lo, hi = (c[k] * scale for k in ("est", "lo", "hi"))
    return f"{est:{spec}} [{lo:{spec}}, {hi:{spec}}]"


def format_report(report: dict) -> str:
    d, cfg = report["data"], report["config"]
    lines = [
        (
            f"pool {d['pool']:,}, validation {d['val']:,}, test {d['eval']:,}, "
            f"{d['classes']} intents; head C = {d['C']}; embeddings via {d.get('backend', '?')}"
        ),
        (
            f"ceiling (head trained on all {d['pool']:,} pool labels): test accuracy "
            f"{report['ceiling']['accuracy']:.3f}"
        ),
    ]
    if "stop" in report:
        r = report["stop"]["rule"]
        lines.append(
            f"stopping rule (tuned on {cfg['tune_seeds']} validation seeds): prediction change < "
            f"{r['threshold']} for {r['k']} rounds; stops at {r['labels']:.0f} labels on average, "
            f"leaving {r['acc_gap']:.3f} accuracy (tolerance {r['tol']})"
        )
    main = report.get("main")
    if main:
        lines.append(
            f"\n== batch size {cfg['batch']}, {cfg['seeds']} seeds, "
            f"{cfg['init']} cold-start + {cfg['budget']} queried labels =="
        )
        lines.append(
            f"{'strategy':<17}{'accuracy @ end':>24}{'ALC':>24}{'ALC - random':>26}"
            f"{'labels to 90%':>16}{'saving vs random':>26}{'saving @95%':>26}"
        )
        for name, e in main["summary"].items():
            t90, t95 = e["targets"]["0.90"], e["targets"]["0.95"]
            labels = (
                "-"
                if t90["labels"] is None
                else f"{t90['labels']:.0f} ({t90['reached']}/{e['seeds']})"
            )
            lines.append(
                f"{name:<17}{_ci(e['acc_final']):>24}{_ci(e['alc']):>24}{_ci(e['alc_vs_random']):>26}"
                f"{labels:>16}{_ci(t90['saving_vs_random'], spec='.0f'):>26}"
                f"{_ci(t95['saving_vs_random'], spec='.0f'):>26}"
            )
        lines.append(
            f"\n{'strategy':<17}{'intents':>9}{'top share':>11}{'err on picks':>14}{'err on pool':>13}"
            f"{'select s/round':>16}{'stop labels':>13}{'acc left':>10}"
        )
        for name, e in main["summary"].items():
            stop = e.get("stop")
            lines.append(
                f"{name:<17}{e['coverage_final']:>9.1f}{e['skew_final']:>11.3f}"
                f"{(e['sel_err'] or 0):>14.3f}{(e['pool_err'] or 0):>13.3f}"
                f"{(e['select_seconds'] or 0):>16.3f}"
                f"{('-' if not stop else format(stop['labels'], '.0f')):>13}"
                f"{('-' if not stop else format(stop['acc_gap']['est'], '.3f')):>10}"
            )
    for b, block in (report.get("batch") or {}).items():
        lines.append(f"\n== batch size {b} (sensitivity, {cfg['sens_seeds']} seeds) ==")
        lines.append(
            f"{'strategy':<17}{'accuracy @ end':>24}{'ALC':>24}{'ALC - random':>26}"
        )
        for name, e in block["summary"].items():
            lines.append(
                f"{name:<17}{_ci(e['acc_final']):>24}{_ci(e['alc']):>24}{_ci(e['alc_vs_random']):>26}"
            )
    if report["missing"]:
        lines.append("\nnot computed: " + ", ".join(report["missing"]))
    return "\n".join(lines)


@app.command()
def benchmark(
    seeds: Annotated[
        int, typer.Option(min=1, help="Seeds for the main comparison.")
    ] = 10,
    budget: Annotated[int, typer.Option(min=1, help="Queried labels per run.")] = 1500,
    batch: Annotated[int, typer.Option(min=1, help="Batch size per round.")] = 50,
    stage: Annotated[
        list[str] | None, typer.Option(help="Stage(s) to run: main, batch, stop.")
    ] = None,
    fresh: Annotated[
        bool, typer.Option(help="Recompute the requested stages.")
    ] = False,
    jobs: Annotated[int, typer.Option(min=1, help="Parallel worker processes.")] = 8,
) -> None:
    """Run the strategies against random sampling; writes results/report.json (resumable)."""
    stages = tuple(stage) if stage else pipeline.STAGES
    unknown = set(stages) - set(pipeline.STAGES)
    if unknown:
        raise _fail(
            f"unknown stage(s) {sorted(unknown)}; choose from {pipeline.STAGES}"
        )
    try:
        cfg = pipeline.Config(seeds=seeds, budget=budget, batch=batch)
        problem, _, info = load_problem()
        report = pipeline.run_all(
            problem,
            pipeline.RESULTS_DIR,
            cfg,
            stages=stages,
            fresh=fresh,
            n_jobs=jobs,
            meta=info,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(format_report(report))


def _stop_rule() -> dict | None:
    """The tuned stopping rule from the last benchmark, if there is one."""
    import json

    path = pipeline.RESULTS_DIR / "report.json"
    if not path.exists():
        return None
    rule = json.loads(path.read_text(encoding="utf-8")).get("stop", {}).get("rule")
    if not rule:
        return None
    spacing = json.loads(path.read_text(encoding="utf-8"))["config"]["batch"]
    return {"threshold": rule["threshold"], "k": rule["k"], "spacing": spacing}


@app.command()
def init(
    path: Annotated[
        Path, typer.Argument(metavar="PROJECT", help="New project folder.")
    ],
    demo: Annotated[
        bool, typer.Option(help="Banking77 pool with gold labels.")
    ] = False,
    csv: Annotated[
        Path | None, typer.Option(help="CSV with the texts to label.")
    ] = None,
    classes: Annotated[
        Path | None, typer.Option(help="Class list, one per line.")
    ] = None,
    label_col: Annotated[
        str | None, typer.Option(help="CSV column with gold labels.")
    ] = None,
    text_col: Annotated[str, typer.Option(help="CSV column with the texts.")] = "text",
    c: Annotated[
        float, typer.Option("--c", min=0.0, help="Head regularization.")
    ] = 10.0,
) -> None:
    """Create a labeling project from the Banking77 demo pool or from your own CSV."""
    if demo == (csv is not None):
        raise _fail("give exactly one of --demo or --csv FILE")
    if path.exists():
        raise _fail(f"{path} already exists")
    try:
        if demo:
            problem, splits, _ = load_problem()
            names = splits.classes
            created = project_mod.Project.create(
                path,
                splits.pool_texts,
                problem.X,
                names,
                gold=[names[i] for i in splits.pool_y],
                C=problem.C,
                rule=_stop_rule(),
                X_eval=problem.X_eval,
                y_eval=problem.y_eval,
            )
        else:
            texts, gold = data.load_pool_csv(csv, text_col, label_col)
            if classes is not None:
                names = data.read_classes(classes)
            elif gold is not None:
                names = sorted({g for g in gold if g})
                if len(names) < 2:
                    raise ValueError(
                        f"{csv}: --label-col has fewer than two distinct classes; pass --classes"
                    )
            else:
                raise ValueError(
                    "give --classes FILE (or --label-col to take the classes from the data)"
                )
            unknown = sorted({g for g in gold or [] if g and g not in names})
            if unknown:
                raise ValueError(f"gold labels {unknown[:3]} are not in the class list")
            X, _ = embed.embed_cached(texts, data.DATA_DIR / "embeddings", get_choice)
            created = project_mod.Project.create(
                path, texts, X, names, gold=gold, C=c, rule=_stop_rule()
            )
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(
        f"created {path}: {len(created.texts):,} items, {len(created.classes)} classes"
        + (", gold labels and an evaluation set" if created.X_eval is not None else "")
    )


@app.command()
def suggest(
    path: Annotated[Path, typer.Argument(metavar="PROJECT")],
    strategy: Annotated[
        str, typer.Option(help=f"One of {', '.join(strategies.NAMES)}.")
    ] = "margin",
    batch: Annotated[int, typer.Option(min=1, help="How many items to suggest.")] = 10,
    seed: Annotated[int, typer.Option(help="Tie-break / sampling seed.")] = 0,
) -> None:
    """Suggest which unlabeled items to label next, and why."""
    proj = _open(path)
    try:
        selection, reasons, _ = proj.suggest(strategy, batch, seed)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    if not len(selection.idx):
        typer.echo("every item is labeled")
        return
    for r in reasons:
        typer.echo(f"#{r.idx}  {proj.texts[r.idx]}")
        for line in explain.render(r, proj.classes, proj.texts).splitlines():
            typer.echo(f"    {line}")
    typer.echo(f"\nlabel one with: python cli.py label {path} <id> <class>")


@app.command()
def label(
    path: Annotated[Path, typer.Argument(metavar="PROJECT")],
    item: Annotated[int, typer.Argument(help="Item id, as printed by suggest.")],
    label_name: Annotated[str, typer.Argument(metavar="CLASS")],
) -> None:
    """Label one item (the last label wins; earlier ones stay in the history)."""
    proj = _open(path)
    try:
        proj.submit({item: label_name})
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"labeled #{item} as {label_name}; {proj.status()['labeled']} labels")


@app.command()
def status(path: Annotated[Path, typer.Argument(metavar="PROJECT")]) -> None:
    """Labels so far, intent coverage, per-round accuracy and the stopping signal."""
    s = _open(path).status()
    typer.echo(f"{s['labeled']} of {s['items']} items labeled")
    typer.echo(
        f"{s['classes_with_label']} of {s['classes']} intents have at least one label"
    )
    for i, r in enumerate(s["rounds"], 1):
        acc = "-" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
        chg = "-" if r["change"] is None else f"{r['change']:.3f}"
        typer.echo(
            f"round {i}: {r['n_labeled']} labels, accuracy {acc}, prediction change {chg}"
        )
    rule = s["rule"]
    state = (
        f"fired at {s['stop']['at_labels']} labels"
        if s["stop"]["fired"]
        else "not reached yet"
    )
    typer.echo(
        f"stopping signal (prediction change < {rule['threshold']} for {rule['k']} rounds, "
        f"measured between rounds {rule['spacing']}+ labels apart): {state}"
    )


@app.command()
def export(
    path: Annotated[Path, typer.Argument(metavar="PROJECT")],
    output: Annotated[Path, typer.Argument(help="CSV to write.")],
) -> None:
    """Write id, text, current label and gold label for every item."""
    _open(path).export(output)
    typer.echo(f"wrote {output}")


if __name__ == "__main__":
    app()
