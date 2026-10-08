"""Typer CLI: fetch and prepare Amazon Reviews 2023, embed it, run the calibration benchmark,
write the report, and score a single review.

    review-stars fetch | prepare | embed | train | calibrate | benchmark | report | predict

Any ``Config`` field can be overridden with ``--set key=value`` (lists as ``a,b,c``), for example
``--set n_boot=200`` or ``--set recal_ns=100,500,2500``.
"""

from __future__ import annotations

import dataclasses
import functools
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from review_stars import config, data, embed, pipeline, predict
from review_stars import report as report_mod
from review_stars.config import Config

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


def _safe_output() -> None:
    """Never crash while printing a review: on a stream that cannot encode a character (a
    redirected Windows console is cp1252), show ``?`` instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass  # not a real text stream (the test runner swaps these)


@app.callback()
def _main() -> None:
    _safe_output()


SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set", help="Override a Config field: key=value (lists as a,b,c). Repeatable."
    ),
]


def get_choice():
    """Indirection so tests can swap in a fake embedding backend."""
    return embed.default_choice()


def default_config() -> Config:
    """Indirection so tests can swap in a small configuration."""
    return Config()


def load_features(cfg: Config, progress=None):
    """Indirection so tests can swap in synthetic features."""
    return pipeline.load_features(cfg, config.data_dir(), _shared_choice(), progress)


@functools.cache
def _shared_choice_cached():
    return get_choice()


def _shared_choice():
    return lambda: _shared_choice_cached()


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


def apply_overrides(cfg: Config, items: list[str] | None) -> Config:
    fields = {f.name for f in dataclasses.fields(Config)}
    changes = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or key not in fields:
            raise typer.BadParameter(
                f"{item!r}: use key=value with one of the Config fields: {', '.join(sorted(fields))}"
            )
        current = getattr(cfg, key)
        try:
            if isinstance(current, tuple):
                kind = type(current[0]) if current else float
                changes[key] = tuple(kind(v) for v in value.split(","))
            else:
                changes[key] = type(current)(value)
        except ValueError as exc:
            raise typer.BadParameter(f"{item!r}: {exc}") from exc
    try:
        return dataclasses.replace(cfg, **changes)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _cfg(sets: list[str] | None) -> Config:
    try:
        return apply_overrides(default_config(), sets)
    except typer.BadParameter as exc:
        raise _fail(str(exc)) from exc


# ---------------------------------------------------------------- data commands


@app.command()
def fetch(sets: SetOption = None) -> None:
    """Download the two Amazon Reviews 2023 archives into data/raw (about 730 MB; resumable)."""
    cfg = _cfg(sets)
    try:
        paths = data.fetch(config.data_dir(), cfg)
    except (OSError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    for name, path in paths.items():
        typer.echo(f"{name}: {path} ({Path(path).stat().st_size / 1e6:.0f} MB)")


@app.command()
def prepare(sets: SetOption = None) -> None:
    """Index, de-duplicate, window and sample the archives into data/prepared; prints the month table."""
    cfg = _cfg(sets)
    names = (cfg.in_domain, cfg.out_domain)
    paths = {n: data.raw_path(config.data_dir(), n) for n in names}
    for p in paths.values():
        if not p.exists():
            raise _fail(f"{p} is missing; run `review-stars fetch` first")
    started = time.time()
    try:
        summary = data.prepare(paths, config.data_dir() / "prepared", cfg)
    except (OSError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    w = summary["windows"]
    typer.echo(
        f"prepared in {time.time() - started:.0f}s; windows counted back from {w['end']}"
    )
    for cat, counts in summary["month_counts"].items():
        tail = list(counts.items())[-14:]
        typer.echo(
            f"\n{cat}: reviews per month (after de-duplication), last {len(tail)} months"
        )
        typer.echo("  " + "  ".join(f"{m}:{n}" for m, n in tail))
    typer.echo(
        f"\nwindows: test {w['test'][0]}..{w['test'][1]}, calibration {w['cal'][0]}..{w['cal'][1]} "
        f"({w['cal_months']} months), validation {w['val'][0]}..{w['val'][1]} ({w['val_months']} months), "
        f"train before {w['train_before']}"
    )
    for note in w["notes"]:
        typer.echo(f"  note: {note}")
    typer.echo(
        "\nexact duplicates removed: "
        + ", ".join(f"{k} {v:,}" for k, v in summary["duplicates_removed"].items())
    )
    typer.echo("\nsplit        selected / available in window   auto 'N stars' titles")
    for split, info in summary["splits"].items():
        share = summary["auto_title_share_by_split"][split]
        typer.echo(
            f"{split:<10}{info['selected']:>10,} / {info['available']:<10,}"
            f"{'-' if share is None else format(share, '.1%'):>14}"
        )


@app.command("embed")
def embed_cmd(sets: SetOption = None) -> None:
    """Embed every prepared review once (shard-cached, resumable); shows the verified backend."""
    cfg = _cfg(sets)

    def progress(done, total):
        typer.echo(f"  shard {done}/{total}")

    started = time.time()
    try:
        features = load_features(cfg, progress)
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    info = features.meta["embedding"]
    typer.echo(f"embedding backend: {info['backend']} ({info.get('model', '')})")
    for name, verdict in info.get("tried", []):
        typer.echo(f"  {name}: {verdict}")
    shards = info["shards"]
    typer.echo(
        f"shards: {shards['computed']} computed, {shards['reused']} reused of {shards['total']}; "
        f"{time.time() - started:.0f}s"
    )
    sizes = ", ".join(f"{s} {len(d.y):,}" for s, d in features.splits.items())
    typer.echo(
        f"reviews: {sizes}; empty after cleaning: {features.meta.get('empty_texts', 0)}"
    )


# ---------------------------------------------------------------- the benchmark


def format_summary(report: dict) -> str:
    lines = [
        f"data fingerprint {report['data']['fingerprint']}; reviews per split: {report['data']['sizes']}"
    ]
    ev = report["stages"].get("evaluate")
    if ev:
        lines.append(
            f"\n{'model':<22}{'test ECE none':>14}{'temp':>8}{'test NLL':>10}{'Software ECE none':>19}{'temp':>8}{'NLL':>8}"
        )
        for m in report["models"]:
            t, o = ev["splits"]["test"]["rows"], ev["splits"]["ood_test"]["rows"]
            lines.append(
                f"{m:<22}{report_mod.fmt(t[f'{m}|none']['point']['ece']):>14}"
                f"{report_mod.fmt(t[f'{m}|temperature']['point']['ece']):>8}"
                f"{report_mod.fmt(t[f'{m}|temperature']['point']['nll']):>10}"
                f"{report_mod.fmt(o[f'{m}|none']['point']['ece']):>19}"
                f"{report_mod.fmt(o[f'{m}|temperature']['point']['ece']):>8}"
                f"{report_mod.fmt(o[f'{m}|temperature']['point']['nll']):>8}"
            )
    lines.append("\ncomputed this run: " + (", ".join(report["ran"]) or "none"))
    if report["missing"]:
        lines.append("not computed: " + ", ".join(report["missing"]))
    return "\n".join(lines)


def _run(stages, cfg: Config, fresh: bool = False) -> None:
    started = time.time()
    try:
        features = load_features(cfg, None)
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    provider = None
    if "ablation" in stages:
        try:
            provider = pipeline.make_ablation_provider(
                cfg, config.data_dir(), _shared_choice()
            )
        except FileNotFoundError:
            provider = None
    report = pipeline.run_all(
        features, config.results_dir(), cfg, stages=tuple(stages), fresh=fresh,
        ablation_provider=provider, progress=lambda m: typer.echo(f"  {m}"),
    )  # fmt: skip
    typer.echo(format_summary(report))
    typer.echo(f"\n{time.time() - started:.0f}s; results in {config.results_dir()}")


@app.command()
def train(sets: SetOption = None) -> None:
    """Fit the TF-IDF baselines and the six learned heads (and the MLP ensembles)."""
    _run(("train",), _cfg(sets))


@app.command()
def calibrate(sets: SetOption = None) -> None:
    """Fit every post-hoc calibrator on the calibration split (trains first if needed)."""
    _run(("calibrate",), _cfg(sets))


@app.command()
def benchmark(
    stage: Annotated[
        list[str] | None,
        typer.Option(help=f"Stage(s) to run: {', '.join(pipeline.STAGES)}."),
    ] = None,
    fresh: Annotated[
        bool, typer.Option(help="Recompute the requested stages.")
    ] = False,
    jobs: Annotated[
        int, typer.Option(min=1, help="Worker processes for the bootstrap.")
    ] = 1,
    sets: SetOption = None,
) -> None:
    """Run the whole study, resumable stage by stage; writes results/report.json."""
    stages = tuple(stage) if stage else pipeline.STAGES
    unknown = sorted(set(stages) - set(pipeline.STAGES))
    if unknown:
        raise _fail(f"unknown stage(s) {unknown}; choose from {pipeline.STAGES}")
    cfg = dataclasses.replace(_cfg(sets), n_jobs=jobs)
    _run(stages, cfg, fresh)


@app.command("report")
def report_cmd() -> None:
    """Write results/tables.md and the PNG figures from results/report.json."""
    try:
        written = report_mod.render_file(
            config.results_dir() / "report.json", config.results_dir()
        )
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    for path in written:
        typer.echo(str(path))


# ---------------------------------------------------------------- one review


def format_prediction(out: dict) -> str:
    text = out["text"] if len(out["text"]) <= 90 else out["text"][:87] + "..."
    lines = [
        f"review ({out['n_tokens']} tokens{', truncated to 512' if out['truncated'] else ''}): {text}",
        (
            f"\n{'framing':<16}{'model':<20}{'stars':>6}{'expected':>10}{'confidence':>12}"
            f"{f'{1 - out["alpha"]:.0%} set':>14}{'abstain':>10}"
        ),
    ]
    for framing, r in out["framings"].items():
        lines.append(
            f"{framing:<16}{r['model']:<20}{r['star']:>6}{r['expected']:>10.2f}{r['confidence']:>12.2f}"
            f"{'{' + ','.join(map(str, r['set'])) + '}':>14}{'yes' if r['abstain'] else 'no':>10}"
        )
    lines.append(
        f"\nabstain means confidence < {out['threshold']:.2f}. Probabilities of 1..5 stars:"
    )
    for framing, r in out["framings"].items():
        lines.append(
            f"  {framing:<16}"
            + "  ".join(f"{k + 1}: {p:.2f}" for k, p in enumerate(r["probs"]))
        )
    return "\n".join(lines)


@app.command("predict")
def predict_cmd(
    text: Annotated[str, typer.Argument(help="The review text.")],
    title: Annotated[str, typer.Option(help="The review title, if any.")] = "",
    threshold: Annotated[
        float, typer.Option(min=0.0, help="Abstain below this confidence.")
    ] = 0.8,
) -> None:
    """Predict stars for one review with calibrated confidence and a conformal set."""
    try:
        bundle = predict.load_bundle()
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    try:
        out = predict.predict_review(
            bundle, predict.get_encoder(), text, title, threshold
        )
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(format_prediction(out))


if __name__ == "__main__":  # pragma: no cover
    app()
