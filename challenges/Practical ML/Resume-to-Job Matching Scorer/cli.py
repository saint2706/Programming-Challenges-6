"""Typer CLI: fetch the data, run the benchmark, rank jobs for a resume (or the reverse)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import data
import pipeline
import typer
from ranker import RESULTS, SCORER_NAMES, Match, Ranker, load_default_ranker

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

SCORER_ORDER = ("random", "tfidf", "bm25", "embedding", "fusion", "rrf")
SCORER_HELP = f"One of: {', '.join(SCORER_NAMES)}."


def get_ranker() -> Ranker:
    """Indirection so tests (and the app) can swap in a small ranker."""
    return load_default_ranker()


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


def _read(path: Path, what: str) -> str:
    if not path.is_file():
        raise _fail(f"{what} file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise _fail(f"{what} file is empty: {path}")
    return text


def _check_scorer(scorer: str) -> None:
    if scorer not in SCORER_NAMES:
        raise _fail(f"unknown scorer {scorer!r}; choose from {', '.join(SCORER_NAMES)}")


def _print_matches(matches: list[Match], gaps: bool) -> None:
    for rank, m in enumerate(matches, 1):
        typer.echo(f"{rank:>2}. {m.score:7.3f}  {m.title} [{m.category}]  (id {m.id})")
        terms = ", ".join(f"{t} {w:.3f}" for t, w in m.terms) or "none"
        if m.explanation == "exact":
            typer.echo(f"      evidence: {terms}   (these terms sum to the score)")
        else:
            typer.echo(
                f"      evidence: {terms}   (lexical overlap only; the score came from embeddings)"
            )
        if gaps and m.gaps:
            typer.echo(f"      missing:  {', '.join(m.gaps)}")


@app.command("rank-jobs")
def rank_jobs(
    resume_file: Annotated[Path, typer.Argument(help="Plain-text resume.")],
    top: Annotated[int, typer.Option(min=1, help="How many jobs to list.")] = 10,
    scorer: Annotated[str, typer.Option(help=SCORER_HELP)] = "fusion",
    explain_dense: Annotated[
        bool,
        typer.Option("--explain-dense", help="Probe the embedding match (slower)."),
    ] = False,
) -> None:
    """Rank job postings for a resume."""
    _check_scorer(scorer)
    text = _read(resume_file, "resume")
    ranker = get_ranker()
    try:
        matches = ranker.rank_jobs(text, top=top, scorer=scorer)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    _print_matches(matches, gaps=True)
    if explain_dense and matches:
        typer.echo(
            "\npost-hoc: resume sentences the embedding match to the top job depends on"
        )
        typer.echo(
            "(re-scored with each sentence removed; a probe of the model, not a decomposition)"
        )
        for sentence, drop in ranker.explain_dense(text, matches[0].id):
            typer.echo(f"  {drop:+.4f}  {sentence}")


@app.command("rank-resumes")
def rank_resumes(
    job_file: Annotated[Path, typer.Argument(help="Plain-text job description.")],
    top: Annotated[int, typer.Option(min=1, help="How many resumes to list.")] = 10,
    scorer: Annotated[str, typer.Option(help=SCORER_HELP)] = "fusion",
) -> None:
    """Rank the resume collection for a job description."""
    _check_scorer(scorer)
    text = _read(job_file, "job")
    try:
        matches = get_ranker().rank_resumes(text, top=top, scorer=scorer)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    _print_matches(matches, gaps=False)


def _interval(metric: dict) -> str:
    return f"{metric['mean']:.3f} [{metric['lo']:.3f}, {metric['hi']:.3f}]"


def format_report(report: dict) -> str:
    n = report["n"]
    weights = ", ".join(f"{v} {w:.2f}" for v, w in report["fusion_weight"].items())
    lines = [
        (
            f"test split: {n['test_resumes']} resumes x {n['test_jobs']} postings   "
            f"embedding backend: {report['backend']['name']}"
        ),
        f"fusion weight on the embedding (tuned on validation): {weights}",
    ]
    if report["skipped_categories"]:
        lines.append(
            f"skipped categories (too few postings): {report['skipped_categories']}"
        )
    for variant, directions in report["results"].items():
        for direction, scorers in directions.items():
            lines += [
                "",
                f"{direction}, {variant} resume text",
                f"{'scorer':<10}{'nDCG@10 [95% CI]':<26}{'MRR [95% CI]':<26}P@10    MAP@50",
            ]
            for name in SCORER_ORDER:
                m = scorers[name]
                lines.append(
                    f"{name:<10}{_interval(m['ndcg@10']):<26}{_interval(m['mrr']):<26}"
                    f"{m['p@10']['mean']:.3f}  {m['map@50']['mean']:.3f}"
                )
    return "\n".join(lines)


@app.command()
def fetch() -> None:
    """Download the resume and job-posting CSVs from Kaggle (token in ~/.kaggle/kaggle.json)."""
    data.fetch()
    typer.echo(f"data in {data.DATA_DIR}")


@app.command()
def evaluate() -> None:
    """Run the full benchmark and write results/report.json (embeds everything; minutes)."""
    typer.echo(format_report(pipeline.run_all(data.DATA_DIR, RESULTS.parent)))


@app.command()
def report() -> None:
    """Print the last benchmark from results/report.json."""
    if not RESULTS.is_file():
        raise _fail("no results/report.json yet; run `evaluate` first")
    typer.echo(format_report(json.loads(RESULTS.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    app()
