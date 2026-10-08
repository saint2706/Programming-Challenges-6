"""Typer CLI: fetch the corpus, build and train the sorter, rank a day of held-out mail."""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated

import typer

from inbox_sorter import data, inbox, pipeline
from inbox_sorter.models import MODEL_NAMES

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

RESULTS = pipeline.RESULTS_DIR / "report.json"


def get_context():
    """Indirection so tests can swap in a small trained pipeline."""
    return inbox.load_context()


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


def _users(users: str | None) -> list[str]:
    return (
        [u.strip() for u in users.split(",") if u.strip()]
        if users
        else pipeline.DEFAULT_USERS
    )


@app.command()
def fetch() -> None:
    """Download the Enron emails.csv from Kaggle (token in ~/.kaggle/kaggle.json)."""
    data.fetch()
    typer.echo(f"data in {data.DATA_DIR}")


@app.command()
def prepare(
    users: Annotated[
        str | None, typer.Option(help="Comma-separated mailboxes.")
    ] = None,
) -> None:
    """Parse, label and featurize the mailboxes into data/dataset.parquet."""
    df, stats = pipeline.prepare(
        data.DATA_DIR / data.CSV_NAME, _users(users), data.DATA_DIR
    )
    for box, s in stats.items():
        typer.echo(
            f"{box:<14} kept {s['kept']:>6}  acted {s['acted_rate']:.1%}  "
            f"(reply {s['reply_rate']:.1%}, forward {s['forward_rate']:.1%})  "
            f"censored {s['censored']}"
        )
    typer.echo(
        f"{df.height} messages written to {data.DATA_DIR / pipeline.DATASET_FILE}"
    )


@app.command()
def train(
    users: Annotated[
        str | None, typer.Option(help="Comma-separated mailboxes.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
) -> None:
    """Prepare, fit, calibrate and evaluate everything; writes results/report.json."""
    report = pipeline.run_all(
        data.DATA_DIR, pipeline.RESULTS_DIR, _users(users), seed=seed
    )
    typer.echo(format_report(report))


def _interval(metric: dict) -> str:
    return f"{metric['mean']:.3f} [{metric['lo']:.3f}, {metric['hi']:.3f}]"


def format_report(report: dict) -> str:
    daily_days = report["models"]["random"]["daily"]["n_days"]
    lines = [
        f"test split: {report['data']['splits']['test']} messages, {daily_days} eligible inbox days",
        "",
        f"{'model':<16}{'PR-AUC':<9}{'ROC-AUC':<9}{'P@3 [95% CI]':<24}{'nDCG@5 [95% CI]':<24}recall@top20%",
    ]
    for name in MODEL_NAMES:
        m = report["models"][name]
        d = m["daily"]
        lines.append(
            f"{name:<16}{m['overall']['pr_auc']:<9.3f}{m['overall']['roc_auc']:<9.3f}"
            f"{_interval(d['precision_at_3']):<24}{_interval(d['ndcg_at_5']):<24}"
            f"{_interval(d['recall_top20'])}"
        )
    return "\n".join(lines)


@app.command()
def report() -> None:
    """Print the last benchmark from results/report.json."""
    if not RESULTS.is_file():
        raise _fail("no results/report.json yet; run `train` first")
    typer.echo(format_report(json.loads(RESULTS.read_text(encoding="utf-8"))))


@app.command("rank-inbox")
def rank_inbox(
    mailbox: Annotated[str, typer.Argument(help="Mailbox id, e.g. mann-k.")],
    day: Annotated[str, typer.Option(help="Local day to rank, YYYY-MM-DD.")],
    top: Annotated[int, typer.Option(min=1, help="How many messages to list.")] = 10,
    reveal: Annotated[
        bool, typer.Option("--reveal", help="Also show what the owner actually did.")
    ] = False,
) -> None:
    """Rank one day of held-out mail by how likely the owner is to answer it."""
    try:
        when = date.fromisoformat(day)
    except ValueError as exc:
        raise _fail(f"--day must be YYYY-MM-DD, got {day!r}") from exc
    art, df = get_context()
    try:
        items = inbox.rank_inbox(art, df, mailbox, when, top=top)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("no mail") and mailbox in inbox.mailboxes(df):
            days = sorted(inbox.days_with_mail(df, mailbox))
            message += f"; test days run {days[0]} to {days[-1]}"
        raise _fail(message) from exc
    for item in items:
        outcome = ""
        if reveal:
            outcome = f"  [{'replied/forwarded' if item.acted else 'ignored'}]"
        typer.echo(
            f"{item.rank:>2}. {item.probability:5.1%}  {item.sender} | {item.subject}{outcome}"
        )
        because = ", ".join(f"{label} {w:+.2f}" for label, w in item.reasons)
        typer.echo(f"      because: {because}")


if __name__ == "__main__":
    app()
