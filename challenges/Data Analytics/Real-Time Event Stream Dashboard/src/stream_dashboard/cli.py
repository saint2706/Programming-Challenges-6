"""Command line: run the producer and workers, start the dashboard, check the result."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import polars as pl
import redis
import typer
from rich.console import Console
from rich.table import Table

from stream_dashboard.broker import Broker, connect
from stream_dashboard.events import Disorder, arrival_order, load_events
from stream_dashboard.oracle import verify
from stream_dashboard.producer import ProducerConfig, produce
from stream_dashboard.windows import (
    DEFAULT_LATENESS_MS,
    DEFAULT_SPECS,
    Layout,
    WindowSpec,
    parse_duration,
)
from stream_dashboard.worker import Worker, WorkerConfig

app = typer.Typer(
    help="Replay a real clickstream through Redis Streams into a live dashboard.",
    no_args_is_help=True,
)
console = Console()

UrlOpt = Annotated[str, typer.Option("--url", envvar="REDIS_URL", help="Redis server.")]
PrefixOpt = Annotated[
    str, typer.Option(help="Key prefix; runs with different prefixes do not collide.")
]
SpecsOpt = Annotated[
    str,
    typer.Option(help="Window specs: `1m` tumbling, `5m/1m` sliding. Comma separated."),
]
LatenessOpt = Annotated[str, typer.Option(help="Allowed lateness, e.g. 30s.")]


def _specs(text: str) -> tuple[WindowSpec, ...]:
    try:
        return tuple(WindowSpec.parse(t) for t in text.split(",") if t.strip())
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc


def _lateness(text: str) -> int:
    try:
        return parse_duration(text)
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc


def _broker(url: str, prefix: str) -> Broker:
    client = connect(url)
    try:
        client.ping()
    except redis.RedisError as exc:
        console.print(f"[red]error:[/red] cannot reach Redis at {url}: {exc}")
        raise typer.Exit(2) from exc
    return Broker(client, Layout(prefix=prefix))


def _frame(limit: int | None, data: Path | None, disorder: Disorder) -> pl.DataFrame:
    try:
        return arrival_order(load_events(data, limit), disorder)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2) from exc


DEFAULT_SPEC_TEXT = ",".join(w.name.replace("-", "/")[1:] for w in DEFAULT_SPECS)


@app.command("produce")
def produce_cmd(
    url: UrlOpt = "redis://localhost:6379/0",
    prefix: PrefixOpt = "sd:",
    speed: Annotated[
        float,
        typer.Option(help="Event-seconds per wall-second; 0 = as fast as possible."),
    ] = 60.0,
    limit: Annotated[int | None, typer.Option(help="Only the first N events.")] = None,
    data: Annotated[
        Path | None,
        typer.Option(help="Events CSV (default: data/ or the committed sample)."),
    ] = None,
    backpressure: Annotated[str, typer.Option(help="block or none.")] = "block",
    high_water: int = 5_000,
    low_water: int = 1_000,
    disorder: Annotated[
        float, typer.Option(help="Fraction of events delayed in transit (injected).")
    ] = 0.05,
    max_delay: Annotated[str, typer.Option(help="Longest delay of those.")] = "20s",
    stragglers: Annotated[
        float, typer.Option(help="Fraction delayed past max-delay, up to 5 minutes.")
    ] = 0.002,
    reset: Annotated[
        bool, typer.Option(help="Delete this prefix's keys first.")
    ] = True,
) -> None:
    """Replay the events into the stream."""
    broker = _broker(url, prefix)
    if reset:
        broker.reset()
    d = Disorder(disorder, _lateness(max_delay), stragglers)
    frame = _frame(limit, data, d)
    cfg = ProducerConfig(speed, 500, backpressure, high_water, low_water)
    console.print(
        f"producing {frame.height:,} events at {speed:g}x, backpressure={backpressure}"
    )
    res = produce(broker, frame, cfg)
    console.print(
        f"sent {res.sent:,} in {res.wall_s:.1f}s (blocked {res.blocked_ms / 1000:.1f}s)"
    )


@app.command()
def worker(
    url: UrlOpt = "redis://localhost:6379/0",
    prefix: PrefixOpt = "sd:",
    name: Annotated[str, typer.Option(help="Consumer name; unique per worker.")] = "w1",
    specs: SpecsOpt = DEFAULT_SPEC_TEXT,
    lateness: LatenessOpt = "30s",
    batch: int = 200,
    claim_idle: Annotated[
        str, typer.Option(help="Reclaim others' entries idle this long.")
    ] = "2s",
    process_delay_ms: Annotated[
        float, typer.Option(help="Artificial cost per entry (slow consumer).")
    ] = 0.0,
    crash_after: Annotated[
        int | None, typer.Option(help="Die uncleanly after reading N entries.")
    ] = None,
    drain: Annotated[
        bool, typer.Option(help="Exit when the producer is done and nothing is left.")
    ] = True,
) -> None:
    """Consume the stream and maintain the windows."""
    broker = _broker(url, prefix)
    cfg = WorkerConfig(
        name=name,
        specs=_specs(specs),
        lateness_ms=_lateness(lateness),
        batch=batch,
        claim_idle_ms=_lateness(claim_idle),
        process_delay_ms=process_delay_ms,
        crash_after=crash_after,
        drain=drain,
    )
    stats = Worker(broker, cfg).run()
    console.print(
        f"{name}: read {stats.read:,}, applied {stats.applied:,}, skipped duplicates {stats.duplicates:,}, reclaimed {stats.reclaimed:,}"
    )


@app.command()
def demo(
    url: UrlOpt = "redis://localhost:6379/0",
    prefix: PrefixOpt = "sd:",
    speed: float = 600.0,
    limit: Annotated[int | None, typer.Option(help="Only the first N events.")] = None,
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 2,
    crash: Annotated[
        bool, typer.Option(help="Kill worker w2 mid-run; w1 must recover its entries.")
    ] = False,
    slow_ms: Annotated[
        float, typer.Option(help="Artificial per-entry cost in each worker.")
    ] = 0.0,
    backpressure: str = "block",
    lateness: LatenessOpt = "30s",
    disorder: float = 0.05,
    stragglers: float = 0.002,
    check: Annotated[
        bool, typer.Option(help="Compare the result with the offline oracle.")
    ] = True,
    data: Annotated[
        Path | None, typer.Option(help="Events CSV (default: data/ or the sample).")
    ] = None,
) -> None:
    """Run producer + workers as separate processes, then verify. Open the dashboard meanwhile."""
    broker = _broker(url, prefix)
    broker.reset()
    broker.ensure_group()
    common = ["--url", url, "--prefix", prefix]
    procs: list[subprocess.Popen] = []
    for i in range(1, workers + 1):
        cmd = [
            sys.executable,
            "-m",
            "stream_dashboard",
            "worker",
            *common,
            "--name",
            f"w{i}",
            "--lateness",
            lateness,
            "--process-delay-ms",
            str(slow_ms),
        ]
        if crash and i == 2:
            cmd += ["--crash-after", "400"]
        procs.append(subprocess.Popen(cmd, env={**os.environ, "NO_COLOR": "1"}))
    d = Disorder(disorder, 20_000, stragglers)
    frame = _frame(limit, data, d)
    console.print(
        f"{frame.height:,} events, {workers} workers{' (w2 will crash)' if crash else ''}; dashboard: uv run streamlit run src/stream_dashboard/app.py"
    )
    cfg = ProducerConfig(speed, 500, backpressure)
    res = produce(broker, frame, cfg)
    console.print(
        f"producer done: {res.sent:,} sent in {res.wall_s:.1f}s, blocked {res.blocked_ms / 1000:.1f}s"
    )
    for p in procs:
        p.wait()
    if check:
        _print_verify(broker, frame, DEFAULT_SPECS)


def _print_verify(
    broker: Broker, frame: pl.DataFrame, specs: tuple[WindowSpec, ...]
) -> bool:
    rep = verify(broker.r, broker.layout, frame, specs)
    t = Table(
        title=f"{rep.applied:,} of {rep.total_events:,} events applied exactly once"
    )
    for c in (
        "spec",
        "windows",
        "exact",
        "short (late drops)",
        "over",
        "wrong",
        "late",
    ):
        t.add_column(c, justify="right" if c != "spec" else "left")
    for s in rep.specs:
        t.add_row(
            s.spec,
            f"{s.windows_found}/{s.windows_expected}",
            str(s.exact),
            str(s.short),
            str(s.over),
            str(s.wrong_other),
            str(s.late),
        )
    console.print(t)
    for s in rep.specs:
        for p in s.problems[:5]:
            console.print(f"[red]{s.spec}: {p}[/red]")
    console.print("[green]verified[/green]" if rep.ok else "[red]MISMATCH[/red]")
    return rep.ok


@app.command("verify")
def verify_cmd(
    url: UrlOpt = "redis://localhost:6379/0",
    prefix: PrefixOpt = "sd:",
    specs: SpecsOpt = DEFAULT_SPEC_TEXT,
    limit: Annotated[
        int | None, typer.Option(help="Same --limit the producer used.")
    ] = None,
    data: Annotated[
        Path | None, typer.Option(help="The same events CSV the producer used.")
    ] = None,
) -> None:
    """Compare what is in Redis with the offline polars answer."""
    broker = _broker(url, prefix)
    frame = _frame(limit, data, Disorder())
    if not _print_verify(broker, frame, _specs(specs)):
        raise typer.Exit(1)


@app.command()
def status(url: UrlOpt = "redis://localhost:6379/0", prefix: PrefixOpt = "sd:") -> None:
    """Print backlog, watermark and late counts once."""
    from stream_dashboard.snapshot import health

    broker = _broker(url, prefix)
    h = health(broker.r, broker.layout, DEFAULT_LATENESS_MS)
    console.print(h)


@app.command()
def reset(url: UrlOpt = "redis://localhost:6379/0", prefix: PrefixOpt = "sd:") -> None:
    """Delete every key under the prefix."""
    n = _broker(url, prefix).reset()
    console.print(f"deleted {n:,} keys")


@app.command()
def dashboard(port: int = 8501) -> None:
    """Start the Streamlit dashboard."""
    app_path = Path(__file__).with_name("app.py")
    raise typer.Exit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(port),
            ]
        )
    )


@app.command()
def sweep(
    data: Annotated[
        Path | None, typer.Option(help="Events CSV (default: data/ or the sample).")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N events.")] = None,
    disorder: float = 0.05,
    max_delay: str = "20s",
    stragglers: float = 0.002,
) -> None:
    """How many events each allowed lateness drops as late (offline, no Redis needed)."""
    from stream_dashboard.engine import simulate
    from stream_dashboard.events import to_event

    frame = _frame(limit, data, Disorder(disorder, _lateness(max_delay), stragglers))
    events = [to_event(r) for r in frame.iter_rows(named=True)]
    specs = DEFAULT_SPECS
    t = Table(
        title=f"{len(events):,} events, {disorder:.0%} delayed up to {max_delay}, {stragglers:.1%} up to 5m"
    )
    t.add_column("allowed lateness", justify="right")
    for s in specs:
        t.add_column(f"dropped {s.name}", justify="right")
    for lateness in ("0ms", "10s", "20s", "30s", "1m", "2m", "5m"):
        late = simulate(
            events,
            specs,
            0 if lateness == "0ms" else parse_duration(lateness),
            Layout(),
        ).hashes[Layout().late]
        t.add_row(
            lateness,
            *[
                f"{late.get(s.name, 0):,} ({late.get(s.name, 0) / len(events):.2%})"
                for s in specs
            ],
        )
    console.print(t)
