import re
import subprocess
import sys

from conftest import ROOT, SAMPLE
from stream_dashboard.cli import app
from stream_dashboard.producer import ProducerConfig, produce
from stream_dashboard.worker import Worker, WorkerConfig
from typer.testing import CliRunner

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def clean(text: str) -> str:
    """Strip colour codes and Rich box drawing so assertions survive GitHub Actions' styling."""
    text = ANSI.sub("", text)
    text = re.sub(r"[─-╿]", " ", text)
    return re.sub(r"\s+", " ", text)


def run(*args):
    result = CliRunner().invoke(app, [str(a) for a in args])
    return result.exit_code, clean(result.output)


def test_unreachable_redis_is_a_clear_error():
    code, out = run("status", "--url", "redis://127.0.0.1:1/0")
    assert code == 2 and "cannot reach Redis" in out


def test_bad_window_spec_is_a_clear_error(redis_url, layout):
    code, out = run(
        "worker", "--url", redis_url, "--prefix", layout.prefix, "--specs", "1m/5m"
    )
    assert code == 2 and "hop" in out or "gaps" in out


def test_bad_lateness_is_a_clear_error(redis_url, layout):
    code, out = run(
        "worker", "--url", redis_url, "--prefix", layout.prefix, "--lateness", "soon"
    )
    assert code == 2 and "bad duration" in out


def test_produce_then_worker_then_verify_via_the_cli(redis_url, layout, broker):
    base = ["--url", redis_url, "--prefix", layout.prefix]
    code, out = run(
        "produce",
        *base,
        "--speed",
        0,
        "--limit",
        3000,
        "--backpressure",
        "none",
        "--data",
        SAMPLE,
        "--stragglers",
        0,
        "--disorder",
        0.05,
    )
    assert code == 0 and "sent 3,000" in out
    code, out = run("worker", *base, "--lateness", "10m", "--batch", 500)
    assert code == 0 and "applied 3,000" in out
    code, out = run("verify", *base, "--limit", 3000)
    assert code == 0 and "verified" in out and "3,000 of 3,000" in out
    code, out = run("status", *base)
    assert code == 0 and "producer_done=True" in out
    code, out = run("reset", *base)
    assert code == 0 and "deleted" in out


def test_verify_exits_nonzero_on_a_mismatch(redis_url, layout, broker):
    base = ["--url", redis_url, "--prefix", layout.prefix]
    assert (
        run(
            "produce",
            *base,
            "--speed",
            0,
            "--limit",
            1500,
            "--backpressure",
            "none",
            "--disorder",
            0,
            "--stragglers",
            0,
        )[0]
        == 0
    )
    assert run("worker", *base, "--lateness", "10m")[0] == 0
    broker.r.spop(layout.done)
    code, out = run("verify", *base, "--limit", 1500)
    assert code == 1 and "MISMATCH" in out


def test_demo_runs_separate_processes_survives_a_killed_worker_and_verifies(
    redis_url, layout, broker
):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "stream_dashboard",
            "demo",
            "--url",
            redis_url,
            "--prefix",
            layout.prefix,
            "--limit",
            "6000",
            "--speed",
            "0",
            "--crash",
            "--lateness",
            "10m",
            "--disorder",
            "0.05",
            "--stragglers",
            "0",
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        env=_env(),
    )
    out = clean(proc.stdout)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "6,000 of 6,000 events applied exactly once" in out and "verified" in out
    assert "w1: read" in out or "w2: read" in out


def _env():
    import os

    return {**os.environ, "NO_COLOR": "1", "COLUMNS": "200"}


def test_streamlit_app_renders_a_populated_pipeline(
    redis_url, layout, broker, events, monkeypatch
):
    from streamlit.testing.v1 import AppTest

    head = events.head(4000)
    produce(
        broker,
        head.with_columns(arrival_ms=head["ts_ms"]),
        ProducerConfig(speed=0, backpressure="none"),
    )
    Worker(broker, WorkerConfig(name="w1", lateness_ms=60_000, block_ms=50)).run()
    monkeypatch.setenv("REDIS_URL", redis_url)
    at = AppTest.from_file(
        str(ROOT / "src" / "stream_dashboard" / "app.py"), default_timeout=60
    )
    at.run()
    assert not at.exception
    at.sidebar.text_input[1].set_value(layout.prefix)
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels == [
        "Produced",
        "Events/s (last 5 s)",
        "Backlog",
        "Latency p95",
        "Watermark (UTC)",
        "Late (dropped)",
    ]
    produced = next(m for m in at.metric if m.label == "Produced")
    assert produced.value == "4,000" and produced.delta.startswith("of 4,000")
    assert [t.label for t in at.tabs] == [
        "Windows",
        "Top products and brands",
        "Pipeline health",
    ]


def test_streamlit_app_explains_an_empty_stream_and_a_bad_spec(
    redis_url, layout, monkeypatch
):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("REDIS_URL", redis_url)
    at = AppTest.from_file(
        str(ROOT / "src" / "stream_dashboard" / "app.py"), default_timeout=60
    )
    at.run()
    at.sidebar.text_input[1].set_value(layout.prefix)
    at.run()
    assert not at.exception and any("stream is empty" in i.value for i in at.info)
    at.sidebar.text_input[2].set_value("1m/5m")
    at.run()
    assert at.error and "hop" in at.error[0].value
