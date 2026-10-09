import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from cronwatch.cli import app
from cronwatch.store import Store
from helpers import WebhookServer
from typer.testing import CliRunner

runner = CliRunner()
PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    crontab = tmp_path / "crontab"
    crontab.write_text(
        "# nightly dump\n0 2 * * * /usr/bin/pg_backup\n*/15 * * * * /x/cronwatch run --job pinger -- sh -c 'echo pong'\n"
    )
    return {"CRONWATCH_HOME": str(home), "CRONWATCH_CRONTAB": str(crontab)}


def run(env, *args):
    return runner.invoke(app, list(args), env={**env, "COLUMNS": "200"})


def subprocess_run(env, *args, **kwargs):
    """The real thing: a new interpreter, exactly what cron would start."""
    return subprocess.run(
        [sys.executable, "-m", "cronwatch", *args],
        capture_output=True,
        text=True,
        cwd=PROJECT,
        env={**os.environ, **env},
        timeout=60,
        check=False,
        **kwargs,
    )


def test_wrapper_in_a_real_process_passes_output_and_exit_status_through(env):
    result = subprocess_run(
        env,
        "run",
        "--job",
        "t",
        "--",
        "sh",
        "-c",
        "echo to-out; echo to-err >&2; exit 7",
    )
    assert (
        result.returncode == 7
        and result.stdout == "to-out\n"
        and result.stderr == "to-err\n"
    )
    (recorded,) = Store(Path(env["CRONWATCH_HOME"]) / "cronwatch.db").runs()
    assert (recorded.job_id, recorded.exit_code, recorded.status) == ("t", 7, "failed")


def test_options_after_the_command_belong_to_the_command_not_to_cronwatch(env):
    result = subprocess_run(
        env, "run", "--job", "t", "sh", "-c", "echo $0 $1", "--job", "inner"
    )
    assert result.returncode == 0 and "--job inner" in result.stdout


def test_wrapper_timeout_and_ok_codes_options(env):
    slow = subprocess_run(
        env, "run", "--job", "s", "--timeout", "1", "--", "sleep", "30"
    )
    assert slow.returncode == 124 and "timeout" in slow.stderr
    tolerated = subprocess_run(
        env, "run", "--job", "r", "--ok-codes", "0,24", "--", "sh", "-c", "exit 24"
    )
    assert tolerated.returncode == 24
    statuses = {
        r.job_id: r.status
        for r in Store(Path(env["CRONWATCH_HOME"]) / "cronwatch.db").runs()
    }
    assert statuses == {"s": "timeout", "r": "ok"}


def test_wrapper_ignores_a_broken_config_but_still_runs_the_job(env):
    (Path(env["CRONWATCH_HOME"]) / "config.toml").write_text(
        "[alerts]\nalert_after = 0\n"
    )
    result = subprocess_run(
        env, "run", "--job", "t", "--", "sh", "-c", "echo still-ran"
    )
    assert (
        result.returncode == 0
        and "still-ran" in result.stdout
        and "ignoring config" in result.stderr
    )


def test_wrapper_with_an_unusable_history_database_still_runs_the_job(env, tmp_path):
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    result = subprocess_run(
        {**env, "CRONWATCH_HOME": str(blocker / "sub")},
        "run",
        "--job",
        "t",
        "--",
        "sh",
        "-c",
        "echo ran; exit 3",
    )
    assert result.returncode == 3 and "ran" in result.stdout


def test_bad_ok_codes_is_a_usage_error_not_a_crash(env):
    result = run(env, "run", "--job", "t", "--ok-codes", "zero", "--", "true")
    assert result.exit_code == 2 and "integers" in result.output


def test_failure_alerts_through_a_real_webhook_exactly_once(env):
    with WebhookServer() as server:
        (Path(env["CRONWATCH_HOME"]) / "config.toml").write_text(
            f'[alerts]\n[alerts.webhook]\nurl = "{server.url}"\n'
        )
        for _ in range(3):
            subprocess_run(env, "run", "--job", "flaky", "--", "sh", "-c", "exit 1")
        recovered = subprocess_run(env, "run", "--job", "flaky", "--", "true")
    assert recovered.returncode == 0
    kinds = [json.loads(r["body"])["kind"] for r in server.requests]
    assert kinds == [
        "failed",
        "recovered",
    ]  # three failures produced one alert; the fix produced one more


def test_list_shows_meaning_next_run_and_monitoring_state(env):
    result = run(env, "list")
    assert result.exit_code == 0
    for fragment in [
        "nightly dump",
        "At 02:00",
        "pinger",
        "Every 15 minutes",
        "not monitored",
        "no runs yet",
    ]:
        assert fragment in result.output


def test_list_on_an_empty_crontab(tmp_path):
    result = run(
        {"CRONWATCH_HOME": str(tmp_path), "CRONWATCH_CRONTAB": str(tmp_path / "none")},
        "list",
    )
    assert "No jobs" in result.output


def test_history_table_and_single_run_output(env):
    subprocess_run(
        env, "run", "--job", "t", "--", "sh", "-c", "echo captured-text; exit 2"
    )
    table = run(env, "history")
    assert (
        "t" in table.output
        and "failed" in table.output
        and "│ 2" in table.output.replace("┃", "│")
    )
    detail = run(env, "history", "--run", "1")
    assert "captured-text" in detail.output and "exit 2" in detail.output
    assert run(env, "history", "--run", "999").exit_code == 1
    assert "pinger" not in run(env, "history", "--job", "nothing").output


def test_timeline_command_draws_rows(env):
    result = run(env, "timeline", "--width", "60")
    assert (
        result.exit_code == 0
        and "pinger" in result.output
        and "nightly dump" in result.output
        and "now" in result.output
    )


def test_check_reports_missed_runs_and_exit_status(env):
    first = run(env, "check", "--dry-run")
    assert (
        first.exit_code == 0 and "No missed runs" in first.output
    )  # first sighting starts the clock


def test_test_alert_without_channels_is_an_error_with_guidance(env):
    result = run(env, "test-alert")
    assert result.exit_code == 1 and "config.toml" in result.output


def test_test_alert_reports_each_channel(env):
    with WebhookServer(statuses=[200]) as server:
        (Path(env["CRONWATCH_HOME"]) / "config.toml").write_text(
            f'[alerts]\n[alerts.webhook]\nurl = "{server.url}"\n'
        )
        result = run(env, "test-alert")
    assert result.exit_code == 0 and "webhook: sent" in result.output
    assert json.loads(server.requests[0]["body"])["kind"] == "test"
    (Path(env["CRONWATCH_HOME"]) / "config.toml").write_text(
        '[alerts]\n[alerts.webhook]\nurl = "http://127.0.0.1:9/nothing"\n'
    )
    failed = run(env, "test-alert")
    assert failed.exit_code == 1 and "FAILED" in failed.output


def test_install_monitor_adds_one_entry_and_the_home_variable_once(env):
    first = run(env, "install-monitor", "--every", "10")
    text = Path(env["CRONWATCH_CRONTAB"]).read_text()
    assert "Installed" in first.output and "*/10 * * * *" in text and " check" in text
    assert f"CRONWATCH_HOME={env['CRONWATCH_HOME']}" in text
    again = run(env, "install-monitor")
    assert (
        "already installed" in again.output
        and Path(env["CRONWATCH_CRONTAB"]).read_text() == text
    )


def test_a_broken_config_is_a_clear_error_for_interactive_commands(env):
    (Path(env["CRONWATCH_HOME"]) / "config.toml").write_text(
        "[alerts]\nalert_after = 0\n"
    )
    result = run(env, "list")
    assert result.exit_code == 2 and "alert_after" in result.output
