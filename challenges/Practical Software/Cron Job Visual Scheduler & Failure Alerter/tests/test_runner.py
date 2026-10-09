import io
import os
import signal
import sys
import threading
import time

import pytest
from cronwatch.alerts import Alerter
from cronwatch.config import Config
from cronwatch.runner import TailBuffer, display_command, run_wrapped
from cronwatch.store import ERROR, FAILED, OK, TIMEOUT, Store

PY = sys.executable


class Recording:
    name = "recording"

    def __init__(self):
        self.alerts = []

    def send(self, alert):
        self.alerts.append(alert)


class Setup:
    def __init__(self, tmp_path, config=None):
        self.store = Store(tmp_path / "c.db")
        self.channel = Recording()
        self.alerter = Alerter([self.channel], self.store)
        self.config = config or Config()
        self.out, self.err, self.warnings = io.BytesIO(), io.BytesIO(), []

    def run(self, code, job="job", **kwargs):
        argv = [PY, "-c", code] if isinstance(code, str) else code
        kwargs.setdefault("stdout", self.out)
        kwargs.setdefault("stderr", self.err)
        return run_wrapped(
            argv,
            job,
            self.store,
            self.config,
            self.alerter,
            warn=self.warnings.append,
            **kwargs,
        )

    @property
    def runs(self):
        return self.store.runs()


@pytest.fixture
def setup(tmp_path):
    return Setup(tmp_path)


def test_success_is_recorded_passed_through_and_silent(setup):
    code = setup.run("import sys; print('hello'); print('oops', file=sys.stderr)")
    assert (
        code == 0
        and setup.out.getvalue() == b"hello\n"
        and setup.err.getvalue() == b"oops\n"
    )
    (run,) = setup.runs
    assert (run.job_id, run.status, run.exit_code) == ("job", OK, 0)
    assert run.started_at <= run.ended_at and run.duration_s >= 0 and run.host
    assert "hello" in run.output_tail and "oops" in run.output_tail
    assert setup.channel.alerts == [] and setup.warnings == []


def test_exit_status_is_passed_through_and_a_failure_alerts_once(setup):
    assert setup.run("import sys; sys.exit(3)") == 3
    assert setup.run("import sys; sys.exit(3)") == 3
    failed = [r for r in setup.runs if r.status == FAILED]
    assert len(failed) == 2 and failed[0].exit_code == 3
    assert [a.kind for a in setup.channel.alerts] == [
        "failed"
    ]  # the second failure is deduplicated
    alert = setup.channel.alerts[0]
    assert (
        alert.job_id == "job"
        and alert.exit_code == 3
        and alert.consecutive_failures == 1
    )


def test_recovery_after_failure_sends_one_recovered_alert(setup):
    setup.run("raise SystemExit(1)")
    setup.run("pass")
    setup.run("pass")
    assert [a.kind for a in setup.channel.alerts] == ["failed", "recovered"]


def test_alert_after_threshold_is_honoured(tmp_path):
    setup = Setup(tmp_path, Config(alert_after=2))
    setup.run("raise SystemExit(1)")
    assert setup.channel.alerts == []
    setup.run("raise SystemExit(1)")
    assert [a.kind for a in setup.channel.alerts] == ["failed"]


def test_ok_codes_treat_chosen_nonzero_statuses_as_success(setup):
    assert (
        setup.run("raise SystemExit(24)", ok_codes=frozenset({0, 24})) == 24
    )  # rsync "files vanished"
    assert setup.runs[0].status == OK and setup.channel.alerts == []


def test_output_tail_keeps_the_end_of_large_output(setup):
    setup.run("[print(f'line {i}' + 'x' * 80) for i in range(5000)]")
    tail = setup.runs[0].output_tail
    assert (
        0 < len(tail.encode()) <= 16 * 1024
        and "line 4999" in tail
        and "line 0x" not in tail
    )
    assert (
        setup.out.getvalue().count(b"\n") == 5000
    )  # the passthrough is complete even though the record is not


def test_a_missing_command_is_a_127_with_a_message_not_a_crash(setup):
    assert setup.run(["/definitely/not/here", "--x"]) == 127
    run = setup.runs[0]
    assert (
        run.status == ERROR
        and "cannot run" in run.output_tail
        and b"cannot run" in setup.err.getvalue()
    )
    assert [a.kind for a in setup.channel.alerts] == ["failed"]


def test_a_non_executable_file_is_a_126(setup, tmp_path):
    script = tmp_path / "plain.txt"
    script.write_text("not executable")
    assert setup.run([str(script)]) == 126


def test_timeout_kills_the_whole_process_group_and_reports_124(setup, tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    started = time.monotonic()
    assert setup.run(code, timeout=1) == 124
    assert time.monotonic() - started < 15
    run = setup.runs[0]
    assert (
        run.status == TIMEOUT and run.exit_code == 124 and "timeout" in run.output_tail
    )
    grandchild = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the grandchild survived the timeout")
    assert [a.kind for a in setup.channel.alerts] == [
        "failed"
    ] and setup.channel.alerts[0].status == TIMEOUT


def test_a_child_killed_by_a_signal_reports_128_plus_the_signal(setup):
    assert setup.run("import os, signal; os.kill(os.getpid(), signal.SIGKILL)") == 137
    assert setup.runs[0].status == FAILED


def test_sigterm_to_the_wrapper_is_forwarded_to_the_job(setup):
    timer = threading.Timer(0.7, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    started = time.monotonic()
    code = setup.run("import time; time.sleep(60)")
    timer.cancel()
    assert code == 143 and time.monotonic() - started < 15
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL or callable(
        signal.getsignal(signal.SIGTERM)
    )


def test_a_broken_database_never_changes_the_jobs_result_but_still_alerts_on_failure(
    tmp_path,
):
    class BrokenStore:
        def start_run(self, *_a):
            raise OSError("database is locked")

    channel, warnings = Recording(), []
    alerter = Alerter([channel])
    ok = run_wrapped(
        [PY, "-c", "pass"],
        "j",
        BrokenStore(),
        Config(),
        alerter,
        stdout=io.BytesIO(),
        stderr=io.BytesIO(),
        warn=warnings.append,
    )
    bad = run_wrapped(
        [PY, "-c", "raise SystemExit(5)"],
        "j",
        BrokenStore(),
        Config(),
        alerter,
        stdout=io.BytesIO(),
        stderr=io.BytesIO(),
        warn=warnings.append,
    )
    assert (ok, bad) == (0, 5)
    assert any("database is locked" in w for w in warnings)
    assert [a.kind for a in channel.alerts] == ["failed"] and channel.alerts[
        0
    ].exit_code == 5


def test_a_store_that_dies_at_the_end_is_a_warning_not_a_changed_exit_code(tmp_path):
    real = Store(tmp_path / "x.db")

    class FlakyStore:
        def __getattr__(self, name):
            return getattr(real, name)

        def finish_run(self, *_a):
            raise OSError("disk full")

    warnings = []
    code = run_wrapped(
        [PY, "-c", "raise SystemExit(2)"],
        "j",
        FlakyStore(),
        Config(),
        Alerter([]),
        stdout=io.BytesIO(),
        stderr=io.BytesIO(),
        warn=warnings.append,
    )
    assert code == 2 and any("disk full" in w for w in warnings)


def test_a_failing_channel_never_changes_the_exit_code(tmp_path):
    class Boom:
        name = "boom"

        def send(self, _alert):
            raise RuntimeError("smtp exploded")

    setup = Setup(tmp_path)
    setup.alerter = Alerter([Boom()], setup.store)
    assert setup.run("raise SystemExit(4)") == 4
    assert setup.store.alerts()[0].ok is False


def test_a_closed_output_pipe_does_not_stall_or_fail_the_job(setup):
    class Closed:
        def write(self, _data):
            raise BrokenPipeError

        def flush(self):
            pass

    assert setup.run("print('x' * 100000)", stdout=Closed()) == 0
    assert len(setup.runs[0].output_tail) > 1000  # still recorded


def test_display_command_unwraps_sh_c_and_quotes_other_argv():
    assert display_command(["sh", "-c", "a | b && c"]) == "a | b && c"
    assert (
        display_command(["/usr/bin/backup", "--dest", "my dir"])
        == "/usr/bin/backup --dest 'my dir'"
    )


def test_tail_buffer_keeps_the_end_and_survives_split_utf8():
    tail = TailBuffer(limit=10)
    for chunk in (b"abcdef", b"ghijkl", b"mnop"):
        tail.add(chunk)
    assert tail.text() == "ghijklmnop"
    wide = TailBuffer(limit=5)
    wide.add("héllo wörld".encode())
    assert (
        wide.text().endswith("rld")
        and "�" in wide.text()
        or wide.text().endswith("wörld"[-3:])
    )
