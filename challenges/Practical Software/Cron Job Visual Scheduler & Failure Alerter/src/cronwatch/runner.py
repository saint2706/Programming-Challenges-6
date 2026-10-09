"""``cronwatch run``: execute a command exactly as cron would, and record what happened.

The wrapper is transparent. The command's output still reaches cron's stdout/stderr (so ``MAILTO``
and ``>> log 2>&1`` keep working), its exit status is passed through, and *our* failures (an
unwritable database, a dead webhook) are warnings that can never change the job's result.
"""

import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import BinaryIO

from cronwatch.alerts import Alerter, alert_from_run, evaluate_run
from cronwatch.config import Config
from cronwatch.store import ERROR, FAILED, OK, TIMEOUT, Run, Store

TAIL_BYTES = 16 * 1024
KILL_GRACE = 5
READER_JOIN = 2
EXIT_TIMEOUT, EXIT_NOT_EXECUTABLE, EXIT_NOT_FOUND = (
    124,
    126,
    127,
)  # same codes as timeout(1) and the shell
PRUNE_EVERY = 50
_FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


class TailBuffer:
    """The last ``limit`` bytes written, safe to fill from two reader threads."""

    def __init__(self, limit: int = TAIL_BYTES) -> None:
        self._chunks: deque[bytes] = deque()
        self._size, self._limit = 0, limit
        self._lock = threading.Lock()

    def add(self, data: bytes) -> None:
        with self._lock:
            self._chunks.append(data)
            self._size += len(data)
            while self._size - len(self._chunks[0]) >= self._limit:
                self._size -= len(self._chunks.popleft())

    def text(self) -> str:
        with self._lock:
            joined = b"".join(self._chunks)[-self._limit :]
        return joined.decode("utf-8", errors="replace")


def display_command(argv: list[str]) -> str:
    """What to show for a job: the original line when wrapped via ``sh -c``, else the quoted argv."""
    if (
        len(argv) == 3
        and argv[0] in {"sh", "bash", "/bin/sh", "/bin/bash"}
        and argv[1] == "-c"
    ):
        return argv[2]
    return shlex.join(argv)


def _pump(source: BinaryIO, sink: BinaryIO | None, tail: TailBuffer) -> None:
    """Copy a child's pipe to ``sink`` as it arrives, remembering the tail. A closed sink never stalls the child."""
    while chunk := source.read1(4096):
        tail.add(chunk)
        if sink is not None:
            try:
                sink.write(chunk)
                sink.flush()
            except (BrokenPipeError, ValueError, OSError):
                sink = None


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _stop(proc: subprocess.Popen) -> None:
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(KILL_GRACE)
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGKILL)
        proc.wait()


def _exit_code(returncode: int) -> int:
    return 128 + -returncode if returncode < 0 else returncode


def run_wrapped(
    argv: list[str],
    job_id: str,
    store: Store | None,
    config: Config,
    alerter: Alerter,
    *,
    timeout: float | None = None,
    ok_codes: frozenset[int] = frozenset({0}),
    stdout: BinaryIO | None = None,
    stderr: BinaryIO | None = None,
    warn: Callable[[str], None] = lambda message: print(
        f"cronwatch: warning: {message}", file=sys.stderr
    ),
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    """Run ``argv``, record the run, maybe alert, and return the exit status cron should see."""
    out = stdout if stdout is not None else sys.stdout.buffer
    err = stderr if stderr is not None else sys.stderr.buffer
    command, host, started = display_command(argv), socket.gethostname(), clock()
    run_id = None
    if store is not None:
        try:
            run_id = store.start_run(job_id, command, host, started)
        except Exception as exc:  # noqa: BLE001
            warn(f"could not record the start of {job_id!r}: {exc}")

    tail = TailBuffer()
    status, returncode = OK, 0
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
    except OSError as exc:
        status = ERROR
        returncode = (
            EXIT_NOT_FOUND
            if isinstance(exc, FileNotFoundError)
            else EXIT_NOT_EXECUTABLE
        )
        message = f"cronwatch: cannot run {argv[0]!r}: {exc.strerror or exc}\n".encode()
        tail.add(message)
        err.write(message)
        err.flush()
    else:
        readers = [
            threading.Thread(target=_pump, args=(proc.stdout, out, tail), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, err, tail), daemon=True),
        ]
        for reader in readers:
            reader.start()
        previous = _install_forwarding(proc)
        timed_out = False
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop(proc)
        finally:
            _restore(previous)
        for reader in readers:
            reader.join(
                READER_JOIN
            )  # a daemonised grandchild holding the pipe must not hang us
        if timed_out:
            status, returncode = TIMEOUT, EXIT_TIMEOUT
            note = f"cronwatch: killed after {timeout:g}s timeout\n".encode()
            tail.add(note)
            err.write(note)
            err.flush()
        else:
            returncode = _exit_code(proc.returncode)
            status = OK if returncode in ok_codes else FAILED

    ended = clock()
    _record_and_alert(
        store,
        config,
        alerter,
        run_id,
        job_id,
        command,
        host,
        started,
        ended,
        returncode,
        status,
        tail.text(),
        warn,
    )
    return returncode


def _record_and_alert(
    store,
    config,
    alerter,
    run_id,
    job_id,
    command,
    host,
    started,
    ended,
    returncode,
    status,
    output,
    warn,
) -> None:
    run: Run | None = None
    if store is not None and run_id is not None:
        try:
            store.finish_run(run_id, ended, returncode, status, output)
            run = store.get_run(run_id)
        except Exception as exc:  # noqa: BLE001
            warn(f"could not record the result of {job_id!r}: {exc}")
    try:
        if run is not None:
            alert = evaluate_run(store, config, run, ended)
            if run_id and run_id % PRUNE_EVERY == 0:
                store.prune(ended - config.retention)
        elif status != OK:
            # No readable history (database down): still tell someone the job failed, without dedup state.
            fallback = Run(
                0,
                job_id,
                command,
                host,
                started,
                ended,
                (ended - started).total_seconds(),
                returncode,
                status,
                output,
            )
            alert = alert_from_run("failed", fallback, 1)
        else:
            alert = None
        if alert is not None:
            alerter.dispatch(alert, run_id)
    except Exception as exc:  # noqa: BLE001
        warn(f"alerting for {job_id!r} failed: {exc}")


def _install_forwarding(proc: subprocess.Popen) -> dict:
    """Pass SIGTERM/SIGINT/SIGHUP to the job's process group, so killing the wrapper kills the job."""
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous = {}
    for sig in _FORWARDED:
        previous[sig] = signal.signal(
            sig, lambda signum, _frame: _signal_group(proc, signum)
        )
    return previous


def _restore(previous: dict) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)
