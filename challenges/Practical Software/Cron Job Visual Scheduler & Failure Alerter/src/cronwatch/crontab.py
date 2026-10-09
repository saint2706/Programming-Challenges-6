"""Read, edit and safely write a user crontab without disturbing anything we do not understand.

Every line is kept verbatim in ``Line.raw``; only the line being edited is regenerated. Comments,
blank lines, ``MAILTO=`` assignments, and even malformed lines survive a load/save cycle untouched,
which is what makes it safe to point this at a crontab you have been curating for years.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol

from cronwatch.cronexpr import CronExpr, CronSyntaxError, parse

OFF_MARKER = (
    "#cronwatch:off#"  # `#cronwatch:off# 0 3 * * * cmd` is a disabled job, reversibly
)
DESCRIPTION_PREFIX = "# cronwatch:"
BACKUPS_KEPT = 50

_ENV = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=")
_MACRO_JOB = re.compile(r"^(\s*)(@\w+)\s+(\S.*)$")
_FIELD_JOB = re.compile(r"^(\s*)(\S+\s+\S+\s+\S+\s+\S+\s+\S+)\s+(\S.*)$")
_WRAPPER = re.compile(r"\bcronwatch['\"]?\s+run\b(?P<options>.*?)(?:\s--(?:\s|$)|$)")
_JOB_OPTION = re.compile(r"(?:--job|-j)[ =]([A-Za-z0-9_.-]+)")
_UNESCAPED_PERCENT = re.compile(r"(?<!\\)%")


class CrontabError(Exception):
    pass


class ConflictError(CrontabError):
    """The crontab changed on disk after it was loaded, so saving would overwrite someone's edit."""


@dataclass
class Line:
    raw: str
    kind: str  # blank | comment | env | job | other
    schedule: str = ""
    command: str = ""
    enabled: bool = True
    expr: CronExpr | None = None
    error: str | None = None  # a job-shaped line whose schedule is invalid


@dataclass(frozen=True)
class Job:
    index: int  # position in the crontab's lines
    schedule: str
    command: str
    enabled: bool
    expr: CronExpr | None
    error: str | None
    description: str | None
    job_id: (
        str | None
    )  # set when the command already runs under `cronwatch run --job ID`

    @property
    def label(self) -> str:
        if self.job_id:
            return self.job_id
        if self.description:
            return self.description
        words = self.command.split()
        return os.path.basename(words[0]) if words else "(empty)"

    @property
    def monitored(self) -> bool:
        return self.job_id is not None


def _parse_job_text(text: str) -> Line | None:
    match = _MACRO_JOB.match(text) or _FIELD_JOB.match(text)
    if not match:
        return None
    schedule, command = match.group(2), match.group(3).rstrip()
    try:
        return Line(text, "job", schedule, command, True, parse(schedule))
    except CronSyntaxError as exc:
        return Line(text, "job", schedule, command, True, None, str(exc))


def parse_line(raw: str) -> Line:
    stripped = raw.strip()
    if not stripped:
        return Line(raw, "blank")
    if stripped.startswith(OFF_MARKER):
        job = _parse_job_text(stripped[len(OFF_MARKER) :].lstrip())
        if job is not None:
            return replace(job, raw=raw, enabled=False)
        return Line(raw, "comment")
    if stripped.startswith("#"):
        return Line(raw, "comment")
    if _ENV.match(raw):
        return Line(raw, "env")
    job = _parse_job_text(raw)
    return replace(job, raw=raw) if job else Line(raw, "other")


def wrapper_job_id(command: str) -> str | None:
    """The ``--job`` of a command already running under ``cronwatch run``."""
    match = _WRAPPER.search(command)
    if not match:
        return None
    option = _JOB_OPTION.search(match.group("options"))
    return option.group(1) if option else None


def wrapper_timeout(command: str) -> int | None:
    """The ``--timeout`` of a wrapped command, so editing a job does not silently drop it."""
    match = _WRAPPER.search(command)
    found = re.search(r"--timeout[ =](\d+)", match.group("options")) if match else None
    return int(found.group(1)) if found else None


def has_unescaped_percent(command: str) -> bool:
    """Cron turns an unescaped ``%`` into a newline (and feeds the rest to stdin); wrapping would change that."""
    return _UNESCAPED_PERCENT.search(command) is not None


def default_wrapper() -> str:
    """How cron should invoke us: absolute, because cron's PATH is minimal."""
    found = shutil.which("cronwatch")
    return (
        shlex.quote(found) if found else f"{shlex.quote(sys.executable)} -m cronwatch"
    )


def wrap_command(
    command: str, job_id: str, wrapper: str, timeout: int | None = None
) -> str:
    """``cmd`` -> ``cronwatch run --job ID -- sh -c 'cmd'``, so pipes, ``&&`` and redirects stay inside the job."""
    if has_unescaped_percent(command):
        raise CrontabError(
            "this command uses an unescaped % (cron treats it as a newline); escape it as \\% first"
        )
    parts = [wrapper, "run", "--job", job_id]
    if timeout:
        parts += ["--timeout", str(timeout)]
    return " ".join([*parts, "--", "sh", "-c", shlex.quote(command)])


def unwrap_command(command: str) -> str:
    marker = " -- sh -c "
    _, found, tail = command.partition(marker)
    if not found or wrapper_job_id(command) is None:
        raise CrontabError(
            "this job was not wrapped by cronwatch, or was edited by hand; edit it directly"
        )
    try:
        (original,) = shlex.split(tail)
    except ValueError as exc:
        raise CrontabError(
            "cannot unwrap: the wrapped command has unexpected trailing text"
        ) from exc
    return original


def suggest_job_id(command: str, taken: set[str]) -> str:
    words = command.split()
    base = (
        re.sub(
            r"[^A-Za-z0-9_.-]+",
            "-",
            os.path.splitext(os.path.basename(words[0]))[0] if words else "job",
        ).strip("-")
        or "job"
    )
    candidate, n = base, 1
    while candidate in taken:
        n += 1
        candidate = f"{base}-{n}"
    return candidate


class Crontab:
    def __init__(self, lines: list[Line]) -> None:
        self.lines = lines

    @classmethod
    def from_text(cls, text: str) -> "Crontab":
        body = text.removesuffix("\n")
        return cls([parse_line(raw) for raw in body.split("\n")] if text else [])

    def to_text(self) -> str:
        return "".join(line.raw + "\n" for line in self.lines)

    def copy(self) -> "Crontab":
        return Crontab([replace(line) for line in self.lines])

    def environment(self) -> dict[str, str]:
        env = {}
        for line in self.lines:
            if line.kind == "env":
                name, _, value = line.raw.partition("=")
                env[name.strip()] = value.strip().strip("\"'")
        return env

    def _description(self, index: int) -> str | None:
        if (
            index == 0
            or self.lines[index - 1].kind != "comment"
            or self.lines[index - 1].raw.strip().startswith(OFF_MARKER)
        ):
            return None
        text = self.lines[index - 1].raw.strip()
        return text.removeprefix(DESCRIPTION_PREFIX).lstrip("# ").strip() or None

    def jobs(self) -> list[Job]:
        return [
            Job(
                i,
                line.schedule,
                line.command,
                line.enabled,
                line.expr,
                line.error,
                self._description(i),
                wrapper_job_id(line.command),
            )
            for i, line in enumerate(self.lines)
            if line.kind == "job"
        ]

    def job(self, index: int) -> Job:
        for job in self.jobs():
            if job.index == index:
                return job
        raise CrontabError(f"no job at line {index + 1}")

    def _line(self, index: int) -> Line:
        if not 0 <= index < len(self.lines) or self.lines[index].kind != "job":
            raise CrontabError(f"no job at line {index + 1}")
        return self.lines[index]

    @staticmethod
    def _validate(schedule: str, command: str) -> CronExpr:
        if not command.strip():
            raise CrontabError("the command is empty")
        if "\n" in command or "\n" in schedule:
            raise CrontabError("a crontab entry must be a single line")
        try:
            return parse(schedule)
        except CronSyntaxError as exc:
            raise CrontabError(str(exc)) from exc

    @staticmethod
    def _job_line(
        schedule: str, command: str, enabled: bool, expr: CronExpr | None
    ) -> Line:
        text = f"{schedule.strip()} {command.strip()}"
        return Line(
            text if enabled else f"{OFF_MARKER} {text}",
            "job",
            schedule.strip(),
            command.strip(),
            enabled,
            expr,
        )

    def add_job(
        self, schedule: str, command: str, description: str | None = None
    ) -> int:
        expr = self._validate(schedule, command)
        if description:
            self.lines.append(
                Line(f"{DESCRIPTION_PREFIX} {description.strip()}", "comment")
            )
        self.lines.append(self._job_line(schedule, command, True, expr))
        return len(self.lines) - 1

    def update_job(self, index: int, schedule: str, command: str) -> None:
        old = self._line(index)
        self.lines[index] = self._job_line(
            schedule, command, old.enabled, self._validate(schedule, command)
        )

    def delete_job(self, index: int) -> None:
        self._line(index)
        above = self.lines[index - 1] if index else None
        remove_above = above is not None and above.raw.startswith(
            DESCRIPTION_PREFIX
        )  # only our own notes
        del self.lines[index]
        if remove_above:
            del self.lines[index - 1]

    def set_enabled(self, index: int, enabled: bool) -> None:
        line = self._line(index)
        if line.error:
            raise CrontabError(
                f"cannot enable a job with an invalid schedule: {line.error}"
            )
        self.lines[index] = self._job_line(
            line.schedule, line.command, enabled, line.expr
        )

    def wrap_job(
        self, index: int, job_id: str, wrapper: str, timeout: int | None = None
    ) -> None:
        line = self._line(index)
        if wrapper_job_id(line.command):
            raise CrontabError("this job is already wrapped")
        taken = {j.job_id for j in self.jobs() if j.job_id}
        if job_id in taken:
            raise CrontabError(f"job id {job_id!r} is already used by another job")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
            raise CrontabError(
                "job ids may contain only letters, digits, '_', '.' and '-'"
            )
        self.lines[index] = self._job_line(
            line.schedule,
            wrap_command(line.command, job_id, wrapper, timeout),
            line.enabled,
            line.expr,
        )

    def unwrap_job(self, index: int) -> None:
        line = self._line(index)
        self.lines[index] = self._job_line(
            line.schedule, unwrap_command(line.command), line.enabled, line.expr
        )

    def set_env(self, name: str, value: str) -> None:
        """Set ``NAME=value`` above the first job, replacing an existing assignment."""
        text = (
            f"{name}={shlex.quote(value)}"
            if re.search(r"\s", value)
            else f"{name}={value}"
        )
        for i, line in enumerate(self.lines):
            if line.kind == "env" and line.raw.partition("=")[0].strip() == name:
                self.lines[i] = Line(text, "env")
                return
        first_job = next(
            (i for i, line in enumerate(self.lines) if line.kind == "job"),
            len(self.lines),
        )
        self.lines.insert(first_job, Line(text, "env"))


class Backend(Protocol):
    def read(self) -> str: ...
    def write(self, text: str) -> None: ...


class FileCrontab:
    """A crontab kept in a plain file (for ``--crontab FILE``, tests, and trying things out safely)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except OSError as exc:
            raise CrontabError(f"{self.path}: {exc.strerror}") from exc

    def write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(
            dir=self.path.parent, prefix=".crontab-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            os.replace(temp, self.path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise


class SystemCrontab:
    """The real thing, through ``crontab -l`` and ``crontab -`` (which validates before replacing)."""

    def __init__(
        self,
        user: str | None = None,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.user, self._run = user, run

    def _command(self, *args: str) -> list[str]:
        return ["crontab", *(["-u", self.user] if self.user else []), *args]

    def _exec(
        self, command: list[str], text: str | None = None
    ) -> subprocess.CompletedProcess:
        try:
            return self._run(
                command,
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CrontabError(
                "the `crontab` command was not found; use --crontab FILE to edit a file instead"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CrontabError("`crontab` timed out") from exc

    def read(self) -> str:
        result = self._exec(self._command("-l"))
        if result.returncode == 0:
            return result.stdout
        if "no crontab" in result.stderr.lower():
            return ""
        raise CrontabError(
            f"crontab -l failed: {result.stderr.strip() or result.returncode}"
        )

    def write(self, text: str) -> None:
        result = self._exec(self._command("-"), text)
        if result.returncode != 0:
            raise CrontabError(
                f"crontab rejected the new contents (the old crontab is unchanged): {result.stderr.strip()}"
            )


class Session:
    """Edits a crontab with a safety net: conflict check, a backup per write, and undo.

    Every mutation re-reads the backend first; if it no longer matches what was loaded, someone
    (or another cronwatch) edited it meanwhile and we refuse rather than clobber their change.
    """

    def __init__(
        self,
        backend: Backend,
        backup_dir: Path | None = None,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.backend = backend
        self.backup_dir = backup_dir
        self._clock = clock
        self._undo: list[str] = []
        self.loaded_text = ""
        self.crontab = Crontab([])
        self.reload()

    def reload(self) -> None:
        self.loaded_text = self.backend.read()
        self.crontab = Crontab.from_text(self.loaded_text)

    def _backup(self, text: str) -> None:
        if self.backup_dir is None:
            return
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = self._clock().strftime("%Y%m%d-%H%M%S-%f")
        (self.backup_dir / f"crontab-{stamp}.txt").write_text(text, encoding="utf-8")
        for old in sorted(self.backup_dir.glob("crontab-*.txt"))[:-BACKUPS_KEPT]:
            old.unlink(missing_ok=True)

    def _commit(self, new_text: str) -> None:
        if self.backend.read() != self.loaded_text:
            raise ConflictError(
                "the crontab changed since it was loaded; reload before editing"
            )
        self._backup(self.loaded_text)
        self.backend.write(new_text)
        self._undo.append(self.loaded_text)
        self.loaded_text = new_text
        self.crontab = Crontab.from_text(new_text)

    def apply(self, mutate: Callable[[Crontab], object]) -> object:
        """Run ``mutate`` on a copy and write the result; the live crontab is untouched if it raises."""
        draft = self.crontab.copy()
        result = mutate(draft)
        self._commit(draft.to_text())
        return result

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    def undo(self) -> None:
        if not self._undo:
            raise CrontabError("nothing to undo")
        previous = self._undo[-1]
        if self.backend.read() != self.loaded_text:
            raise ConflictError(
                "the crontab changed since it was loaded; reload before undoing"
            )
        self.backend.write(previous)
        self._undo.pop()
        self.loaded_text = previous
        self.crontab = Crontab.from_text(previous)
