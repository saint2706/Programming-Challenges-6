"""When to alert (a small state machine), what to say, and how to deliver it.

Policy (:func:`decide`) is pure so the awkward cases are unit-testable: alert once when a job
starts failing, stay quiet while it keeps failing, nag again only after ``renotify_after``, and say
so once when it recovers. A cron job that fails every minute must not send a thousand emails.
"""

import json
import re
import smtplib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.message import EmailMessage
from typing import Protocol

from cronwatch.config import Config, DesktopConfig, EmailConfig, WebhookConfig
from cronwatch.store import JobState, Run, Store

OUTPUT_LINES = 20
OUTPUT_CHARS = 1800
WEBHOOK_TIMEOUT = 10
_URL = re.compile(r"https?://\S+")


class AlertError(Exception):
    """Delivery failed. The message never contains a URL or credential."""


@dataclass(frozen=True)
class Alert:
    kind: str  # failed | still-failing | recovered | missed | test
    job_id: str
    host: str
    when: datetime
    command: str = ""
    status: str | None = None
    exit_code: int | None = None
    duration_s: float | None = None
    output_tail: str = ""
    consecutive_failures: int = 0
    scheduled_for: datetime | None = (
        None  # missed runs: the latest instant that passed with no run
    )
    missed_count: int = 0


def alert_from_run(kind: str, run: Run, failures: int) -> Alert:
    return Alert(
        kind, run.job_id, run.host, run.ended_at or run.started_at, run.command, run.status, run.exit_code,
        run.duration_s, run.output_tail, failures,
    )  # fmt: skip


def decide(
    state: JobState, succeeded: bool, config: Config, now: datetime
) -> tuple[JobState, str | None]:
    """The next state and the alert kind to send (or None) after a run finishes.

    ``alert_after`` consecutive failures trigger the first alert; ``renotify_after`` must pass before
    the next one; a success after an alert triggers a single ``recovered`` (if enabled).
    """
    if succeeded:
        kind = "recovered" if state.alerted_at and config.recovery else None
        return replace(
            state, consecutive_failures=0, alerted_at=None, last_status="ok"
        ), kind
    failures = state.consecutive_failures + 1
    updated = replace(state, consecutive_failures=failures, last_status="failed")
    if failures < config.alert_after:
        return updated, None
    if state.alerted_at is None:
        return replace(updated, alerted_at=now), "failed"
    if now - state.alerted_at >= config.renotify_after:
        return replace(updated, alerted_at=now), "still-failing"
    return updated, None


def evaluate_run(
    store: Store, config: Config, run: Run, now: datetime | None = None
) -> Alert | None:
    """Apply :func:`decide` to a finished run, persist the new state, and return the alert to send."""
    now = now or datetime.now(UTC)
    state, kind = decide(store.get_state(run.job_id), run.succeeded, config, now)
    store.save_state(state)
    return alert_from_run(kind, run, state.consecutive_failures) if kind else None


def _tail(text: str) -> str:
    lines = text.rstrip().splitlines()[-OUTPUT_LINES:]
    return "\n".join(lines)[-OUTPUT_CHARS:]


_TITLES = {
    "failed": "FAILED",
    "still-failing": "STILL FAILING",
    "recovered": "RECOVERED",
    "missed": "MISSED RUN",
    "test": "TEST",
}


def render(alert: Alert, include_output: bool) -> tuple[str, str]:
    """(subject, body). Output is opt-in: job output can hold secrets and the channel may be a third party."""
    title = _TITLES[alert.kind]
    detail = ""
    if alert.status == "timeout":
        detail = " (timed out)"
    elif alert.exit_code not in (None, 0) and alert.kind in {"failed", "still-failing"}:
        detail = f" (exit {alert.exit_code})"
    subject = f"[cronwatch] {title}: {alert.job_id} on {alert.host}{detail}"
    lines = [
        f"Job:      {alert.job_id}",
        f"Host:     {alert.host}",
        f"When:     {alert.when.astimezone(UTC):%Y-%m-%d %H:%M:%S} UTC",
    ]
    if alert.command:
        lines.append(f"Command:  {alert.command}")
    if alert.kind == "missed" and alert.scheduled_for:
        lines.append(
            f"Expected: {alert.scheduled_for.astimezone(UTC):%Y-%m-%d %H:%M} UTC, but no run was recorded"
        )
    if alert.kind == "missed" and alert.missed_count > 1:
        lines.append(f"Missed runs: {alert.missed_count} (the latest is shown above)")
    if alert.kind in {"failed", "still-failing"}:
        lines.append(f"Failures in a row: {alert.consecutive_failures}")
    if alert.duration_s is not None:
        lines.append(f"Duration: {alert.duration_s:.1f}s")
    if alert.kind == "recovered":
        lines.append("The job succeeded again.")
    if include_output and alert.output_tail.strip():
        lines += ["", "Last output:", _tail(alert.output_tail)]
    return subject, "\n".join(lines)


def _scrub(message: str) -> str:
    return _URL.sub("<url>", message)


class Channel(Protocol):
    name: str

    def send(self, alert: Alert) -> None: ...


class WebhookChannel:
    name = "webhook"

    def __init__(
        self,
        config: WebhookConfig,
        opener: Callable = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        retries: int = 2,
    ) -> None:
        self.config, self._open, self._sleep, self._retries = (
            config,
            opener,
            sleep,
            retries,
        )

    def payload(self, alert: Alert) -> dict:
        subject, body = render(alert, self.config.include_output)
        text = f"{subject}\n{body}"
        if self.config.format == "slack":
            return {"text": text}
        if self.config.format == "discord":
            return {"content": text[:2000]}
        document = {
            "kind": alert.kind, "job": alert.job_id, "host": alert.host, "status": alert.status, "exit_code": alert.exit_code,
            "duration_s": alert.duration_s, "when": alert.when.astimezone(UTC).isoformat(timespec="seconds"),
            "consecutive_failures": alert.consecutive_failures, "message": subject,
        }  # fmt: skip
        if self.config.include_output:
            document["output"] = _tail(alert.output_tail)
        return document

    def send(self, alert: Alert) -> None:
        body = json.dumps(self.payload(alert)).encode()
        request = urllib.request.Request(
            self.config.url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "cronwatch"},
        )
        failure = "unknown error"
        for attempt in range(self._retries + 1):
            try:
                with self._open(request, timeout=WEBHOOK_TIMEOUT) as response:
                    if 200 <= response.status < 300:
                        return
                    failure = f"HTTP {response.status}"
            except urllib.error.HTTPError as exc:
                failure = f"HTTP {exc.code}"
                if 400 <= exc.code < 500 and exc.code != 429:
                    break  # a client error will not fix itself on retry
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                failure = _scrub(str(getattr(exc, "reason", exc)))
            if attempt < self._retries:
                self._sleep(2**attempt)
        raise AlertError(f"webhook delivery failed: {failure}")


class EmailChannel:
    name = "email"

    def __init__(
        self,
        config: EmailConfig,
        environ: Mapping[str, str],
        smtp_factory: Callable | None = None,
    ) -> None:
        self.config, self._environ = config, environ
        self._factory = smtp_factory or (
            smtplib.SMTP_SSL if config.ssl else smtplib.SMTP
        )

    def send(self, alert: Alert) -> None:
        cfg = self.config
        subject, body = render(alert, cfg.include_output)
        message = EmailMessage()
        message["Subject"], message["From"], message["To"] = (
            subject,
            cfg.sender,
            ", ".join(cfg.to),
        )
        message.set_content(body)
        password = None
        if cfg.password_env:
            password = self._environ.get(cfg.password_env)
            if not password:
                raise AlertError(f"environment variable {cfg.password_env} is not set")
        try:
            with self._factory(cfg.host, cfg.port, timeout=15) as smtp:
                if cfg.starttls and not cfg.ssl:
                    smtp.starttls()
                if cfg.username and password:
                    smtp.login(cfg.username, password)
                smtp.send_message(message)
        except (smtplib.SMTPException, OSError) as exc:
            raise AlertError(
                f"email delivery failed: {_scrub(type(exc).__name__ + ': ' + str(exc))}"
            ) from exc


class DesktopChannel:
    """notify-send on Linux, osascript on macOS. From cron there is often no desktop session to talk to."""

    name = "desktop"

    def __init__(
        self,
        config: DesktopConfig,
        run: Callable = subprocess.run,
        platform: str = sys.platform,
    ) -> None:
        self.config, self._run, self._platform = config, run, platform

    def send(self, alert: Alert) -> None:
        subject, body = render(alert, include_output=False)
        if self._platform.startswith("linux"):
            urgency = "low" if alert.kind == "recovered" else "critical"
            command = [
                "notify-send",
                "--app-name=cronwatch",
                f"--urgency={urgency}",
                subject,
                body,
            ]
        elif self._platform == "darwin":
            command = [
                "osascript",
                "-e",
                f"display notification {json.dumps(body)} with title {json.dumps(subject)}",
            ]
        else:
            raise AlertError("desktop notifications are not supported on this platform")
        try:
            result = self._run(command, capture_output=True, timeout=10, check=False)
        except FileNotFoundError as exc:
            raise AlertError(f"{command[0]} is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise AlertError(f"{command[0]} timed out") from exc
        if result.returncode != 0:
            raise AlertError(
                f"{command[0]} exited with status {result.returncode} (is a desktop session available to cron?)"
            )


def build_channels(config: Config, environ: Mapping[str, str]) -> list[Channel]:
    channels: list[Channel] = []
    if config.webhook:
        channels.append(WebhookChannel(config.webhook))
    if config.email:
        channels.append(EmailChannel(config.email, environ))
    if config.desktop:
        channels.append(DesktopChannel(config.desktop))
    return channels


@dataclass(frozen=True)
class Delivery:
    channel: str
    ok: bool
    detail: str


class Alerter:
    def __init__(
        self,
        channels: list[Channel],
        store: Store | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.channels, self.store, self._clock = channels, store, clock

    def dispatch(self, alert: Alert, run_id: int | None = None) -> list[Delivery]:
        """Send through every channel. One broken channel never blocks the others, and nothing raises."""
        results = []
        for channel in self.channels:
            try:
                channel.send(alert)
                outcome = Delivery(channel.name, True, "")
            except Exception as exc:  # noqa: BLE001 - delivery must never take the job's result down with it
                detail = (
                    str(exc)
                    if isinstance(exc, AlertError)
                    else f"{type(exc).__name__}: {_scrub(str(exc))}"
                )
                outcome = Delivery(channel.name, False, detail)
            results.append(outcome)
            if self.store is not None:
                try:
                    self.store.log_alert(
                        alert.job_id,
                        run_id,
                        alert.kind,
                        outcome.channel,
                        self._clock(),
                        outcome.ok,
                        outcome.detail,
                    )
                except Exception:  # noqa: BLE001, S110
                    pass
        return results
