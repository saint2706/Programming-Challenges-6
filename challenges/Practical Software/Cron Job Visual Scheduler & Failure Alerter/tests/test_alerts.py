import json
import smtplib
import subprocess
import urllib.error
from datetime import timedelta
from email import message_from_string

import pytest
from cronwatch.alerts import (
    Alert,
    Alerter,
    AlertError,
    DesktopChannel,
    EmailChannel,
    WebhookChannel,
    build_channels,
    decide,
    evaluate_run,
    render,
)
from cronwatch.config import Config, DesktopConfig, EmailConfig, WebhookConfig
from cronwatch.store import FAILED, TIMEOUT, JobState, Store
from helpers import SMTPServer, WebhookServer, at

CFG = Config(alert_after=1, renotify_after=timedelta(hours=6))


def step(state, ok, now, config=CFG):
    return decide(state, ok, config, now)


# --- policy ---


def test_first_failure_alerts_and_records_when():
    state, kind = step(JobState("j"), False, at(0))
    assert (
        kind == "failed"
        and state.consecutive_failures == 1
        and state.alerted_at == at(0)
    )


def test_continued_failure_stays_quiet_until_renotify_after():
    state, _ = step(JobState("j"), False, at(0))
    for minute in (1, 60, 359):
        state, kind = step(state, False, at(minute))
        assert kind is None
    state, kind = step(state, False, at(360))
    assert kind == "still-failing" and state.alerted_at == at(360)
    _, kind = step(state, False, at(361))
    assert kind is None  # the renotify clock restarts


def test_recovery_alerts_once_then_state_resets():
    state, _ = step(JobState("j"), False, at(0))
    state, kind = step(state, True, at(5))
    assert (
        kind == "recovered"
        and state.consecutive_failures == 0
        and state.alerted_at is None
    )
    _, kind = step(state, True, at(10))
    assert kind is None


def test_success_without_a_prior_alert_is_silent():
    assert step(JobState("j"), True, at(0))[1] is None


def test_recovery_can_be_turned_off():
    cfg = Config(recovery=False)
    state, _ = step(JobState("j"), False, at(0), cfg)
    state, kind = step(state, True, at(5), cfg)
    assert (
        kind is None and state.alerted_at is None
    )  # and the next failure alerts afresh
    assert step(state, False, at(6), cfg)[1] == "failed"


def test_alert_after_n_ignores_blips_and_a_success_resets_the_count():
    cfg = Config(alert_after=3)
    state = JobState("j")
    kinds = []
    for ok in (False, False, True, False, False, False):
        state, kind = step(state, ok, at(len(kinds)), cfg)
        kinds.append(kind)
    assert kinds == [None, None, None, None, None, "failed"]


def test_a_blip_below_the_threshold_never_triggers_a_recovery_message():
    cfg = Config(alert_after=3)
    state, _ = step(JobState("j"), False, at(0), cfg)
    state, kind = step(state, True, at(1), cfg)
    assert kind is None


def test_evaluate_run_persists_state_and_builds_the_alert(tmp_path):
    store = Store(tmp_path / "d.db")
    run_id = store.start_run("backup", "pg_dump", "web1", at(0))
    store.finish_run(run_id, at(1), 2, FAILED, "pg_dump: connection refused\n")
    alert = evaluate_run(store, CFG, store.get_run(run_id), at(1))
    assert (
        alert.kind,
        alert.job_id,
        alert.exit_code,
        alert.host,
        alert.consecutive_failures,
    ) == ("failed", "backup", 2, "web1", 1)
    assert "connection refused" in alert.output_tail
    assert store.get_state("backup").alerted_at == at(1)
    assert evaluate_run(store, CFG, store.get_run(run_id), at(2)) is None


# --- rendering ---

ALERT = Alert(
    "failed",
    "backup",
    "web1",
    at(0),
    "pg_dump --all",
    FAILED,
    2,
    12.5,
    "line1\nsecret-ish output\n",
    3,
)


def test_subject_and_body_describe_the_failure():
    subject, body = render(ALERT, include_output=False)
    assert subject == "[cronwatch] FAILED: backup on web1 (exit 2)"
    for fragment in [
        "Job:      backup",
        "Host:     web1",
        "Command:  pg_dump --all",
        "Failures in a row: 3",
        "Duration: 12.5s",
        "2026-10-09 12:00:00 UTC",
    ]:
        assert fragment in body


def test_output_is_included_only_on_request_and_is_bounded():
    assert "secret-ish" not in render(ALERT, False)[1]
    assert "secret-ish output" in render(ALERT, True)[1]
    long = Alert(
        "failed", "j", "h", at(0), output_tail="\n".join(f"line{i}" for i in range(100))
    )
    body = render(long, True)[1]
    assert "line99" in body and "line70" not in body and "line80" in body


def test_other_kinds_read_naturally():
    assert (
        "(timed out)"
        in render(Alert("failed", "j", "h", at(0), status=TIMEOUT), False)[0]
    )
    assert render(Alert("recovered", "j", "h", at(0)), False)[1].endswith(
        "The job succeeded again."
    )
    subject, body = render(
        Alert("missed", "j", "h", at(0), scheduled_for=at(-5)), False
    )
    assert (
        "MISSED RUN" in subject
        and "2026-10-09 11:55 UTC, but no run was recorded" in body
    )
    assert "TEST" in render(Alert("test", "j", "h", at(0)), False)[0]


# --- webhook, against a real HTTP server ---


def test_webhook_posts_json_with_the_documented_fields():
    with WebhookServer() as server:
        WebhookChannel(WebhookConfig(server.url)).send(ALERT)
    (request,) = server.requests
    assert (
        request["path"] == "/hook?token=SECRETTOKEN"
        and request["headers"]["Content-Type"] == "application/json"
    )
    document = json.loads(request["body"])
    assert (
        document["kind"] == "failed"
        and document["job"] == "backup"
        and document["exit_code"] == 2
    )
    assert document["when"] == "2026-10-09T12:00:00+00:00" and "output" not in document


def test_webhook_output_is_opt_in():
    with WebhookServer() as server:
        WebhookChannel(WebhookConfig(server.url, include_output=True)).send(ALERT)
    assert "secret-ish" in json.loads(server.requests[0]["body"])["output"]


@pytest.mark.parametrize(("fmt", "key"), [("slack", "text"), ("discord", "content")])
def test_slack_and_discord_formats(fmt, key):
    payload = WebhookChannel(WebhookConfig("http://x", fmt)).payload(ALERT)
    assert list(payload) == [key] and "FAILED: backup" in payload[key]


def test_discord_content_is_capped_at_its_2000_character_limit():
    huge = Alert("failed", "j", "h", at(0), command="x" * 5000)
    assert (
        len(
            WebhookChannel(WebhookConfig("http://x", "discord")).payload(huge)[
                "content"
            ]
        )
        == 2000
    )


def test_webhook_retries_server_errors_then_succeeds():
    sleeps = []
    with WebhookServer(statuses=[503, 502, 200]) as server:
        WebhookChannel(WebhookConfig(server.url), sleep=sleeps.append).send(ALERT)
    assert len(server.requests) == 3 and sleeps == [1, 2]


def test_webhook_gives_up_after_retries_and_does_not_leak_the_url():
    with (
        WebhookServer(statuses=[500, 500, 500]) as server,
        pytest.raises(AlertError) as error,
    ):
        WebhookChannel(WebhookConfig(server.url), sleep=lambda _s: None).send(ALERT)
    assert "HTTP 500" in str(error.value) and "SECRETTOKEN" not in str(error.value)
    assert len(server.requests) == 3


def test_webhook_does_not_retry_client_errors():
    with (
        WebhookServer(statuses=[404, 200]) as server,
        pytest.raises(AlertError, match="HTTP 404"),
    ):
        WebhookChannel(WebhookConfig(server.url), sleep=lambda _s: None).send(ALERT)
    assert len(server.requests) == 1


def test_webhook_connection_failure_is_scrubbed():
    def refuse(_request, timeout):
        raise urllib.error.URLError(
            "failed to reach https://hooks.example.com/SECRETTOKEN"
        )

    with pytest.raises(AlertError) as error:
        WebhookChannel(
            WebhookConfig("https://hooks.example.com/SECRETTOKEN"),
            opener=refuse,
            sleep=lambda _s: None,
            retries=1,
        ).send(ALERT)
    assert "SECRETTOKEN" not in str(error.value) and "<url>" in str(error.value)


# --- email, against a real SMTP server ---

EMAIL = EmailConfig(
    "127.0.0.1",
    "cron@example.com",
    ("ops@example.com", "dev@example.com"),
    starttls=False,
)


def test_email_is_delivered_with_headers_and_body():
    with SMTPServer() as server:
        EmailChannel(
            EmailConfig(
                "127.0.0.1",
                "cron@example.com",
                ("ops@example.com", "dev@example.com"),
                port=server.port,
                starttls=False,
            ),
            {},
        ).send(ALERT)
    (received,) = server.messages
    assert received["from"] == "<cron@example.com>" and received["to"] == [
        "<ops@example.com>",
        "<dev@example.com>",
    ]
    message = message_from_string(received["data"])
    assert message["Subject"] == "[cronwatch] FAILED: backup on web1 (exit 2)"
    assert message["To"] == "ops@example.com, dev@example.com"
    assert (
        "secret-ish output" in message.get_payload()
    )  # email includes output by default


def test_email_login_uses_the_password_from_the_environment_and_starttls():
    events = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            events.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def starttls(self):
            events.append("starttls")

        def login(self, user, password):
            events.append(("login", user, password))

        def send_message(self, message):
            events.append(("send", message["Subject"]))

    config = EmailConfig(
        "smtp.example.com", "a@b", ("c@d",), username="cron", password_env="SMTP_PW"
    )
    EmailChannel(config, {"SMTP_PW": "s3cret"}, FakeSMTP).send(ALERT)
    assert events[:3] == [
        ("connect", "smtp.example.com", 587),
        "starttls",
        ("login", "cron", "s3cret"),
    ]


def test_email_missing_password_variable_names_the_variable_only():
    config = EmailConfig("h", "a@b", ("c@d",), username="u", password_env="SMTP_PW")
    with pytest.raises(AlertError, match="SMTP_PW is not set"):
        EmailChannel(config, {}).send(ALERT)


def test_email_server_failure_becomes_an_alert_error():
    class Broken:
        def __init__(self, *_a, **_k):
            raise smtplib.SMTPConnectError(421, b"busy")

    with pytest.raises(AlertError, match="email delivery failed"):
        EmailChannel(EMAIL, {}, Broken).send(ALERT)


# --- desktop ---


def recorder(returncode=0, raises=None):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if raises:
            raise raises
        return subprocess.CompletedProcess(command, returncode, b"", b"")

    run.calls = calls
    return run


def test_desktop_linux_uses_notify_send_without_a_shell_and_never_includes_output():
    run = recorder()
    DesktopChannel(DesktopConfig(), run, "linux").send(ALERT)
    (command,) = run.calls
    assert command[0] == "notify-send" and "--urgency=critical" in command
    assert not any("secret-ish" in part for part in command)
    DesktopChannel(DesktopConfig(), run, "linux").send(
        Alert("recovered", "j", "h", at(0))
    )
    assert "--urgency=low" in run.calls[1]


def test_desktop_macos_quotes_text_for_applescript():
    run = recorder()
    DesktopChannel(DesktopConfig(), run, "darwin").send(
        Alert("failed", 'we"ird', "h", at(0))
    )
    script = run.calls[0][2]
    assert run.calls[0][:2] == ["osascript", "-e"] and 'we\\"ird' in script


def test_desktop_errors():
    with pytest.raises(AlertError, match="not supported"):
        DesktopChannel(DesktopConfig(), recorder(), "win32").send(ALERT)
    with pytest.raises(AlertError, match="not installed"):
        DesktopChannel(
            DesktopConfig(), recorder(raises=FileNotFoundError()), "linux"
        ).send(ALERT)
    with pytest.raises(AlertError, match="desktop session"):
        DesktopChannel(DesktopConfig(), recorder(returncode=1), "linux").send(ALERT)


# --- dispatch ---


class Fake:
    def __init__(self, name, error=None):
        self.name, self.error, self.sent = name, error, []

    def send(self, alert):
        if self.error:
            raise self.error
        self.sent.append(alert)


def test_one_failing_channel_does_not_block_the_others_and_nothing_raises(tmp_path):
    store = Store(tmp_path / "d.db")
    good, bad, boom = (
        Fake("good"),
        Fake("bad", AlertError("refused")),
        Fake("boom", RuntimeError("see https://x.example/SECRETTOKEN")),
    )
    results = Alerter([bad, boom, good], store, lambda: at(0)).dispatch(ALERT, run_id=7)
    assert [(r.channel, r.ok) for r in results] == [
        ("bad", False),
        ("boom", False),
        ("good", True),
    ]
    assert good.sent == [ALERT]
    logged = {a.channel: a for a in store.alerts()}
    assert (
        logged["bad"].detail == "refused" and not logged["bad"].ok and logged["good"].ok
    )
    assert "SECRETTOKEN" not in logged["boom"].detail and logged["good"].run_id == 7


def test_alerter_survives_a_broken_store():
    class BrokenStore:
        def log_alert(self, *_a):
            raise OSError("disk full")

    assert Alerter([Fake("good")], BrokenStore()).dispatch(ALERT)[0].ok


def test_build_channels_follows_the_config():
    config = Config(
        webhook=WebhookConfig("http://x"), email=EMAIL, desktop=DesktopConfig()
    )
    assert [c.name for c in build_channels(config, {})] == [
        "webhook",
        "email",
        "desktop",
    ]
    assert build_channels(Config(), {}) == []
