from datetime import timedelta

import pytest
from cronwatch.config import Config, ConfigError, load_config, parse_duration


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_missing_file_means_defaults_with_no_channels(tmp_path):
    config = load_config(tmp_path / "nope.toml")
    assert (
        config == Config()
        and config.webhook is None
        and config.email is None
        and config.desktop is None
    )
    assert (config.alert_after, config.renotify_after, config.recovery) == (
        1,
        timedelta(hours=6),
        True,
    )


def test_durations():
    assert parse_duration("90s") == timedelta(seconds=90)
    assert parse_duration("30m") == timedelta(minutes=30) and parse_duration(
        "2d"
    ) == timedelta(days=2)
    assert parse_duration(120) == timedelta(seconds=120)
    for bad in ["soon", "5", "1w", "-5m", ""]:
        with pytest.raises(ConfigError, match="not a duration"):
            parse_duration(bad)


def test_full_config(tmp_path):
    path = write(
        tmp_path,
        """
[alerts]
alert_after = 3
renotify_after = "12h"
recovery = false
retention = "30d"
grace = "10m"
timezone = "Europe/Berlin"
[alerts.webhook]
url = "https://hooks.example.com/x"
format = "slack"
include_output = true
[alerts.email]
host = "smtp.example.com"
from = "cron@example.com"
to = ["a@example.com", "b@example.com"]
username = "cron"
password_env = "SMTP_PW"
[alerts.desktop]
enabled = true
""",
    )
    c = load_config(path)
    assert (
        c.alert_after,
        c.renotify_after,
        c.recovery,
        c.retention,
        c.grace,
        c.timezone,
    ) == (
        3,
        timedelta(hours=12),
        False,
        timedelta(days=30),
        timedelta(minutes=10),
        "Europe/Berlin",
    )
    assert (c.webhook.url, c.webhook.format, c.webhook.include_output) == (
        "https://hooks.example.com/x",
        "slack",
        True,
    )
    assert (
        c.email.port,
        c.email.starttls,
        c.email.ssl,
        c.email.to,
        c.email.include_output,
    ) == (587, True, False, ("a@example.com", "b@example.com"), True)
    assert c.desktop is not None


def test_webhook_url_can_come_from_the_environment(tmp_path):
    path = write(
        tmp_path,
        '[alerts.webhook]\nurl_env = "HOOK"\n'.replace(
            "[alerts.webhook]", "[alerts]\n[alerts.webhook]"
        ),
    )
    assert (
        load_config(path, {"HOOK": "https://x.example/h"}).webhook.url
        == "https://x.example/h"
    )
    with pytest.raises(ConfigError, match="url_env"):
        load_config(path, {})


def test_email_ssl_defaults_to_port_465_without_starttls(tmp_path):
    path = write(
        tmp_path,
        '[alerts.email]\nhost = "h"\nfrom = "a@b"\nto = "c@d"\nssl = true\n'.replace(
            "[alerts.email]", "[alerts]\n[alerts.email]"
        ),
    )
    email = load_config(path).email
    assert (email.port, email.ssl, email.starttls, email.to) == (
        465,
        True,
        False,
        ("c@d",),
    )


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ("[alerts]\nalert_after = 0", "alert_after"),
        ('[alerts]\nalert_after = "x"', "alert_after"),
        ("[alerts]\nbogus = 1", "unknown setting"),
        ('[alerts]\n[alerts.webhook]\nurl = "ftp://x"', "http"),
        ('[alerts]\n[alerts.webhook]\nurl = "https://x"\nformat = "teams"', "format"),
        (
            '[alerts]\n[alerts.email]\nhost = "h"\nfrom = "a"\nto = ["b"]\npassword = "hunter2"',
            "password_env",
        ),
        ('[alerts]\n[alerts.email]\nhost = "h"', "needs `host`, `from` and `to`"),
        ('[alerts]\nrenotify_after = "soon"', "not a duration"),
        ("not toml ===", "config.toml"),
        ("alerts = 3", "must be a table"),
    ],
)
def test_bad_config_is_rejected_with_a_specific_message(tmp_path, toml, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, toml))


def test_inline_password_is_never_echoed_in_the_error(tmp_path):
    path = write(
        tmp_path,
        '[alerts]\n[alerts.email]\nhost = "h"\nfrom = "a"\nto = ["b"]\npassword = "hunter2-secret"',
    )
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert "hunter2-secret" not in str(error.value)


def test_disabled_desktop_section_means_no_desktop_channel(tmp_path):
    path = write(tmp_path, "[alerts]\n[alerts.desktop]\nenabled = false\n")
    assert load_config(path).desktop is None
