"""``config.toml``: alert thresholds and channels, with secrets kept out of the file."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import tomllib

_DURATION = re.compile(r"^\s*(\d+)\s*([smhd])\s*$")
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class ConfigError(ValueError):
    pass


def parse_duration(text: str | int) -> timedelta:
    """``"90s"``, ``"30m"``, ``"6h"``, ``"2d"``; a bare number is seconds."""
    if isinstance(text, int):
        return timedelta(seconds=text)
    match = _DURATION.match(str(text))
    if not match:
        raise ConfigError(f"{text!r} is not a duration; use e.g. 30m, 6h or 2d")
    return timedelta(seconds=int(match.group(1)) * _UNITS[match.group(2)])


@dataclass(frozen=True)
class WebhookConfig:
    url: str
    format: str = "json"  # json | slack | discord
    include_output: bool = False


@dataclass(frozen=True)
class EmailConfig:
    host: str
    sender: str
    to: tuple[str, ...]
    port: int = 587
    starttls: bool = True
    ssl: bool = False
    username: str | None = None
    password_env: str | None = None
    include_output: bool = True


@dataclass(frozen=True)
class DesktopConfig:
    enabled: bool = True


@dataclass(frozen=True)
class Config:
    alert_after: int = 1
    renotify_after: timedelta = timedelta(hours=6)
    recovery: bool = True
    retention: timedelta = timedelta(days=90)
    grace: timedelta = timedelta(minutes=5)
    timezone: str | None = None
    webhook: WebhookConfig | None = None
    email: EmailConfig | None = None
    desktop: DesktopConfig | None = None


def _table(data: Mapping, key: str) -> Mapping:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table")
    return value


def _check_keys(table: Mapping, allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown setting(s) {', '.join(unknown)}")


def load_config(path: Path, environ: Mapping[str, str] | None = None) -> Config:
    env = os.environ if environ is None else environ
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Config()
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    alerts = _table(data, "alerts")
    _check_keys(
        alerts,
        {
            "alert_after",
            "renotify_after",
            "recovery",
            "retention",
            "grace",
            "timezone",
            "webhook",
            "email",
            "desktop",
        },
        "[alerts]",
    )
    alert_after = alerts.get("alert_after", 1)
    if not isinstance(alert_after, int) or alert_after < 1:
        raise ConfigError("[alerts] alert_after must be a whole number, 1 or more")

    webhook = None
    if "webhook" in alerts:
        table = _table(alerts, "webhook")
        _check_keys(
            table, {"url", "url_env", "format", "include_output"}, "[alerts.webhook]"
        )
        url = table.get("url") or env.get(table.get("url_env", ""), "")
        if not url:
            raise ConfigError(
                "[alerts.webhook] needs `url`, or `url_env` naming a set environment variable (keeps the URL out of the file)"
            )
        if not re.match(r"^https?://", url):
            raise ConfigError(
                "[alerts.webhook] the URL must start with http:// or https://"
            )
        fmt = table.get("format", "json")
        if fmt not in {"json", "slack", "discord"}:
            raise ConfigError("[alerts.webhook] format must be json, slack or discord")
        webhook = WebhookConfig(url, fmt, bool(table.get("include_output", False)))

    email = None
    if "email" in alerts:
        table = _table(alerts, "email")
        if "password" in table:
            raise ConfigError(
                "[alerts.email] do not put the password in the file; set `password_env` to the name of an environment variable"
            )
        _check_keys(
            table,
            {
                "host",
                "port",
                "starttls",
                "ssl",
                "username",
                "password_env",
                "from",
                "to",
                "include_output",
            },
            "[alerts.email]",
        )
        recipients = table.get("to", [])
        recipients = [recipients] if isinstance(recipients, str) else recipients
        if not table.get("host") or not table.get("from") or not recipients:
            raise ConfigError("[alerts.email] needs `host`, `from` and `to`")
        email = EmailConfig(
            host=table["host"], sender=table["from"], to=tuple(recipients), port=int(table.get("port", 465 if table.get("ssl") else 587)),
            starttls=bool(table.get("starttls", not table.get("ssl", False))), ssl=bool(table.get("ssl", False)),
            username=table.get("username"), password_env=table.get("password_env"), include_output=bool(table.get("include_output", True)),
        )  # fmt: skip

    desktop = None
    if "desktop" in alerts:
        table = _table(alerts, "desktop")
        _check_keys(table, {"enabled"}, "[alerts.desktop]")
        desktop = DesktopConfig(bool(table.get("enabled", True)))

    return Config(
        alert_after=alert_after,
        renotify_after=parse_duration(alerts.get("renotify_after", "6h")),
        recovery=bool(alerts.get("recovery", True)),
        retention=parse_duration(alerts.get("retention", "90d")),
        grace=parse_duration(alerts.get("grace", "5m")),
        timezone=alerts.get("timezone"),
        webhook=webhook,
        email=email,
        desktop=desktop if desktop and desktop.enabled else None,
    )
