"""The timezone cron schedules are written in: the system zone, unless the config names one."""

import os
from datetime import UTC, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _zone(name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(name.strip())
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


def system_zone(
    environ: dict[str, str] | None = None, root: Path = Path("/etc")
) -> tzinfo:
    """The machine's IANA zone: ``$TZ``, else ``/etc/timezone``, else the ``/etc/localtime`` symlink, else UTC.

    A fixed UTC offset would be wrong across daylight saving changes, so a real zone is required.
    """
    env = os.environ if environ is None else environ
    if env.get("TZ") and (zone := _zone(env["TZ"].lstrip(":"))):
        return zone
    try:
        if zone := _zone((root / "timezone").read_text()):
            return zone
    except OSError:
        pass
    try:
        target = os.readlink(root / "localtime")
        if "zoneinfo/" in target and (zone := _zone(target.split("zoneinfo/", 1)[1])):
            return zone
    except OSError:
        pass
    return UTC


def schedule_zone(configured: str | None) -> tzinfo:
    if configured:
        zone = _zone(configured)
        if zone is None:
            raise ValueError(
                f"unknown timezone {configured!r}; use an IANA name such as Europe/Berlin"
            )
        return zone
    return system_zone()
