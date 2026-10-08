"""Thin wrapper around `plyer`'s cross-platform desktop notifications.

Kept in its own module for two reasons: tests can monkeypatch `notify`
without needing a real display/notification backend, and a missing or
broken backend (headless CI, a Linux box with no notification daemon
running, etc.) degrades to a logged warning instead of crashing the
background scheduler thread that calls it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

APP_NAME = "Subscription Tracker"


def notify(title: str, message: str, timeout: int = 10) -> bool:
    """Fire a real OS desktop notification. Returns True on success, False
    (after logging a warning) if the platform backend isn't available."""
    try:
        from plyer import notification

        notification.notify(
            title=title, message=message, timeout=timeout, app_name=APP_NAME
        )
        return True
    except Exception as exc:  # noqa: BLE001 -- plyer's backends raise different,
        # undocumented exception types per platform (missing D-Bus session,
        # no notification daemon, import errors on an unsupported OS); any
        # of them must degrade to a warning, never crash the caller.
        logger.warning("Desktop notification failed: %s", exc)
        return False
