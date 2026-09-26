"""Structured JSON request logging: every log line goes to stdout *and* gets
appended to logs/requests.jsonl, from a single event dict so the two outputs
can never drift out of sync with each other.

Uses structlog (the modern structured-logging library) instead of stdlib
`logging` + a hand-rolled JSON `Formatter`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import structlog

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "requests.jsonl"


def _append_to_file(
    _logger: object, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """A structlog processor: persist the event dict as one JSON line before
    the later JSONRenderer processor turns it into a rendered string for
    stdout. Runs on every log call, so the file and stdout are always the
    same events."""
    LOG_DIR.mkdir(exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event_dict, default=str) + "\n")
    return event_dict


def configure_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            _append_to_file,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger() -> structlog.BoundLogger:
    return structlog.get_logger()
