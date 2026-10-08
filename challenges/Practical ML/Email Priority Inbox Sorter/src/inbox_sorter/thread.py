"""Subject threading and the "did the owner act on this?" label.

The Enron dump has no ``In-Reply-To``/``References`` headers, so a reply is
recognised heuristically: an owner-sent message whose normalized subject equals
the received one, sent within a window after receipt, and either a ``Re:`` that is
addressed to the original sender or a ``Fw:``/``Fwd:``.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Literal

import polars as pl

from inbox_sorter.data import Message

WINDOW_DAYS = 14

# "Re:", "RE :", "Fw:", "FWD:" at the start; the colon is required so words like
# "Reservations" are not mistaken for a prefix.
_PREFIX = re.compile(r"^\s*(re|fwd?)\s*:\s*", re.IGNORECASE)
_SPACES = re.compile(r"\s+")

Kind = Literal["reply", "forward", "other"]


def strip_prefixes(subject: str) -> tuple[str, int, int]:
    """Drop a ``Re:``/``Fw:``/``Fwd:`` chain: ``(rest, n_re, n_fw)``."""
    s, n_re, n_fw = subject, 0, 0
    while (m := _PREFIX.match(s)) is not None:
        if m.group(1).lower() == "re":
            n_re += 1
        else:
            n_fw += 1
        s = s[m.end() :]
    return s, n_re, n_fw


def normalize_subject(subject: str) -> str:
    """Strip any chain of ``Re:``/``Fw:``/``Fwd:``, collapse whitespace, lowercase."""
    return _SPACES.sub(" ", strip_prefixes(subject)[0]).strip().lower()


def kind_of(subject: str) -> Kind:
    """Reply/forward by the outermost prefix (``Fw: Re: x`` is a forward)."""
    m = _PREFIX.match(subject)
    if m is None:
        return "other"
    return "reply" if m.group(1).lower() == "re" else "forward"


def censor_cutoff(sent: list[Message], window_days: int = WINDOW_DAYS) -> datetime:
    """Mail received after this has no complete reply window inside the dump."""
    dated = [m.date for m in sent if m.date is not None]
    return max(dated) - timedelta(days=window_days)


def label_received(
    received: list[Message],
    sent: list[Message],
    owner: str,
    window_days: int = WINDOW_DAYS,
) -> pl.DataFrame:
    """One row per received message: ``message_id, acted, kind, acted_at``.

    ``acted_at`` is the earliest qualifying owner message in
    ``[receipt, receipt + window_days]`` (both ends inclusive); a message sent
    before receipt can never be its answer.
    """
    # normalized subject -> owner messages (time, kind, recipients), time-sorted
    by_subject: dict[str, list[tuple[datetime, Kind, frozenset[str]]]] = defaultdict(
        list
    )
    for s in sent:
        if s.sender != owner or s.date is None:
            continue
        key = normalize_subject(s.subject)
        if key:
            by_subject[key].append((s.date, kind_of(s.subject), frozenset(s.to + s.cc)))
    for entries in by_subject.values():
        entries.sort(key=lambda e: e[0])
    times_by_subject = {k: [e[0] for e in v] for k, v in by_subject.items()}
    window = timedelta(days=window_days)

    rows = []
    for m in received:
        acted_at: datetime | None = None
        kind: str | None = None
        key = normalize_subject(m.subject)
        entries = by_subject.get(key) if key and m.date is not None else None
        if entries:
            start = bisect_left(times_by_subject[key], m.date)
            for t, k, recipients in entries[start:]:
                if t > m.date + window:
                    break
                if k == "forward" or (k == "reply" and m.sender in recipients):
                    acted_at, kind = t, k
                    break
        rows.append(
            {
                "message_id": m.message_id,
                "acted": acted_at is not None,
                "kind": kind,
                "acted_at": acted_at,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "message_id": pl.Utf8,
            "acted": pl.Boolean,
            "kind": pl.Utf8,
            "acted_at": pl.Datetime("us"),
        },
    )
