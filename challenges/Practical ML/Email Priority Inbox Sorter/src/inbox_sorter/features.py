"""Leakage-safe metadata features for a received message.

Every history feature is computed from what was knowable *at receipt time*:
earlier mail from the sender, replies to that sender that had already been sent
(not merely sent before the dump ended), and what the owner had already written.
Counts use sorted event arrays and ``np.searchsorted`` (strictly-before), and the
tests compare them against a brute-force reference and check that perturbing later
events leaves earlier rows untouched.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from inbox_sorter.data import Message
from inbox_sorter.thread import normalize_subject, strip_prefixes

MASS_MAIL_RECIPIENTS = 10
# Enron was in Houston: shift the naive-UTC timestamps by a fixed -6h (CST, DST
# ignored) before reading hour / weekday, so "business hours" means local time.
LOCAL_OFFSET = timedelta(hours=-6)
BUSINESS_START, BUSINESS_END = 8, 18
# Body text beyond this adds nothing to a question-mark count and is usually a
# quoted forward chain.
BODY_SCAN_CHARS = 2000
# A sender with no history gets a mildly pessimistic rate that decays as evidence
# accrues: (acted + PRIOR_RATE * STRENGTH) / (count + STRENGTH).
PRIOR_RATE, PRIOR_STRENGTH = 0.2, 3.0

_ATTACH = re.compile(r"attach", re.IGNORECASE)

FEATURE_GROUPS: dict[str, list[str]] = {
    "recipients": ["n_to", "n_cc", "owner_in_to", "mass_mail", "sender_is_enron"],
    "content": [
        "subject_len",
        "subject_caps",
        "body_len",
        "question_marks",
        "mentions_attachment",
    ],
    "time": ["hour", "weekday", "business_hours"],
    "thread": ["re_depth", "fw_depth", "owner_in_thread"],
    "sender_history": [
        "sender_prior_count",
        "sender_prior_acted",
        "sender_prior_acted_rate",
    ],
    "owner_history": ["owner_prior_sends_to_sender"],
}
FEATURE_COLUMNS = [c for cols in FEATURE_GROUPS.values() for c in cols]


@dataclass
class SentIndex:
    """The owner's sent mail as sorted time arrays, for strictly-before counts."""

    to_times: dict[str, np.ndarray]  # recipient address -> sorted send times
    subject_times: dict[str, np.ndarray]  # normalized subject -> sorted send times


def _us(times) -> np.ndarray:
    return np.array(list(times), dtype="datetime64[us]")


def build_sent_index(sent: list[Message], owner: str) -> SentIndex:
    to_times: dict[str, list[datetime]] = defaultdict(list)
    subject_times: dict[str, list[datetime]] = defaultdict(list)
    for s in sent:
        if s.sender != owner or s.date is None:
            continue
        for addr in set(s.to + s.cc):
            to_times[addr].append(s.date)
        key = normalize_subject(s.subject)
        if key:
            subject_times[key].append(s.date)
    return SentIndex(
        {k: np.sort(_us(v)) for k, v in to_times.items()},
        {k: np.sort(_us(v)) for k, v in subject_times.items()},
    )


def history_counts(query_keys, query_times, event_keys, event_times) -> np.ndarray:
    """For each ``(key, t)`` query: how many events with that key happened before ``t``."""
    events: dict[object, list[datetime]] = defaultdict(list)
    for k, t in zip(event_keys, event_times, strict=True):
        events[k].append(t)
    sorted_events = {k: np.sort(_us(v)) for k, v in events.items()}
    return _count_before(query_keys, query_times, sorted_events)


def _count_before(
    query_keys, query_times, sorted_events: dict[object, np.ndarray]
) -> np.ndarray:
    keys = list(query_keys)
    times = _us(query_times)
    out = np.zeros(len(keys), dtype=np.int64)
    by_key: dict[object, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    for k, idx in by_key.items():
        arr = sorted_events.get(k)
        if arr is not None:
            out[idx] = np.searchsorted(arr, times[idx], side="left")
    return out


def received_frame(
    received: list[Message], labels: pl.DataFrame, mailbox: str, owner: str
) -> pl.DataFrame:
    """Received messages joined with their labels, ready for feature extraction."""
    df = pl.DataFrame(
        {
            "message_id": [m.message_id for m in received],
            "mailbox": [mailbox] * len(received),
            "owner": [owner] * len(received),
            "date": [m.date for m in received],
            "sender": [m.sender for m in received],
            "to": [m.to for m in received],
            "cc": [m.cc for m in received],
            "subject": [m.subject for m in received],
            "body": [m.body for m in received],
        },
        schema={
            "message_id": pl.Utf8,
            "mailbox": pl.Utf8,
            "owner": pl.Utf8,
            "date": pl.Datetime("us"),
            "sender": pl.Utf8,
            "to": pl.List(pl.Utf8),
            "cc": pl.List(pl.Utf8),
            "subject": pl.Utf8,
            "body": pl.Utf8,
        },
    )
    return df.join(labels, on="message_id", how="left").with_columns(
        pl.col("acted").fill_null(False)
    )


def _is_caps(subject: str) -> int:
    letters = [c for c in strip_prefixes(subject)[0] if c.isalpha()]
    return int(len(letters) >= 4 and all(c.isupper() for c in letters))


def metadata_features(df: pl.DataFrame, sent_index: SentIndex) -> pl.DataFrame:
    """Feature table (``message_id`` + ``FEATURE_COLUMNS``) for one mailbox.

    ``df`` is the output of :func:`received_frame`; ``sent_index`` the owner's
    sent mail. Every history feature looks strictly before the row's own date.
    """
    rows = df.to_dicts()
    dates = [r["date"] for r in rows]
    senders = [r["sender"] for r in rows]
    owner_of = [r["owner"] for r in rows]

    # sender history within this mailbox
    sender_prior_count = history_counts(senders, dates, senders, dates)
    acted_pairs = [
        (r["sender"], r["acted_at"]) for r in rows if r["acted_at"] is not None
    ]
    acted_sorted: dict[object, np.ndarray] = {}
    grouped: dict[str, list[datetime]] = defaultdict(list)
    for s, t in acted_pairs:
        grouped[s].append(t)
    acted_sorted = {k: np.sort(_us(v)) for k, v in grouped.items()}
    sender_prior_acted = _count_before(senders, dates, acted_sorted)
    rate = (sender_prior_acted + PRIOR_RATE * PRIOR_STRENGTH) / (
        sender_prior_count + PRIOR_STRENGTH
    )

    # owner history
    owner_prior_sends = _count_before(senders, dates, sent_index.to_times)
    subjects = [normalize_subject(r["subject"]) for r in rows]
    owner_in_thread = _count_before(subjects, dates, sent_index.subject_times) > 0

    out: dict[str, list] = {c: [] for c in FEATURE_COLUMNS}
    for i, r in enumerate(rows):
        n_to, n_cc = len(r["to"]), len(r["cc"])
        local = r["date"] + LOCAL_OFFSET
        _, n_re, n_fw = strip_prefixes(r["subject"])
        body = r["body"] or ""
        weekday = local.weekday()
        values = {
            "n_to": n_to,
            "n_cc": n_cc,
            "owner_in_to": int(owner_of[i] in r["to"]),
            "mass_mail": int(n_to + n_cc >= MASS_MAIL_RECIPIENTS),
            "sender_is_enron": int(senders[i].endswith("@enron.com")),
            "subject_len": len(r["subject"]),
            "subject_caps": _is_caps(r["subject"]),
            "body_len": len(body),
            "question_marks": body[:BODY_SCAN_CHARS].count("?"),
            "mentions_attachment": int(bool(_ATTACH.search(body[:BODY_SCAN_CHARS]))),
            "hour": local.hour,
            "weekday": weekday,
            "business_hours": int(
                weekday < 5 and BUSINESS_START <= local.hour < BUSINESS_END
            ),
            "re_depth": n_re,
            "fw_depth": n_fw,
            "owner_in_thread": int(owner_in_thread[i]),
            "sender_prior_count": int(sender_prior_count[i]),
            "sender_prior_acted": int(sender_prior_acted[i]),
            "sender_prior_acted_rate": float(rate[i]),
            "owner_prior_sends_to_sender": int(owner_prior_sends[i]),
        }
        for c in FEATURE_COLUMNS:
            out[c].append(values[c])
    return pl.DataFrame({"message_id": df["message_id"].to_list(), **out})
