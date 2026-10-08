"""Parsing the Enron corpus into messages, mailbox owners and received mail.

Source: Kaggle ``wcukierski/enron-email-dataset`` (``emails.csv``, one raw RFC-822
message per row, ``file`` = ``user/folder/N.``). Downloaded once into ``data/``
(gitignored): the corpus is real people's mail, so nothing from it is committed.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email import message_from_string
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

import polars as pl

from inbox_sorter.paths import project_root

HERE = project_root()
DATA_DIR = HERE / "data"
CSV_NAME = "emails.csv"
DATASET = "wcukierski/enron-email-dataset"

# The corpus has messages stamped 1979, 2044 and similar (bad clocks); the real
# mail is 1998-2002.
MIN_YEAR, MAX_YEAR = 1998, 2002
# Folder names (lowercased) that hold mail the owner sent.
SENT_FOLDERS = frozenset({"_sent_mail", "sent", "sent_items"})

_FOLD = re.compile(r"\s*\n\s*")


@dataclass(frozen=True)
class Message:
    message_id: str
    date: datetime | None  # naive UTC; None when the Date header is missing/garbled
    sender: str
    to: list[str]
    cc: list[str]
    subject: str
    body: str


@dataclass
class MailboxData:
    owner: str
    sent: list[Message]
    others: list[Message] = field(default_factory=list)


def _header(msg, name: str) -> str:
    value = msg.get(name)
    return _FOLD.sub(" ", str(value)).strip() if value is not None else ""


def _addresses(value: str) -> list[str]:
    return [a.lower() for _, a in getaddresses([value]) if a]


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def parse_message(raw: str) -> Message:
    """Parse one raw message. Never raises on odd headers; bad fields become empty/None."""
    msg = message_from_string(raw)
    sender = (_addresses(_header(msg, "From")) or [""])[0]
    subject = _header(msg, "Subject")
    body = msg.get_payload() if not msg.is_multipart() else ""
    body = body if isinstance(body, str) else ""
    date = _parse_date(_header(msg, "Date"))
    message_id = _header(msg, "Message-ID")
    if not message_id:
        # stable fallback so the same message in two folders still dedupes
        digest = hashlib.sha1(
            f"{sender}|{date}|{subject}|{body[:200]}".encode("utf-8", "replace")
        ).hexdigest()
        message_id = f"<fallback-{digest[:16]}>"
    return Message(
        message_id=message_id,
        date=date,
        sender=sender,
        to=_addresses(_header(msg, "To")),
        cc=_addresses(_header(msg, "Cc")),
        subject=subject,
        body=body,
    )


def owner_address(sent: list[Message]) -> str:
    """The mailbox owner = the most frequent ``From`` in their sent folders.

    A sent folder can hold mail an assistant sent on the owner's behalf, so the
    mode (ties broken alphabetically) is used rather than the first sender.
    """
    counts = Counter(m.sender for m in sent if m.sender)
    if not counts:
        raise ValueError("no sent mail with a sender; cannot infer the owner")
    top = max(counts.values())
    return min(a for a, c in counts.items() if c == top)


def _fingerprint(m: Message) -> tuple:
    return (
        m.sender,
        m.date,
        m.subject,
        tuple(sorted(m.to)),
        tuple(sorted(m.cc)),
        m.body,
    )


def dedupe(msgs: list[Message]) -> list[Message]:
    """Keep the first copy of each message, in order.

    Copies are matched by Message-ID *and* by content (sender, date, subject,
    recipients, body): the corpus gives each folder's copy of one mail a fresh
    Message-ID, so ID alone leaves about half the rows as duplicates.
    """
    seen_ids: set[str] = set()
    seen_content: set[tuple] = set()
    out: list[Message] = []
    for m in msgs:
        fp = _fingerprint(m)
        if m.message_id in seen_ids or fp in seen_content:
            continue
        seen_ids.add(m.message_id)
        seen_content.add(fp)
        out.append(m)
    return out


def received(msgs: list[Message], owner: str) -> list[Message]:
    """Mail the owner received, once each, in the corpus's real date range, oldest first."""
    kept = [
        m
        for m in msgs
        if m.sender != owner
        and owner in m.to + m.cc
        and m.date is not None
        and MIN_YEAR <= m.date.year <= MAX_YEAR
    ]
    return sorted(dedupe(kept), key=lambda m: (m.date, m.message_id))


def load_mailboxes(csv_path: Path, users: list[str]) -> dict[str, MailboxData]:
    """Parse the given users' mail and split sent folders from everything else."""
    prefixes = [f"{u}/" for u in users]
    rows = (
        pl.scan_csv(csv_path)
        .filter(
            pl.col("file").str.contains(
                r"^(?:" + "|".join(map(re.escape, prefixes)) + ")"
            )
        )
        .collect()
    )
    sent: dict[str, list[Message]] = {u: [] for u in users}
    others: dict[str, list[Message]] = {u: [] for u in users}
    for file, raw in zip(
        rows["file"].to_list(), rows["message"].to_list(), strict=True
    ):
        user, folder = file.split("/")[:2]
        target = sent if folder.lower() in SENT_FOLDERS else others
        target[user].append(parse_message(raw))
    return {
        u: MailboxData(owner=owner_address(sent[u]), sent=sent[u], others=others[u])
        for u in users
        if sent[u]
    }


def _unzip_if_needed(path: Path) -> None:
    if path.exists() and zipfile.is_zipfile(path):
        tmp = path.with_suffix(path.suffix + ".zip")
        path.rename(tmp)
        with zipfile.ZipFile(tmp) as z:
            z.extract(z.namelist()[0], path.parent)
        tmp.unlink()


def fetch(data_dir: Path = DATA_DIR) -> None:
    """Download ``emails.csv`` with the Kaggle API (token in ``~/.kaggle/kaggle.json``)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / CSV_NAME
    if target.exists():
        return
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(
        DATASET, CSV_NAME, path=str(data_dir), force=False, quiet=False
    )
    _unzip_if_needed(target)
