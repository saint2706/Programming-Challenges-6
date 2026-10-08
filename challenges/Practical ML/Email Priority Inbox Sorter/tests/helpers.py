"""Test code shared by several test modules (moved out of test files that used to import each other)."""

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import numpy as np
import polars as pl
from inbox_sorter.features import FEATURE_COLUMNS


def synthetic_csv(tmp_path, n=360):
    """Two mailboxes. The owner nearly always answers `boss`, rarely anyone else."""
    rng = np.random.default_rng(0)
    rows = []
    for user, owner in (("aa-b", "aa.b@enron.com"), ("cc-d", "cc.d@enron.com")):
        k = 0
        for i in range(n):
            when = BASE + timedelta(hours=7 * i)
            sender = ["boss@enron.com", "peer@enron.com", "news@spam.com"][i % 3]
            subject = f"{SECRET} {user} {i}"
            body = (
                "urgent please review? "
                if sender == "boss@enron.com"
                else "newsletter "
            ) * 3
            rows.append(
                (
                    f"{user}/inbox/{k}.",
                    _raw(f"{user}-r{i}", when, sender, owner, subject, body),
                )
            )
            k += 1
            p = 0.9 if sender == "boss@enron.com" else 0.08
            if rng.random() < p:
                rows.append(
                    (
                        f"{user}/_sent_mail/{k}.",
                        _raw(
                            f"{user}-s{i}",
                            when + timedelta(hours=2),
                            owner,
                            sender,
                            f"Re: {subject}",
                            "ok",
                        ),
                    )
                )
                k += 1
        # keep the owner active to the end so nothing but the final window is censored
        for j in range(5):
            when = BASE + timedelta(hours=7 * n + 24 * j)
            rows.append(
                (
                    f"{user}/_sent_mail/{k}.",
                    _raw(
                        f"{user}-x{j}", when, owner, "someone@x.com", f"misc {j}", "hi"
                    ),
                )
            )
            k += 1
    p = tmp_path / "emails.csv"
    pl.DataFrame(rows, schema=["file", "message"], orient="row").write_csv(p)
    return p


BASE = datetime(2000, 3, 1, tzinfo=UTC)


SECRET = "do-not-leak-this-subject-xyz"


def _raw(mid, when, sender, to, subject, body):
    return (
        f"Message-ID: <{mid}>\nDate: {format_datetime(when)}\nFrom: {sender}\n"
        f"To: {to}\nSubject: {subject}\n\n{body}\n"
    )


def synthetic(n_per_box=900, seed=0, boxes=("m1", "m2")):
    """A task with a planted metadata signal and a planted text signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for box in boxes:
        for i in range(n_per_box):
            meta = rng.random()
            urgent = rng.random() < 0.3
            p = 0.04 + 0.5 * meta**3 + 0.35 * urgent
            acted = bool(rng.random() < min(p, 0.95))
            feats = {c: 0.0 for c in FEATURE_COLUMNS}
            feats["sender_prior_acted_rate"] = meta
            feats["n_to"] = float(rng.integers(1, 5))
            feats["owner_in_to"] = float(rng.random() < 0.5)
            words = " ".join(
                rng.choice(["lunch", "report", "fyi", "draft", "meeting"], 6)
            )
            text = ("urgent deadline " if urgent else "casual note ") + words
            rows.append(
                {
                    "message_id": f"{box}-{i}",
                    "mailbox": box,
                    "date": T0 + timedelta(hours=i),
                    "subject": text.split(" ")[0],
                    "body": text,
                    "acted": acted,
                    **feats,
                }
            )
    return pl.DataFrame(rows, schema_overrides={"date": pl.Datetime("us")})


T0 = datetime.fromisoformat("2001-01-01T00:00")
