"""Deterministic synthetic event log with a *known* weekly retention curve.

Every user signs up in one of 10 consecutive ISO weeks (Mon 2024-01-01 ..
Sun 2024-03-10). In each later week n >= 1 they are independently active with
probability TRUE_RETENTION(n) = 0.60 * 0.80**(n-1) (60% in week 1, 48% in week
2, ...), as long as that week is inside the log window (which ends Sun
2024-03-24, so the newest cohorts are right-censored). Fixed seed => the
committed CSV is reproducible byte for byte.

Run with:  uv run python generate_sample.py
"""

from __future__ import annotations

import csv
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

SEED = 20240101
N_USERS = 400
N_SIGNUP_WEEKS = 10
LOG_WEEKS = 12  # window: 2024-01-01 .. 2024-03-24 inclusive
START = datetime(2024, 1, 1, tzinfo=UTC)
OUTPUT = Path(__file__).parent / "sample_data" / "events.csv"


def true_retention(week_offset: int) -> float:
    return 1.0 if week_offset == 0 else 0.60 * 0.80 ** (week_offset - 1)


def generate() -> list[tuple[str, str, str]]:
    rng = random.Random(SEED)
    rows: list[tuple[str, str, str]] = []
    for u in range(N_USERS):
        user = f"u{u:04d}"
        signup_week = rng.randrange(N_SIGNUP_WEEKS)
        signup = START + timedelta(weeks=signup_week, seconds=rng.randrange(7 * 86400))
        rows.append((user, signup.strftime("%Y-%m-%dT%H:%M:%SZ"), "signup"))
        for n in range(LOG_WEEKS - signup_week):
            if n == 0:
                # a couple of extra same-week sessions, after the signup but clamped so
                # they never spill into week 1 and bias the retention curve
                week_end = (
                    START + timedelta(weeks=signup_week + 1) - timedelta(seconds=1)
                )
                extra = [
                    min(
                        signup + timedelta(seconds=rng.randrange(1, 3 * 86400)),
                        week_end,
                    )
                    for _ in range(rng.randrange(0, 3))
                ]
            elif rng.random() < true_retention(n):
                week_start = START + timedelta(weeks=signup_week + n)
                extra = [
                    week_start + timedelta(seconds=rng.randrange(7 * 86400))
                    for _ in range(rng.randrange(1, 4))
                ]
            else:
                extra = []
            for ts in extra:
                if ts < START + timedelta(weeks=LOG_WEEKS):
                    rows.append((user, ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "session"))
    rows.sort(key=lambda r: (r[1], r[0]))
    return rows


def main() -> None:
    rows = generate()
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["user_id", "timestamp", "event_name"])
        w.writerows(rows)
    print(f"{len(rows)} events, {N_USERS} users -> {OUTPUT}")


if __name__ == "__main__":
    main()
