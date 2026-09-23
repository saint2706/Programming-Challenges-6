"""Encryption, persistence, and entry/mood logic for the encrypted diary.

Deliberately has zero dependency on the Textual UI layer (`app.py`) so the
security-critical parts -- key derivation, AEAD encrypt/decrypt, atomic
writes -- are fully unit-testable without driving a terminal app.

## On-disk format

    MAGIC (7 bytes) | VERSION (1 byte) | SALT (16 bytes) | NONCE (12 bytes) | CIPHERTEXT+TAG (rest)

The whole journal (every entry) is serialized to one JSON blob and encrypted
as a single AES-256-GCM ciphertext, re-encrypted in full on every save. At
personal-diary scale (thousands of entries, still a small file) this is
simpler and easier to reason about than per-entry envelope encryption, and
it means there is exactly one nonce/key pairing to get right instead of one
per entry. A fresh random 96-bit nonce is drawn on every save, so nonce
reuse (which is catastrophic for GCM) never happens even though the key
itself is stable for the life of the journal.

The key is derived from the user's passphrase with `scrypt`, not PBKDF2:
scrypt is memory-hard, which meaningfully raises the cost of a GPU/ASIC
brute-force attack on a stolen file in a way an iteration-only KDF like
PBKDF2 does not. Parameters (`SCRYPT_N/R/P`) are tuned for roughly
half-a-second on ordinary hardware -- expensive enough to matter for
brute-forcing, cheap enough that unlocking your own diary isn't annoying.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"EDIARY1"
FORMAT_VERSION = 1
SALT_SIZE = 16
NONCE_SIZE = 12
KEY_SIZE = 32

# scrypt cost parameters: N=2**15 (~32k iterations, memory-hard), r=8, p=1.
# Deliberately higher than the library's interactive-login defaults (N=2**14)
# since this only runs once per unlock, not on every request of a web app.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1


class JournalUnlockError(Exception):
    """Raised when a journal file can't be decrypted.

    Covers both a wrong passphrase and a corrupted/tampered file -- both
    surface as an AEAD authentication failure, and deliberately are not
    distinguished in the exception type or default message, so a caller
    can't be used as an oracle for "is this the right passphrase" without
    also being able to fully decrypt (which needs the passphrase anyway).
    """


class Mood(str, Enum):
    HAPPY = "happy"
    EXCITED = "excited"
    CALM = "calm"
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"
    SAD = "sad"
    ANGRY = "angry"

    @property
    def icon(self) -> str:
        return _MOOD_META[self][0]

    @property
    def color(self) -> str:
        return _MOOD_META[self][1]

    @property
    def score(self) -> int:
        """Valence from 1 (angry) to 7 (happy), used for trend charts.

        Deliberately a 1-7 scale rather than symmetric-around-zero (e.g.
        -3..+3): the mood-trend sparkline fills days with no entries using
        0.0 as a sentinel, and 0 is otherwise not a valid score for any
        mood -- including NEUTRAL -- so a real "no entries that day" gap is
        never visually indistinguishable from a real neutral-mood entry.
        """
        return _MOOD_META[self][2]


_MOOD_META: dict[Mood, tuple[str, str, int]] = {
    Mood.HAPPY: ("😊", "green", 7),
    Mood.EXCITED: ("🤩", "yellow", 6),
    Mood.CALM: ("😌", "cyan", 5),
    Mood.NEUTRAL: ("😐", "white", 4),
    Mood.ANXIOUS: ("😟", "magenta", 3),
    Mood.SAD: ("😢", "blue", 2),
    Mood.ANGRY: ("😠", "red", 1),
}


@dataclass
class Entry:
    id: str
    created_at: datetime
    mood: Mood
    body: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "mood": self.mood.value,
            "body": self.body,
        }

    @staticmethod
    def from_dict(data: dict) -> Entry:
        return Entry(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            mood=Mood(data["mood"]),
            body=data["body"],
        )


@dataclass
class MoodTrendPoint:
    date: str  # ISO date (YYYY-MM-DD)
    average_score: float | None  # None = no entries that day


@dataclass
class MoodTrend:
    points: list[MoodTrendPoint] = field(default_factory=list)
    mood_counts: Counter = field(default_factory=Counter)

    @property
    def sparkline_values(self) -> list[float]:
        """Trend values with gaps filled by 0 (Sparkline can't render None)."""
        return [
            p.average_score if p.average_score is not None else 0.0 for p in self.points
        ]


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=KEY_SIZE, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


def _encrypt(plaintext: bytes, passphrase: str) -> bytes:
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = _derive_key(passphrase, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data=MAGIC)
    return MAGIC + bytes([FORMAT_VERSION]) + salt + nonce + ciphertext


def _decrypt(data: bytes, passphrase: str) -> bytes:
    if len(data) < len(MAGIC) + 1 + SALT_SIZE + NONCE_SIZE or not data.startswith(
        MAGIC
    ):
        raise JournalUnlockError("Not a valid encrypted journal file.")
    offset = len(MAGIC)
    version = data[offset]
    offset += 1
    if version != FORMAT_VERSION:
        raise JournalUnlockError(f"Unsupported journal format version {version}.")
    salt = data[offset : offset + SALT_SIZE]
    offset += SALT_SIZE
    nonce = data[offset : offset + NONCE_SIZE]
    offset += NONCE_SIZE
    ciphertext = data[offset:]
    key = _derive_key(passphrase, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, associated_data=MAGIC)
    except InvalidTag as exc:
        raise JournalUnlockError(
            "Could not unlock the journal -- wrong passphrase, or the file is corrupted/tampered."
        ) from exc


def _atomic_write(path: Path, data: bytes) -> None:
    """Write-to-temp-then-rename so a crash mid-write never corrupts the journal.

    `os.replace` is atomic on both POSIX and Windows -- the journal file at
    `path` is always either the old complete contents or the new complete
    contents, never a partial write, even if the process is killed mid-save.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class JournalStore:
    """An unlocked, in-memory view of a journal, backed by an encrypted file.

    Unlocking a path that doesn't exist yet creates a brand-new empty
    journal that will be encrypted with the given passphrase on first save
    -- there's no separate "create" step, matching how a real diary app's
    first-run flow works ("set a passphrase" and "unlock" are the same
    action the first time).
    """

    def __init__(self, path: str | Path, passphrase: str):
        self.path = Path(path)
        self.passphrase = passphrase
        self._entries: dict[str, Entry] = {}
        if self.path.exists():
            data = self.path.read_bytes()
            plaintext = _decrypt(data, passphrase)
            payload = json.loads(plaintext)
            for raw in payload.get("entries", []):
                entry = Entry.from_dict(raw)
                self._entries[entry.id] = entry

    @property
    def entries(self) -> list[Entry]:
        return sorted(self._entries.values(), key=lambda e: e.created_at, reverse=True)

    def add_entry(
        self, mood: Mood, body: str, created_at: datetime | None = None
    ) -> Entry:
        entry = Entry(
            id=str(uuid4()),
            created_at=created_at or datetime.now(timezone.utc),
            mood=mood,
            body=body,
        )
        self._entries[entry.id] = entry
        self.save()
        return entry

    def delete_entry(self, entry_id: str) -> None:
        del self._entries[entry_id]
        self.save()

    def search(
        self,
        text: str | None = None,
        mood: Mood | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[Entry]:
        results = self.entries
        if text:
            needle = text.lower()
            results = [e for e in results if needle in e.body.lower()]
        if mood is not None:
            results = [e for e in results if e.mood is mood]
        if since is not None:
            results = [e for e in results if e.created_at >= since]
        if until is not None:
            results = [e for e in results if e.created_at <= until]
        return results

    def mood_trend(self, days: int = 30, now: datetime | None = None) -> MoodTrend:
        """Daily average mood score for the trailing `days` days, oldest first."""
        now = now or datetime.now(timezone.utc)
        today = now.date()
        by_day: dict[str, list[int]] = {}
        counts: Counter = Counter()
        window_start = today - timedelta(days=days - 1)
        for entry in self._entries.values():
            day = entry.created_at.astimezone(timezone.utc).date()
            if window_start <= day <= today:
                key = day.isoformat()
                by_day.setdefault(key, []).append(entry.mood.score)
                counts[entry.mood] += 1
        points = []
        for offset in range(days):
            day = window_start + timedelta(days=offset)
            key = day.isoformat()
            scores = by_day.get(key)
            avg = sum(scores) / len(scores) if scores else None
            points.append(MoodTrendPoint(date=key, average_score=avg))
        return MoodTrend(points=points, mood_counts=counts)

    def save(self) -> None:
        payload = {"entries": [e.to_dict() for e in self.entries]}
        plaintext = json.dumps(payload).encode("utf-8")
        ciphertext = _encrypt(plaintext, self.passphrase)
        _atomic_write(self.path, ciphertext)
