"""Keyed fingerprints and value classification: how equality is shown without showing values.

A fingerprint is ``HMAC-SHA256(key, variant || name || value)``:

* Keyed, so a report or snapshot alone cannot be brute-forced against a dictionary of common values.
  A per-run random key makes fingerprints useless outside the run that printed them; snapshots use
  a key derived from a shared passphrase so separate hosts can be compared later.
* Bound to the variable name, so two different variables that happen to hold the same secret do
  not reveal that fact.
* Computed over four variants of the value (exact, trimmed, unquoted, case-folded) so a diff can
  say *why* two values differ ("whitespace only") without anyone seeing either value.
"""

import fnmatch
import hashlib
import hmac
import os
import re
from dataclasses import dataclass

SECRET_NAME = re.compile(
    r"(SECRET|TOKEN|PASSW(OR)?D|PASSPHRASE|API[_-]?KEY|ACCESS[_-]?KEY|PRIVATE|CREDENTIAL|CERT|SIGNING|"
    r"ENCRYPTION|AUTH|DSN|SALT|SESSION|COOKIE|BEARER|WEBHOOK|_KEY$|^KEY$|_PWD$)",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(
    r"^(changeme|change[-_ ]?me|replace[-_ ]?me|todo|tbd|fixme|xxx+|\*{3,}|secret|password|passw0rd|"
    r"example|test|dummy|default|admin|1234(56)*|your[-_ ].*|<.*>|\$\{.*\}|\.{3})$",
    re.IGNORECASE,
)
_KDF_SALT = b"envdiff-v1"
_VARIANTS = ("x", "s", "q", "f")  # exact, stripped, unquoted, case-folded


def is_secret_name(name: str) -> bool:
    return SECRET_NAME.search(name) is not None


def looks_like_placeholder(value: str) -> bool:
    return bool(value.strip()) and PLACEHOLDER.match(value.strip()) is not None


def matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _unquote(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        return stripped[1:-1]
    return stripped


@dataclass(frozen=True)
class Fingerprints:
    exact: str
    stripped: str
    unquoted: str
    folded: str

    def as_list(self) -> list[str]:
        return [self.exact, self.stripped, self.unquoted, self.folded]


class Hasher:
    def __init__(self, key: bytes) -> None:
        self._key = key
        self.key_id = hashlib.sha256(b"envdiff-key-id" + key).hexdigest()[:12]

    @classmethod
    def random(cls) -> "Hasher":
        return cls(os.urandom(32))

    @classmethod
    def from_passphrase(cls, passphrase: str) -> "Hasher":
        """Stretch a shared passphrase so a weak one is still costly to guess offline."""
        key = hashlib.scrypt(
            passphrase.encode(), salt=_KDF_SALT, n=2**14, r=8, p=1, dklen=32
        )
        return cls(key)

    def _mac(self, variant: str, name: str, value: str) -> str:
        message = f"{variant}\0{name}\0{value}".encode()
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def fingerprints(self, name: str, value: str) -> Fingerprints:
        variants = (value, value.strip(), _unquote(value), value.casefold())
        return Fingerprints(
            *(
                self._mac(tag, name, v)
                for tag, v in zip(_VARIANTS, variants, strict=True)
            )
        )


_SHAPES = [
    ("bool", re.compile(r"^(true|false|yes|no|on|off)$", re.IGNORECASE)),
    ("integer", re.compile(r"^-?\d+$")),
    ("number", re.compile(r"^-?\d+\.\d+$")),
    (
        "uuid",
        re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        ),
    ),
    ("jwt", re.compile(r"^eyJ[\w-]+\.[\w-]+\.[\w-]*$")),
    ("url", re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)),
    ("email", re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")),
    ("path", re.compile(r"^(/|\./|\.\./|~/|[A-Za-z]:[\\/])")),
    ("hex", re.compile(r"^[0-9a-fA-F]{16,}$")),
    ("token", re.compile(r"^[A-Za-z0-9+/_-]{20,}={0,2}$")),
]


def classify_shape(value: str) -> str:
    """A coarse type tag. Coarse on purpose: it must say nothing about the content."""
    if not value:
        return "empty"
    for name, pattern in _SHAPES:
        if pattern.match(value):
            return name
    return "text"
