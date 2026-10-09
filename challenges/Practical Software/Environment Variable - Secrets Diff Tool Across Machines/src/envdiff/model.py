"""What the tool keeps of a source after loading it: fingerprints and flags, never values."""

from dataclasses import dataclass, field

from envdiff.fingerprint import (
    Fingerprints,
    Hasher,
    classify_shape,
    is_secret_name,
    looks_like_placeholder,
)
from envdiff.parse import Issue, Parsed


@dataclass(frozen=True)
class Entry:
    fingerprints: Fingerprints | None  # None: declared without a value (pass-through)
    empty: bool
    placeholder: bool
    edge_whitespace: bool
    shape: str  # coarse type; "redacted" for secret-looking names
    line: int = 0

    @property
    def passthrough(self) -> bool:
        return self.fingerprints is None


@dataclass
class Source:
    label: str
    origin: str  # safe to print: a path, ``host:path`` or ``<stdin>``; never content
    fmt: str  # dialect used, or "snapshot"
    entries: dict[str, Entry] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    key_id: str | None = (
        None  # set for snapshots: which hashing key made the fingerprints
    )
    revealed: dict[str, str] = field(
        default_factory=dict
    )  # only keys the user asked to see


def build_source(
    label: str,
    origin: str,
    fmt: str,
    parsed: Parsed,
    hasher: Hasher,
    reveal: frozenset[str] = frozenset(),
) -> Source:
    """Fingerprint every value, then let the raw strings go out of scope with ``parsed``."""
    source = Source(
        label, origin, fmt, issues=list(parsed.issues), key_id=hasher.key_id
    )
    for name, value in parsed.values.items():
        line = parsed.lines.get(name, 0)
        if value is None:
            source.entries[name] = Entry(None, False, False, False, "redacted", line)
            continue
        shape = "redacted" if is_secret_name(name) else classify_shape(value)
        source.entries[name] = Entry(
            hasher.fingerprints(name, value),
            empty=value == "",
            placeholder=looks_like_placeholder(value),
            edge_whitespace=value != value.strip(),
            shape=shape,
            line=line,
        )
        if name in reveal:
            source.revealed[name] = value
    return source
