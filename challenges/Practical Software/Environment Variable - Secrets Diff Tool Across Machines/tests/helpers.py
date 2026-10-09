"""Shared builders: load env text into fingerprinted sources without touching the filesystem."""

from envdiff.fingerprint import Hasher
from envdiff.model import Source, build_source
from envdiff.parse import parse

HASHER = Hasher(b"k" * 32)


def src(
    label: str,
    text: str,
    fmt: str = "dotenv",
    hasher: Hasher = HASHER,
    reveal=frozenset(),
) -> Source:
    parsed, used = parse(text, fmt)
    return build_source(label, label, used, parsed, hasher, reveal)
