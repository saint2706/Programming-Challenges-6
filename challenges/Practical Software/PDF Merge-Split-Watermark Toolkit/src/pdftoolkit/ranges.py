"""Page-range specs: 1-based, inclusive, comma separated.

``3``  ``2-5``  ``7-`` (to the end)  ``5-2`` (descending)  ``end``  ``odd``  ``even``  ``all``.
Parsing returns 0-based indices in the order written, duplicates kept, so one spec serves
selection, reordering and splitting.
"""

import re

from pdftoolkit.errors import PdfToolError

_NUMBER = re.compile(r"(\d+|end|last)", re.IGNORECASE)


def _number(text: str, total: int, token: str) -> int:
    text = text.strip()
    if not _NUMBER.fullmatch(text):
        raise PdfToolError(f"bad page range {token!r}")
    n = total if text.lower() in ("end", "last") else int(text)
    if not 1 <= n <= total:
        raise PdfToolError(
            f"page {n} out of range in {token!r} (document has {total} pages)"
        )
    return n


def _expand(token: str, total: int) -> list[int]:
    word = token.strip().lower()
    if word == "all":
        return list(range(total))
    if word == "odd":
        return list(range(0, total, 2))
    if word == "even":
        return list(range(1, total, 2))
    if "-" not in word:
        return [_number(word, total, token) - 1]
    start_text, _, stop_text = word.partition("-")
    start = _number(start_text, total, token)
    stop = _number(stop_text, total, token) if stop_text.strip() else total
    step = 1 if start <= stop else -1
    return list(range(start - 1, stop - 1 + step, step))


def parse_groups(spec: str, total: int) -> list[list[int]]:
    """One list of 0-based indices per comma-separated token."""
    tokens = [t for t in spec.split(",") if t.strip()]
    if not tokens:
        raise PdfToolError("empty page range")
    return [_expand(t, total) for t in tokens]


def parse_pages(spec: str, total: int) -> list[int]:
    return [i for group in parse_groups(spec, total) for i in group]
