"""Comparing one field of one matched pair, and saying *how* the two values differ.

Every comparison lands in exactly one category:

=================  ==========================================================================
``match``          identical as written
``format_only``    different as written, same after normalizing (case, spaces, accents, punctuation)
``within_tolerance``  numbers or coordinates that differ by no more than the configured tolerance
``near_match``     text that is not the same but is clearly the same thing ("Heathrow" in "London Heathrow Airport")
``mismatch``       a real disagreement
``left_missing``   only the right source has a value
``right_missing``  only the left source has a value
``both_missing``   neither has a value (nothing to reconcile)
``invalid``        a value that cannot be read as the type being compared (text in a number, latitude 200)
``unverifiable``   crosswalk only: too few pairs with this left value to say what it should map to
=================  ==========================================================================

The *differences* (what a person needs to look at) are ``mismatch``, ``near_match``, ``invalid`` and the
two ``*_missing`` categories. The others are agreement of one kind or another.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import polars as pl
from rapidfuzz import fuzz

from data_reconciler.config import Field

MATCH, FORMAT_ONLY, WITHIN, NEAR, MISMATCH = (
    "match",
    "format_only",
    "within_tolerance",
    "near_match",
    "mismatch",
)
LEFT_MISSING, RIGHT_MISSING, BOTH_MISSING = (
    "left_missing",
    "right_missing",
    "both_missing",
)
INVALID, UNVERIFIABLE = "invalid", "unverifiable"
CATEGORIES = (
    MISMATCH,
    INVALID,
    LEFT_MISSING,
    RIGHT_MISSING,
    NEAR,
    WITHIN,
    FORMAT_ONLY,
    UNVERIFIABLE,
    BOTH_MISSING,
    MATCH,
)
DIFFERENCES = frozenset({MISMATCH, INVALID, LEFT_MISSING, RIGHT_MISSING, NEAR})
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Result:
    category: str
    metric: float | None = (
        None  # geo: km apart; number: absolute difference; text: similarity 0 to 1
    )
    note: str | None = None


def is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _missing_result(left, right) -> Result | None:
    lm, rm = is_missing(left), is_missing(right)
    if lm and rm:
        return Result(BOTH_MISSING)
    if lm:
        return Result(LEFT_MISSING)
    if rm:
        return Result(RIGHT_MISSING)
    return None


def normalize_text(value: str) -> str:
    """Case-folded, accent-stripped, punctuation-free, single-spaced."""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", stripped.casefold()).strip()


def compare_text(left: str | None, right: str | None, near: float) -> Result:
    if (r := _missing_result(left, right)) is not None:
        return r
    if left == right:
        return Result(MATCH)
    a, b = normalize_text(left), normalize_text(right)
    if a == b:
        return Result(FORMAT_ONLY)
    similarity = fuzz.token_sort_ratio(a, b) / 100
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return Result(NEAR, similarity, "one name contains the other")
    if similarity >= near:
        return Result(NEAR, similarity)
    return Result(MISMATCH, similarity)


def compare_code(left: str | None, right: str | None) -> Result:
    if (r := _missing_result(left, right)) is not None:
        return r
    if left == right:
        return Result(MATCH)
    if left.strip().upper() == right.strip().upper():
        return Result(FORMAT_ONLY)
    return Result(MISMATCH)


def parse_number(value: str | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compare_number(
    left: str | None, right: str | None, abs_tol: float, rel_tol: float
) -> Result:
    if (r := _missing_result(left, right)) is not None:
        return r
    a, b = parse_number(left), parse_number(right)
    if a is None or b is None:
        bad = "left" if a is None else "right"
        return Result(
            INVALID,
            note=f"{bad} value {(left if a is None else right)!r} is not a number",
        )
    diff = abs(a - b)
    if diff == 0:
        return Result(MATCH, 0.0)
    if diff <= abs_tol or diff <= rel_tol * max(abs(a), abs(b)):
        return Result(WITHIN, diff)
    return Result(MISMATCH, diff)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def _coordinates(pair) -> tuple[float, float] | str | None:
    """``(lat, lon)``, ``None`` if either part is absent, or a message if present but unusable."""
    lat_s, lon_s = pair
    if is_missing(lat_s) or is_missing(lon_s):
        return None
    lat, lon = parse_number(lat_s), parse_number(lon_s)
    if lat is None or lon is None:
        return f"({lat_s}, {lon_s}) is not numeric"
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return f"({lat_s}, {lon_s}) is outside the valid range"
    return lat, lon


def compare_geo(left, right, tolerance_km: float) -> Result:
    a, b = _coordinates(left), _coordinates(right)
    for side, c in (("left", a), ("right", b)):
        if isinstance(c, str):
            return Result(INVALID, note=f"{side} {c}")
    if a is None and b is None:
        return Result(BOTH_MISSING)
    if a is None:
        return Result(LEFT_MISSING)
    if b is None:
        return Result(RIGHT_MISSING)
    if a == b:
        return Result(MATCH, 0.0)
    distance = haversine_km(*a, *b)
    return Result(WITHIN if distance <= tolerance_km else MISMATCH, distance)


@dataclass(frozen=True)
class CrosswalkEntry:
    left_value: str
    right_value: str  # the one right value most pairs map to
    pairs: int
    share: float  # of those pairs, the share that agree on right_value
    trusted: bool


def learn_crosswalk(
    pairs: list[tuple[str, str]], min_support: int, min_share: float
) -> dict[str, CrosswalkEntry]:
    """For each left value, the right value it most often sits next to across all matched pairs.

    A value is *trusted* once it has ``min_support`` pairs and ``min_share`` of them agree. The
    vote includes every pair, the one being judged too, so it is a majority rule: it finds
    the odd one out among many, and cannot judge a value it has seen only once or twice.
    """
    votes: dict[str, Counter[str]] = {}
    for left, right in pairs:
        votes.setdefault(left, Counter())[right] += 1
    table = {}
    for left, counter in votes.items():
        total = sum(counter.values())
        best, count = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        table[left] = CrosswalkEntry(
            left,
            best,
            total,
            count / total,
            total >= min_support and count / total >= min_share,
        )
    return table


def compare_crosswalk(
    lefts: list, rights: list, spec: Field
) -> tuple[list[Result], list[CrosswalkEntry]]:
    clean = [
        (
            None if is_missing(a) else a.strip().casefold(),
            None if is_missing(b) else b.strip().casefold(),
        )
        for a, b in zip(lefts, rights, strict=True)
    ]
    table = learn_crosswalk(
        [(a, b) for a, b in clean if a is not None and b is not None],
        spec.min_support,
        spec.min_share,
    )
    results = []
    for a, b in clean:
        if (r := _missing_result(a, b)) is not None:
            results.append(r)
            continue
        entry = table[a]
        if not entry.trusted:
            results.append(
                Result(UNVERIFIABLE, note=f"only {entry.pairs} pair(s) with {a!r}")
            )
        elif b == entry.right_value:
            results.append(Result(MATCH, note=None))
        else:
            results.append(
                Result(
                    MISMATCH,
                    None,
                    f"{a!r} maps to {entry.right_value!r} in {entry.share:.0%} of {entry.pairs} pairs",
                )
            )
    return results, sorted(table.values(), key=lambda e: (-e.pairs, e.left_value))


def compare_field(
    spec: Field, lefts: list, rights: list
) -> tuple[list[Result], list[CrosswalkEntry]]:
    """Results for each pair of values (geo values are ``(lat, lon)`` tuples), plus the learned crosswalk if any."""
    kind = spec.compare
    if kind == "crosswalk":
        return compare_crosswalk(lefts, rights, spec)
    if kind == "text":
        out = [
            compare_text(a, b, spec.near) for a, b in zip(lefts, rights, strict=True)
        ]
    elif kind == "code":
        out = [compare_code(a, b) for a, b in zip(lefts, rights, strict=True)]
    elif kind == "number":
        out = [
            compare_number(a, b, spec.abs_tolerance, spec.rel_tolerance)
            for a, b in zip(lefts, rights, strict=True)
        ]
    elif kind == "geo":
        out = [
            compare_geo(a, b, spec.tolerance_km)
            for a, b in zip(lefts, rights, strict=True)
        ]
    else:  # config validation makes this unreachable
        raise ValueError(f"unknown comparator {kind!r}")
    return out, []


def crosswalk_frame(entries: list[CrosswalkEntry]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "left_value": [e.left_value for e in entries],
            "right_value": [e.right_value for e in entries],
            "pairs": [e.pairs for e in entries],
            "share": [e.share for e in entries],
            "trusted": [e.trusted for e in entries],
        },
        schema={
            "left_value": pl.Utf8,
            "right_value": pl.Utf8,
            "pairs": pl.Int64,
            "share": pl.Float64,
            "trusted": pl.Boolean,
        },
    )
