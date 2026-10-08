"""Joining tabular data to map regions, with a report of everything that did *not* join.

A choropleth silently lies when keys fail to match: the unmatched rows just vanish, and
the polygons they belonged to look like "no sales".  So every key is resolved through one
normaliser and every failure is kept, with a reason, in a :class:`JoinReport`.

Keys understood for U.S. states: 2-digit FIPS (``1``, ``01``, ``1.0``), USPS abbreviation
(``al``), full name (``Alabama``, ``washington d.c.``).  For counties: 5-digit FIPS
(Excel's dropped leading zero, ``1001``, is repaired) or ``County, State`` names
(``Autauga County, Alabama``, ``autauga, AL``, ``St. Louis county, MO``).
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

_FIPS_RE = re.compile(r"^'?\s*(\d+)(?:\.0+)?\s*$")

# Words dropped from the end of a county name before matching ("Autauga County" -> "autauga").
_COUNTY_SUFFIXES = (
    "city and borough",
    "census area",
    "county",
    "parish",
    "borough",
    "municipality",
    "municipio",
)

# How a county-equivalent's designation is written after its name (Virginia's independent
# cities keep the lower-case "city", as the Census Bureau writes them).
_DISPLAY_LSAD = {
    "county": "County",
    "parish": "Parish",
    "borough": "Borough",
    "census area": "Census Area",
    "city and borough": "City and Borough",
    "municipality": "Municipality",
    "municipio": "Municipio",
    "city": "city",
}

_STATE_ALIASES = {
    "washington dc": "district of columbia",
    "washington d c": "district of columbia",
    "d c": "district of columbia",
    "dc": "district of columbia",
}


def norm_text(text: object) -> str:
    """Casefold, strip accents/punctuation, unify ``Saint``/``St.``, collapse whitespace."""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\bsaint\b", "st", s)
    s = re.sub(r"\bsainte\b", "ste", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_fips(raw: object, width: int) -> str | None:
    """Zero-padded FIPS string, or ``None`` when ``raw`` cannot be a FIPS code of that width.

    Accepts ints, integral floats, and strings like ``"1001"``, ``"01001.0"`` or ``"'01001"``
    (a spreadsheet text-marker).  Numbers longer than ``width`` are rejected, never truncated.
    """
    if raw is None:
        return None
    if isinstance(raw, float):
        if math.isnan(raw) or raw != int(raw):
            return None
        raw = int(raw)
    m = _FIPS_RE.match(str(raw))
    if not m:
        return None
    digits = m.group(1)
    if len(digits) > width:
        return None
    return digits.zfill(width)


def _strip_county_suffix(name: str) -> str:
    for suffix in _COUNTY_SUFFIXES:
        if name == suffix:
            return name
        if name.endswith(" " + suffix):
            return name[: -len(suffix) - 1]
    return name


@dataclass
class Resolution:
    index: int | None
    reason: str = ""  # empty when matched


@dataclass
class RegionIndex:
    """Lookup from the many spellings of a region key to a feature position."""

    level: str  # "state", "county" or "custom"
    fips: list[str]
    names: list[str]
    exact: dict[str, list[int]] = field(
        default_factory=dict
    )  # normalised alias -> feature positions
    fips_width: int | None = None
    state_by_alias: dict[str, str] = field(
        default_factory=dict
    )  # normalised state alias -> state FIPS

    def __len__(self) -> int:
        return len(self.fips)

    @classmethod
    def for_states(cls, features: list[dict]) -> RegionIndex:
        idx = cls("state", [], [], fips_width=2)
        for pos, feat in enumerate(features):
            p = feat["properties"]
            idx.fips.append(p["fips"])
            idx.names.append(p["name"])
            for alias in (p["fips"], p.get("abbr", ""), p["name"]):
                if alias:
                    idx.exact.setdefault(norm_text(alias), []).append(pos)
        idx.state_by_alias = {
            a: features[pos[0]]["properties"]["fips"] for a, pos in idx.exact.items()
        }
        return idx

    @classmethod
    def for_counties(cls, features: list[dict], states: list[dict]) -> RegionIndex:
        state_idx = cls.for_states(states)
        idx = cls(
            "county", [], [], fips_width=5, state_by_alias=state_idx.state_by_alias
        )
        state_name = {s["properties"]["fips"]: s["properties"]["name"] for s in states}
        aliases_of: dict[str, list[str]] = {}
        for alias, fips in state_idx.state_by_alias.items():
            aliases_of.setdefault(fips, []).append(alias)
        for pos, feat in enumerate(features):
            p = feat["properties"]
            idx.fips.append(p["fips"])
            st = state_name.get(p["state_fips"], p["state_fips"])
            lsad = p.get("lsad") or ""
            label = (
                p["name"]
                if lsad in ("", "00") or p["name"].lower().endswith(lsad)
                else f"{p['name']} {_DISPLAY_LSAD.get(lsad, lsad)}"
            )
            idx.names.append(f"{label}, {st}")
            base = _strip_county_suffix(norm_text(p["name"]))
            idx.exact.setdefault(p["fips"], []).append(pos)
            # Virginia, Maryland and Missouri have independent cities sharing a name with a county
            # ("Richmond" city and county): the bare name is then ambiguous, but the name with its
            # designation ("Richmond city", "Richmond County") is not.
            spellings = [base]
            if lsad not in ("", "00") and not base.endswith(" " + lsad):
                spellings.append(f"{base} {lsad}")
            # keyed by every spelling of the state so "Autauga, AL" and "Autauga, Alabama" both hit
            for alias in aliases_of.get(p["state_fips"], []):
                for spelling in spellings:
                    idx.exact.setdefault(f"{spelling}|{alias}", []).append(pos)
        return idx

    @classmethod
    def for_custom(
        cls, features: list[dict], key_prop: str, name_prop: str | None
    ) -> RegionIndex:
        idx = cls("custom", [], [])
        for pos, feat in enumerate(features):
            p = feat["properties"]
            key = str(p[key_prop])
            idx.fips.append(key)
            idx.names.append(str(p[name_prop]) if name_prop and name_prop in p else key)
            idx.exact.setdefault(norm_text(key), []).append(pos)
            if name_prop and name_prop in p:
                idx.exact.setdefault(norm_text(p[name_prop]), []).append(pos)
        return idx

    def resolve(self, raw: object) -> Resolution:
        if (
            raw is None
            or (isinstance(raw, float) and math.isnan(raw))
            or str(raw).strip() == ""
        ):
            return Resolution(None, "blank key")
        if self.level == "custom":
            return self._lookup(norm_text(raw), raw)
        assert self.fips_width is not None
        code = normalize_fips(raw, self.fips_width)
        if code is not None:
            return self._lookup(code, raw)
        if self.level == "state":
            key = _STATE_ALIASES.get(norm_text(raw), norm_text(raw))
            return self._lookup(key, raw)
        return self._resolve_county_name(str(raw))

    def _resolve_county_name(self, raw: str) -> Resolution:
        if "," not in raw:
            return Resolution(
                None, "county name without a state (ambiguous across states)"
            )
        county, _, state = raw.rpartition(",")
        st_key = _STATE_ALIASES.get(norm_text(state), norm_text(state))
        if st_key not in self.state_by_alias:
            return Resolution(None, f"unrecognised state {state.strip()!r}")
        # Try the spelling exactly as given first ("Richmond city" vs "Richmond County"); only if
        # that finds nothing fall back to the bare name, which may honestly be ambiguous.
        exact = self._lookup(f"{norm_text(county)}|{st_key}", raw)
        if exact.index is not None or exact.reason.startswith("ambiguous"):
            return exact
        return self._lookup(f"{_strip_county_suffix(norm_text(county))}|{st_key}", raw)

    def _lookup(self, key: str, raw: object) -> Resolution:
        hits = self.exact.get(key)
        if not hits:
            return Resolution(None, "no region with this key")
        if len(set(hits)) > 1:
            return Resolution(None, "ambiguous: matches several regions")
        return Resolution(hits[0])


@dataclass
class JoinReport:
    """What happened when data keys met map regions (nothing is dropped silently)."""

    level: str
    rows: int = 0
    matched_rows: int = 0
    matched_regions: int = 0
    unparseable_values: int = 0  # value cell present but not a number
    blank_values: int = 0  # value cell empty / null
    duplicate_rows_merged: int = (
        0  # extra rows beyond the first for a region (aggregated)
    )
    unmatched: list[tuple[str, str, int]] = field(
        default_factory=list
    )  # raw key, reason, rows
    regions_without_record: list[str] = field(
        default_factory=list
    )  # region names with no data row
    total_regions: int = 0

    @property
    def unmatched_rows(self) -> int:
        return sum(n for _k, _r, n in self.unmatched)


def resolve_keys(
    keys: list[object], index: RegionIndex
) -> tuple[list[int | None], list[tuple[str, str, int]]]:
    """Resolve each raw key; also return ``(raw, reason, row_count)`` for every key that failed."""
    cache: dict[str, Resolution] = {}
    positions: list[int | None] = []
    failed: dict[tuple[str, str], int] = {}
    for raw in keys:
        token = "" if raw is None else str(raw)
        if token not in cache:
            cache[token] = index.resolve(raw)
        res = cache[token]
        positions.append(res.index)
        if res.index is None:
            failed[(token, res.reason)] = failed.get((token, res.reason), 0) + 1
    return positions, sorted(
        ((k, r, n) for (k, r), n in failed.items()), key=lambda t: (-t[2], t[0])
    )
