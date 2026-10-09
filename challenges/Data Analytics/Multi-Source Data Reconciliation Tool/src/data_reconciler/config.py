"""The reconciliation job as a TOML file: two sources, the keys to match on, the fields to compare.

Unknown sections, keys and comparator names are errors, so a typo cannot silently turn a check off.
See ``airports.toml`` for a complete, commented example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tomllib

COMPARATORS = ("text", "code", "number", "geo", "crosswalk")
SOURCE_KEYS = {
    "name",
    "path",
    "header",
    "columns",
    "null_values",
    "derive",
    "scope",
    "delimiter",
    "explain",
}
KEY_KEYS = {"left", "right"}
FIELD_KEYS = {
    "name",
    "left",
    "right",
    "compare",
    "near",
    "abs_tolerance",
    "rel_tolerance",
    "tolerance_km",
    "min_support",
    "min_share",
}
TOP_KEYS = {"left", "right", "keys", "fields", "group_by", "identity", "identity_min"}


class ConfigError(ValueError):
    """The configuration is not usable; the message says which part and why."""


@dataclass(frozen=True)
class Source:
    name: str
    path: str  # ``{data}`` is replaced by the data folder (full data when fetched, else the sample)
    header: bool = True
    columns: tuple[str, ...] = ()  # names for a headerless file
    null_values: tuple[str, ...] = ("",)
    delimiter: str = ","
    derive: dict[str, str] = field(
        default_factory=dict
    )  # new column -> Polars SQL expression
    scope: str | None = (
        None  # Polars SQL expression: rows that fail it are not part of the job
    )
    explain: tuple[
        str, ...
    ] = ()  # columns to show for unmatched rows of the other side found out of scope


@dataclass(frozen=True)
class Key:
    left: str
    right: str


@dataclass(frozen=True)
class Field:
    name: str
    left: tuple[str, ...]
    right: tuple[str, ...]
    compare: str
    near: float = 0.85  # text: similarity (0 to 1) at which a difference is only "near"
    abs_tolerance: float = 0.0  # number: differences up to this are within tolerance
    rel_tolerance: float = 0.0  # number: ... or up to this fraction of the larger value
    tolerance_km: float = 0.5  # geo: distances up to this are within tolerance
    min_support: int = (
        5  # crosswalk: pairs a left value needs before its mapping is trusted
    )
    min_share: float = (
        0.6  # crosswalk: share of those pairs that must agree on one right value
    )


@dataclass(frozen=True)
class Config:
    left: Source
    right: Source
    keys: tuple[Key, ...]
    fields: tuple[Field, ...]
    group_by: str | None = (
        None  # a left column: per-group difference rates in the report
    )
    identity: tuple[str, ...] = ()  # fields that together say "same real-world thing"
    identity_min: int = (
        2  # a pair is a different entity when this many identity fields mismatch
    )


def _unknown(section: str, got: set[str], allowed: set[str]) -> None:
    extra = got - allowed
    if extra:
        raise ConfigError(
            f"[{section}] has unknown key(s) {sorted(extra)}; allowed: {sorted(allowed)}"
        )


def _source(raw: dict, section: str) -> Source:
    _unknown(section, set(raw), SOURCE_KEYS)
    for required in ("name", "path"):
        if required not in raw:
            raise ConfigError(f"[{section}] needs `{required}`")
    header = raw.get("header", True)
    columns = tuple(raw.get("columns", ()))
    if not header and not columns:
        raise ConfigError(
            f"[{section}] has header = false, so it needs `columns` to name them"
        )
    return Source(
        name=raw["name"],
        path=raw["path"],
        header=header,
        columns=columns,
        null_values=tuple(raw.get("null_values", [""])),
        delimiter=raw.get("delimiter", ","),
        derive=dict(raw.get("derive", {})),
        scope=raw.get("scope"),
        explain=tuple(raw.get("explain", ())),
    )


def _cols(value, where: str) -> tuple[str, ...]:
    cols = (value,) if isinstance(value, str) else tuple(value)
    if not cols or not all(isinstance(c, str) for c in cols):
        raise ConfigError(f"{where} must be a column name or a list of column names")
    return cols


def _field(raw: dict, index: int) -> Field:
    where = f"[[fields]] #{index + 1}"
    _unknown(where, set(raw), FIELD_KEYS)
    for required in ("name", "left", "right", "compare"):
        if required not in raw:
            raise ConfigError(f"{where} needs `{required}`")
    if raw["compare"] not in COMPARATORS:
        raise ConfigError(
            f"{where}: compare must be one of {COMPARATORS}, not {raw['compare']!r}"
        )
    left, right = (
        _cols(raw["left"], f"{where} left"),
        _cols(raw["right"], f"{where} right"),
    )
    width = 2 if raw["compare"] == "geo" else 1
    if len(left) != width or len(right) != width:
        raise ConfigError(
            f"{where}: a {raw['compare']} comparison takes {width} column(s) per side"
            + (" (latitude, longitude)" if width == 2 else "")
        )
    params = {
        k: raw[k] for k in FIELD_KEYS - {"name", "left", "right", "compare"} if k in raw
    }
    if not 0 <= params.get("near", 0.85) <= 1:
        raise ConfigError(f"{where}: near must be between 0 and 1")
    if not 0 < params.get("min_share", 0.6) <= 1:
        raise ConfigError(f"{where}: min_share must be in (0, 1]")
    if any(
        params.get(k, 0) < 0 for k in ("abs_tolerance", "rel_tolerance", "tolerance_km")
    ):
        raise ConfigError(f"{where}: tolerances cannot be negative")
    return Field(raw["name"], left, right, raw["compare"], **params)


def parse(raw: dict) -> Config:
    _unknown("config", set(raw), TOP_KEYS)
    for required in ("left", "right", "keys", "fields"):
        if required not in raw:
            raise ConfigError(f"the config needs a `{required}` section")
    keys = []
    for i, k in enumerate(raw["keys"]):
        _unknown(f"[[keys]] #{i + 1}", set(k), KEY_KEYS)
        if set(k) != KEY_KEYS:
            raise ConfigError(f"[[keys]] #{i + 1} needs both `left` and `right`")
        keys.append(Key(k["left"], k["right"]))
    if not keys:
        raise ConfigError("at least one [[keys]] entry is needed")
    fields = tuple(_field(f, i) for i, f in enumerate(raw["fields"]))
    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise ConfigError("field names must be unique")
    identity = tuple(raw.get("identity", ()))
    unknown = [n for n in identity if n not in names]
    if unknown:
        raise ConfigError(
            f"identity names unknown field(s) {unknown}; fields are {names}"
        )
    identity_min = raw.get("identity_min", 2)
    if identity and not 1 <= identity_min <= len(identity):
        raise ConfigError(f"identity_min must be between 1 and {len(identity)}")
    return Config(
        _source(raw["left"], "left"),
        _source(raw["right"], "right"),
        tuple(keys),
        fields,
        raw.get("group_by"),
        identity,
        identity_min,
    )


def load(path: Path) -> Config:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid TOML: {exc}") from exc
    return parse(raw)
