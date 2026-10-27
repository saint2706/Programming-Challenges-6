"""Settings with precedence: built-in defaults < config file < CLI flags.

The config file is `gitaudit.toml` (top-level `[gitaudit]` table) or `pyproject.toml`
(`[tool.gitaudit]`). It comes from --config, or is discovered in the directory the command runs in
unless that directory is inside the audited worktree. A repository's own committed config is never
used implicitly: a hostile commit must not be able to weaken its own audit.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from gitaudit.findings import Severity
from gitaudit.safety import parse_size, sanitize

MAX_CONFIG_BYTES = 256 * 1024
MAX_REGEX_LEN = 200


@dataclass(frozen=True)
class Settings:
    repo: Path = Path(".")
    ref: str = "HEAD"
    default_branch: str | None = None
    stale_days: int = 90
    long_lived_days: int = 365
    max_blob_size: int = 5 * 1024**2
    lfs_threshold: int = 1024**2
    max_scan_size: int = 2 * 1024**2
    top: int = 20
    fail_on: str = "high"
    history: bool = True
    since: str | None = None
    check_signatures: bool = False
    signature_limit: int = 100
    check_identities: bool = False
    huge_commit_files: int = 1000
    huge_commit_bytes: int = 100 * 1024**2
    max_branches: int = 500
    baseline: str | None = None
    allow_paths: tuple[str, ...] = ()
    allow_regexes: tuple[str, ...] = ()
    stopwords: tuple[str, ...] = ()
    honor_inline_allow: bool = False
    time_budget: int = 600
    scan_budget_bytes: int = 2 * 1024**3
    config_source: str | None = field(default=None, compare=False)

    def fail_threshold(self) -> Severity | None:
        """None means never fail on findings (`--fail-on none`)."""
        if self.fail_on.lower() == "none":
            return None
        return Severity.parse(self.fail_on)


_INT_KEYS = {
    "stale_days": (0, 36500),
    "long_lived_days": (0, 36500),
    "top": (1, 1000),
    "signature_limit": (1, 5000),
    "huge_commit_files": (1, 10_000_000),
    "max_branches": (1, 100_000),
    "time_budget": (1, 86_400),
}
_SIZE_KEYS = (
    "max_blob_size",
    "lfs_threshold",
    "max_scan_size",
    "huge_commit_bytes",
    "scan_budget_bytes",
)
_BOOL_KEYS = (
    "history",
    "check_signatures",
    "check_identities",
    "honor_inline_allow",
)
_STR_KEYS = ("ref", "default_branch", "fail_on", "baseline", "since")
_LIST_KEYS = ("allow_paths", "allow_regexes", "stopwords")
_SECRET_TABLE_KEYS = {"allow_paths", "allow_regexes", "stopwords"}
_SINCE_RE = re.compile(r"[0-9A-Za-z .:+,-]{1,64}")


def _coerce(key: str, value: object) -> object:
    if key in _INT_KEYS:
        low, high = _INT_KEYS[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not low <= value <= high
        ):
            raise ValueError(f"{key} must be an integer from {low} to {high}")
        return value
    if key in _SIZE_KEYS:
        return parse_size(value)
    if key in _BOOL_KEYS:
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        return value
    if key in _STR_KEYS:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if key == "since" and not _SINCE_RE.fullmatch(value):
            raise ValueError("since has characters that are not allowed in a date")
        if key == "fail_on" and value.lower() != "none":
            Severity.parse(value)
        return value
    if key in _LIST_KEYS:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{key} must be a list of strings")
        if key == "allow_regexes":
            for pattern in value:
                if len(pattern) > MAX_REGEX_LEN:
                    raise ValueError(
                        "allow_regexes entries must be at most 200 characters"
                    )
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"invalid allow regex {sanitize(pattern, 60)!r}: {exc}"
                    ) from exc
        return tuple(value)
    raise ValueError(f"unknown setting {sanitize(key, 60)!r}")


def _flatten(table: dict[str, object]) -> dict[str, object]:
    """Accept `[gitaudit.secrets]` as a sub-table; its keys are flattened into the top level."""
    flat: dict[str, object] = {}
    for key, value in table.items():
        if key == "secrets" and isinstance(value, dict):
            unknown = set(value) - _SECRET_TABLE_KEYS
            if unknown:
                raise ValueError(
                    f"unknown [secrets] setting {sanitize(min(unknown), 60)!r}"
                )
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _tool_table(data: dict[str, object]) -> object:
    """The `[tool.gitaudit]` table, or the top-level `[gitaudit]` one. A non-table `tool` is refused."""
    if "tool" not in data:
        return data.get("gitaudit")
    tool = data["tool"]
    if not isinstance(tool, dict):
        raise ValueError("[tool] must be a table")  # noqa: TRY004
    return tool.get("gitaudit")


def parse_config(data: dict[str, object]) -> dict[str, object]:
    table = _tool_table(data)
    if table is None:
        return {}
    if not isinstance(table, dict):
        raise ValueError("gitaudit config must be a table")  # noqa: TRY004
    flat = _flatten(table)
    known = {f.name for f in fields(Settings)} - {"repo", "config_source"}
    values: dict[str, object] = {}
    for key, value in flat.items():
        if key not in known:
            raise ValueError(f"unknown setting {sanitize(key, 60)!r}")
        values[key] = _coerce(key, value)
    return values


def load_config_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"config file not found: {sanitize(path, 200)}")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("config file is too large")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"cannot parse {sanitize(path.name, 80)}: {sanitize(exc, 120)}"
        ) from exc
    return parse_config(data)


def audited_worktree(repo: Path) -> Path:
    """The nearest ancestor-or-self of `repo` that holds `.git`, else `repo` itself.

    Everything under this directory is the audited repository's content, including files the
    repository committed, so none of it may supply configuration.
    """
    current = Path(repo).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def discover_config(cwd: Path, repo: Path) -> Path | None:
    """gitaudit.toml wins over pyproject.toml; neither is searched upwards.

    Discovery is skipped when `cwd` is inside the audited worktree: a repository could otherwise
    commit a permissive config file that the audit then reads. Use --config to pick one explicitly.
    """
    if _inside(cwd, audited_worktree(repo)):
        return None
    toml = cwd / "gitaudit.toml"
    if toml.is_file():
        return toml
    pyproject = cwd / "pyproject.toml"
    if pyproject.is_file() and _has_table(pyproject):
        return pyproject
    return None


def _has_table(path: Path) -> bool:
    try:
        return _tool_table(tomllib.loads(path.read_text(encoding="utf-8"))) is not None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError):
        return False


def build_settings(
    repo: Path,
    config_path: Path | None,
    cwd: Path,
    overrides: dict[str, object],
) -> Settings:
    """Merge defaults, the discovered or explicit config file, then non-None CLI overrides."""
    source = None
    file_values: dict[str, object] = {}
    if config_path is not None:
        file_values = load_config_file(config_path)
        source = str(config_path)
    else:
        found = discover_config(cwd, repo)
        if found is not None:
            file_values = load_config_file(found)
            source = str(found)
    settings = Settings(repo=repo, config_source=source)
    valid = {f.name for f in fields(Settings)}
    merged = {k: v for k, v in file_values.items() if k in valid}
    settings = replace(settings, **merged)
    cli = {k: v for k, v in overrides.items() if v is not None and k in valid}
    for key, value in cli.items():
        # CLI values get the same validation as file values, so `--stale-days -4` is refused.
        settings = replace(settings, **{key: _coerce(key, value)})
    settings.fail_threshold()  # validates fail_on early
    return settings
