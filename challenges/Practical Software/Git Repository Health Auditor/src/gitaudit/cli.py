"""Command line entry point.

Exit codes: 0 no finding at or above --fail-on (or --fail-on none); 1 at least one such finding;
2 usage, configuration or git error. Argparse usage errors also exit 2.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from gitaudit import __version__
from gitaudit.audit import AuditResult, run_audit
from gitaudit.baseline import write_baseline
from gitaudit.config import build_settings
from gitaudit.git import GitError, GitRepo
from gitaudit.reporters import render_html, render_json, render_sarif, render_table
from gitaudit.safety import atomic_write, check_output_path, sanitize

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gitaudit",
        description="Read-only health audit of a git repository: branches, blobs, secrets, hygiene.",
    )
    p.add_argument(
        "repo",
        nargs="?",
        default=".",
        help="repository path (default: current directory)",
    )
    p.add_argument("--version", action="version", version=f"gitaudit {__version__}")
    p.add_argument(
        "--config", type=Path, help="config file (gitaudit.toml or pyproject.toml)"
    )
    p.add_argument("--ref", help="commit-ish to audit (default HEAD)")
    p.add_argument(
        "--default-branch", dest="default_branch", help="name of the default branch"
    )
    p.add_argument("--stale-days", dest="stale_days", type=int)
    p.add_argument("--long-lived-days", dest="long_lived_days", type=int)
    p.add_argument("--max-blob-size", dest="max_blob_size", help="e.g. 5MiB")
    p.add_argument("--lfs-threshold", dest="lfs_threshold", help="e.g. 1MiB")
    p.add_argument(
        "--max-scan-size",
        dest="max_scan_size",
        help="per-blob content scan limit, e.g. 2MiB",
    )
    p.add_argument("--top", type=int, help="rows per list in table output")
    p.add_argument(
        "--fail-on",
        dest="fail_on",
        help="lowest severity that fails the run: info, low, medium, high, critical, or none",
    )
    p.add_argument(
        "--no-history",
        dest="history",
        action="store_false",
        default=None,
        help="audit only the tree at --ref, skipping history",
    )
    p.add_argument(
        "--since", help="only history newer than this git date, e.g. '2 years ago'"
    )
    p.add_argument(
        "--check-signatures",
        dest="check_signatures",
        action="store_true",
        default=None,
        help="verify recent commit signatures (runs gpg)",
    )
    p.add_argument(
        "--check-identities",
        dest="check_identities",
        action="store_true",
        default=None,
        help="flag placeholder author emails",
    )
    p.add_argument("--baseline", help="accept findings listed in this baseline file")
    p.add_argument(
        "--write-baseline",
        dest="write_baseline",
        type=Path,
        help="write the current fingerprints to this file and exit 0",
    )
    p.add_argument(
        "--allow-path",
        dest="allow_paths",
        action="append",
        default=None,
        help="glob of paths to skip for file-name and content secret checks (repeatable)",
    )
    p.add_argument(
        "--honor-inline-allow",
        dest="honor_inline_allow",
        action="store_true",
        default=None,
        help="honour 'gitaudit:allow' comments on a line (off by default: a commit must not hide its own secrets)",
    )
    p.add_argument(
        "--allow-partial",
        dest="allow_partial",
        action="store_true",
        default=None,
        help="do not fail the run when a budget or cap stopped the audit early (exit 0 with an INFO HY013)",
    )
    p.add_argument(
        "--time-budget",
        dest="time_budget",
        type=int,
        help="wall-clock seconds for the whole audit; when reached, results are marked partial",
    )
    p.add_argument(
        "--scan-budget",
        dest="scan_budget_bytes",
        help="total blob bytes to content-scan, e.g. 2GiB; when reached, results are marked partial",
    )
    p.add_argument(
        "--allow-regex",
        dest="allow_regexes",
        action="append",
        default=None,
        help="regex; matching secret values are ignored (repeatable)",
    )
    p.add_argument(
        "--format", choices=("table", "json", "sarif", "html"), default="table"
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        help="write json, sarif or html here instead of stdout",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="table only: no per-row output beyond summary",
    )
    return p


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    keys = (
        "ref", "default_branch", "stale_days", "long_lived_days", "max_blob_size",
        "lfs_threshold", "max_scan_size", "top", "fail_on", "history", "since",
        "check_signatures", "check_identities", "baseline", "honor_inline_allow",
        "allow_partial", "time_budget", "scan_budget_bytes",
    )  # fmt: skip
    values: dict[str, object] = {k: getattr(args, k) for k in keys}
    values["allow_paths"] = args.allow_paths
    values["allow_regexes"] = args.allow_regexes
    return values


def _render(
    result: AuditResult,
    args: argparse.Namespace,
    console: Console,
    forbidden: Path,
) -> None:
    if args.format == "table":
        if args.output is not None:
            raise ValueError(
                "--output is for json, sarif and html; table goes to the terminal"
            )
        render_table(result, console, limit=0 if args.quiet else None)
        return
    text = {"json": render_json, "sarif": render_sarif, "html": render_html}[
        args.format
    ](result)
    if args.output is None:
        # JSON and SARIF are machine-read: always UTF-8, whatever the console code page is.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stdout.write(text)
        return
    target = check_output_path(args.output, forbidden_root=forbidden)
    atomic_write(target, text)
    print(f"wrote {sanitize(target, 200)}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    console = Console(stderr=False, highlight=False)
    err = Console(stderr=True, highlight=False)
    try:
        repo = Path(args.repo)
        settings = build_settings(repo, args.config, Path.cwd(), _overrides(args))
        forbidden = GitRepo(repo).git_dir  # reports are never written inside .git
        result = run_audit(settings)
        if args.write_baseline is not None:
            count = write_baseline(args.write_baseline, result.findings, forbidden)
            err.print(
                f"wrote {count} fingerprints to {sanitize(args.write_baseline, 200)}",
                markup=False,
                highlight=False,
            )
            return EXIT_OK
        _render(result, args, console, forbidden)
        threshold = settings.fail_threshold()
        # A partial audit fails open in CI unless the operator explicitly allowed it.
        partial_fails = result.partial and not settings.allow_partial
        return EXIT_FINDINGS if partial_fails or result.failing(threshold) else EXIT_OK
    except (GitError, ValueError, OSError) as exc:
        err.print(
            f"gitaudit: error: {sanitize(exc, 400)}", markup=False, highlight=False
        )
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 an unexpected bug must still exit 2, not 1 or a traceback
        err.print(
            f"gitaudit: internal error: {type(exc).__name__}: {sanitize(exc, 300)}",
            markup=False,
            highlight=False,
        )
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
