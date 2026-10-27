"""Command line: ``actionitems extract`` turns notes into action items, ``actionitems eval`` scores
the extractor against the labelled corpus.

Exit codes: 0 success, 1 a runtime error, 2 bad input or usage.
"""

import argparse
import sys
from pathlib import Path

from actionitems import evaluate
from actionitems.errors import ActionItemsError, InputError
from actionitems.export import to_json, to_markdown, to_table
from actionitems.extract import Options, extract_documents
from actionitems.ics import to_ics
from actionitems.ingest import check_output_path, read_documents, write_atomic
from actionitems.llm import refine
from actionitems.text import is_utf8_safe, one_line, safe_text, strip_controls

MAX_ATTENDEES = 200
MAX_NAME_CHARS = 80
EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2
RENDERERS = {"md": to_markdown, "json": to_json, "ics": to_ics, "table": to_table}


def _attendees(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    names = [strip_controls(n).strip() for n in raw.split(",")]
    names = [n for n in names if n]
    if not all(is_utf8_safe(n) for n in names):
        raise InputError(
            "--attendees contains characters that cannot be encoded as UTF-8"
        )
    if len(names) > MAX_ATTENDEES or any(len(n) > MAX_NAME_CHARS for n in names):
        raise InputError(
            f"at most {MAX_ATTENDEES} attendees, each at most {MAX_NAME_CHARS} characters"
        )
    return tuple(names)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actionitems",
        description="Extract action items (task, owner, due date) from meeting notes and transcripts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser(
        "extract", help="extract action items from files, directories or stdin"
    )
    ex.add_argument(
        "inputs", nargs="*", default=["-"], help="files, directories, or - for stdin"
    )
    ex.add_argument("--format", choices=sorted(RENDERERS), default="md")
    ex.add_argument("--out", help="write the result to this file instead of stdout")
    ex.add_argument(
        "--force", action="store_true", help="overwrite an existing --out file"
    )
    ex.add_argument(
        "--attendees", help="comma-separated attendee names (owner matching)"
    )
    ex.add_argument(
        "--meeting-date", help="meeting date, YYYY-MM-DD or an ISO datetime"
    )
    ex.add_argument(
        "--timezone", default="UTC", help="IANA timezone name (default UTC)"
    )
    ex.add_argument(
        "--min-confidence", type=float, default=0.4, help="drop items below this"
    )
    ex.add_argument("--no-nlp", action="store_true", help="rules only; skip spaCy")
    ex.add_argument(
        "--llm-url", help="llama-server chat endpoint for the refinement pass"
    )
    ex.add_argument(
        "--allow-remote", action="store_true", help="allow a non-loopback LLM host"
    )

    ev = sub.add_parser("eval", help="score the extractor on the labelled corpus")
    ev.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ev.add_argument(
        "--write",
        action="store_true",
        help="write results/metrics.json and report.md (all splits)",
    )
    ev.add_argument("--no-nlp", action="store_true", help="rules only; skip spaCy")
    ev.add_argument(
        "--min-confidence",
        type=float,
        default=0.4,
        help="drop items below this (LLM-added items sit at 0.35)",
    )
    ev.add_argument(
        "--llm-url", help="llama-server chat endpoint for the refinement pass"
    )
    ev.add_argument(
        "--allow-remote", action="store_true", help="allow a non-loopback LLM host"
    )
    return parser


def _extract(args: argparse.Namespace) -> int:
    if not 0.0 <= args.min_confidence <= 1.0:
        raise InputError("--min-confidence must be between 0 and 1")
    options = Options(
        attendees=_attendees(args.attendees),
        meeting_date=args.meeting_date,
        timezone=args.timezone,
        min_confidence=args.min_confidence,
        use_nlp=not args.no_nlp,
    )
    out_path = (
        check_output_path(Path(args.out), args.inputs, args.force) if args.out else None
    )
    documents = read_documents(args.inputs)
    extractions = extract_documents(documents, options)
    if args.llm_url:
        extractions = [
            refine(
                extraction,
                doc.text,
                set(options.attendees),
                args.llm_url,
                allow_remote=args.allow_remote,
                min_confidence=options.min_confidence,
            )
            for doc, extraction in zip(documents, extractions, strict=True)
        ]
    text = RENDERERS[args.format](extractions)
    if out_path is not None:
        write_atomic(out_path, text)
    else:
        sys.stdout.write(text)
    return EXIT_OK


def _eval(args: argparse.Namespace) -> int:
    if args.write and args.split != "all":
        raise InputError("--write needs --split all so that both splits are recorded")
    splits = evaluate.SPLITS if args.split == "all" else (args.split,)
    if not 0.0 <= args.min_confidence <= 1.0:
        raise InputError("--min-confidence must be between 0 and 1")
    base = Options(use_nlp=not args.no_nlp, min_confidence=args.min_confidence)
    metrics, scores = evaluate.evaluate_splits(
        splits, base, llm_url=args.llm_url, allow_remote=args.allow_remote
    )
    sys.stdout.write(evaluate.summary(metrics))
    if args.write:
        metrics_path, report_path = evaluate.write_results(metrics, scores)
        sys.stdout.write(f"wrote {metrics_path} and {report_path}\n")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    args = _parser().parse_args(argv)
    try:
        if args.command == "extract":
            return _extract(args)
        return _eval(args)
    except InputError as exc:
        print(f"actionitems: {safe_text(one_line(str(exc)))}", file=sys.stderr)
        return EXIT_USAGE
    except ActionItemsError as exc:
        print(f"actionitems: {safe_text(one_line(str(exc)))}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
