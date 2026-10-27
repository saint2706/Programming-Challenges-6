"""Command line: `voicememo <command>`. Plain argparse, so output has no box drawing to normalise."""

import argparse
import json
import sys
from pathlib import Path

from voicememo.asr import FakeTranscriber, LanguageUnsupportedError, normalize_language
from voicememo.audio import DecodeError
from voicememo.export import clean_text, library_json, render
from voicememo.ingest import (
    FixedEngine,
    IngestResult,
    RoutingEngine,
    ingest_file,
    retranscribe,
)
from voicememo.library import Library, LibraryError
from voicememo.paths import data_dir

AUDIO_SUFFIXES = (
    ".m4a",
    ".mp3",
    ".wav",
    ".ogg",
    ".oga",
    ".opus",
    ".flac",
    ".amr",
    ".3gp",
    ".aac",
    ".caf",
    ".webm",
)


def _say(text: str) -> None:
    sys.stdout.write(clean_text(text) + "\n")


def _open_library() -> Library:
    return Library(data_dir())


def _engine(args: argparse.Namespace):
    """`--backend fake` reads a JSON map {file stem: transcript} from $VOICEMEMO_FAKE_SCRIPT."""
    if getattr(args, "backend", "auto") == "fake":
        import os

        script_path = os.environ.get("VOICEMEMO_FAKE_SCRIPT")
        script = (
            json.loads(Path(script_path).read_text(encoding="utf-8"))
            if script_path
            else {}
        )
        return FixedEngine(FakeTranscriber(script))
    return RoutingEngine(whisper_model=getattr(args, "whisper", "base"))


def _expand(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(
                sorted(
                    p
                    for p in path.iterdir()
                    if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
                )
            )
        else:
            found.append(path)
    return found


def cmd_models(args: argparse.Namespace) -> int:
    from voicememo import models

    if args.action == "status":
        for name, ready in models.status().items():
            _say(f"{name}: {'ready' if ready else 'missing'}")
        return 0
    if args.parakeet:
        models.pull_parakeet()
    for name in args.whisper or []:
        models.pull_whisper(name)
    if args.embedder:
        models.pull_embedder()
    if not (args.parakeet or args.whisper or args.embedder):
        models.pull_parakeet()
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    engine = _engine(args)
    added = 0
    failures = 0
    with _open_library() as library:
        for path in _expand(args.paths):
            try:
                result: IngestResult = ingest_file(
                    library,
                    path,
                    engine,
                    language=args.language,
                    retry_failed=args.retry_failed,
                )
            except (DecodeError, LibraryError, FileNotFoundError) as exc:
                _say(f"{path}: error: {exc}")
                failures += 1
                continue
            if result.status == "added":
                added += 1
            if result.status == "failed":
                failures += 1
            _say(f"{path.name}: {result.status} #{result.memo_id} {result.message}")
        if added:
            _retag(library, _default_embedder())
    return 1 if failures else 0


def _retag(library: Library, embedder) -> str:
    from voicememo.tagging import plan_tags

    plan = plan_tags(library.texts_by_memo(), embedder)
    library.set_auto_tags(plan.assignments)
    return plan.method


def cmd_watch(args: argparse.Namespace) -> int:
    import time

    from voicememo.watch import Watcher

    engine = _engine(args)
    with _open_library() as library:
        watcher = Watcher(
            root=Path(args.directory),
            library=library,
            engine=engine,
            language=args.language,
            settle_seconds=args.settle,
        )
        if args.once:
            # The first sighting only starts each settle window, so wait it out and scan again.
            watcher.scan()
            time.sleep(args.settle)
            done = watcher.scan()
            if done:
                _retag(library, _default_embedder())
            return 0
        try:
            watcher.run(interval=args.interval, sleep=time.sleep)
        except KeyboardInterrupt:
            _say("stopped")
    return 0


def _default_embedder():
    from voicememo import models
    from voicememo.paths import models_dir
    from voicememo.tagging import LsaEmbedder, Model2VecEmbedder

    target = models_dir() / "model2vec"
    if models.is_ready(target):
        return Model2VecEmbedder(target)
    return LsaEmbedder()


def cmd_list(args: argparse.Namespace) -> int:
    with _open_library() as library:
        for memo in library.list_memos(tag=args.tag, limit=args.limit):
            tags = ", ".join(t for t, _k in library.tags(memo.id))
            _say(
                f"#{memo.id}\t{memo.status}\t{memo.duration:.0f}s\t{memo.title}\t[{tags}]"
            )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    with _open_library() as library:
        try:
            memo = library.get(args.id)
        except LibraryError as exc:
            _say(str(exc))
            return 1
        _say(f"#{memo.id} {memo.title}")
        _say(
            f"status={memo.status} duration={memo.duration:.1f}s language={memo.language or 'unknown'}"
        )
        if memo.error:
            _say(f"error: {memo.error}")
        _say(
            "tags: "
            + (", ".join(f"{t} ({k})" for t, k in library.tags(memo.id)) or "none")
        )
        for seg in library.segments(memo.id):
            _say(f"[{seg['start']:8.2f}] {seg['text']}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    with _open_library() as library:
        hits = library.search(args.query, limit=args.limit)
        for hit in hits:
            _say(f"#{hit.memo_id} {hit.title} @{hit.start:.1f}s: {hit.snippet}")
        if not hits:
            _say("no matches")
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    with _open_library() as library:
        try:
            if args.action == "add":
                _say(f"tagged #{args.id}: {library.add_tag(args.id, args.tag)}")
            else:
                removed = library.remove_tag(args.id, args.tag)
                _say("removed" if removed else "no such tag")
        except LibraryError as exc:
            _say(str(exc))
            return 1
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    with _open_library() as library:
        try:
            library.delete(args.id)
        except LibraryError as exc:
            _say(str(exc))
            return 1
    _say(f"removed #{args.id}")
    return 0


def cmd_retranscribe(args: argparse.Namespace) -> int:
    with _open_library() as library:
        try:
            result = retranscribe(
                library, args.id, _engine(args), normalize_language(args.language)
            )
        except (LibraryError, DecodeError, LanguageUnsupportedError) as exc:
            _say(str(exc))
            return 1
        _say(result.message)
    return 0


def cmd_retag(args: argparse.Namespace) -> int:
    from voicememo import models
    from voicememo.paths import models_dir
    from voicememo.tagging import LsaEmbedder, Model2VecEmbedder

    with _open_library() as library:
        if args.embedder == "model2vec":
            target = models_dir() / "model2vec"
            if not models.is_ready(target):
                _say("model2vec is not pulled; run `voicememo models pull --embedder`")
                return 1
            embedder = Model2VecEmbedder(target)
        else:
            embedder = LsaEmbedder()
        method = _retag(library, embedder)
        _say(f"retagged with {method}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    with _open_library() as library:
        try:
            memo = library.get(args.id)
        except LibraryError as exc:
            _say(str(exc))
            return 1
        text = render(library, memo, args.format)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8", newline="\n")
        _say(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_export_library(args: argparse.Namespace) -> int:
    with _open_library() as library:
        text = library_json(library)
    Path(args.output).write_text(text, encoding="utf-8", newline="\n")
    _say(f"wrote {args.output}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from voicememo.web import create_app

    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        _say("refusing to bind a non-loopback address without --allow-remote")
        return 2
    with _open_library() as library:
        app = create_app(library)
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_eval_tags(args: argparse.Namespace) -> int:
    from voicememo.tagging import LsaEmbedder, Model2VecEmbedder, evaluate

    items = json.loads(Path(args.file).read_text(encoding="utf-8"))
    pairs = [(item["transcript"], item["topic"]) for item in items]
    if args.embedder == "model2vec":
        from voicememo.paths import models_dir

        embedder = Model2VecEmbedder(models_dir() / "model2vec")
    else:
        embedder = LsaEmbedder()
    _say(json.dumps(evaluate(pairs, embedder), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicememo", description="Transcribe, tag and search voice memos locally."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def backend_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--language", default="auto", help="language code or 'auto' (default)"
        )
        p.add_argument(
            "--backend",
            choices=["auto", "fake"],
            default="auto",
            help="'fake' is for demos and tests",
        )
        p.add_argument(
            "--whisper",
            default="base",
            help="whisper.cpp model used for non-Parakeet languages",
        )

    p = sub.add_parser(
        "models", help="show or download models (the only command that downloads)"
    )
    p.add_argument("action", choices=["pull", "status"])
    p.add_argument("--parakeet", action="store_true")
    p.add_argument(
        "--whisper",
        action="append",
        metavar="NAME",
        help="whisper.cpp model, e.g. base",
    )
    p.add_argument(
        "--embedder", action="store_true", help="model2vec static embeddings"
    )
    p.set_defaults(func=cmd_models)

    p = sub.add_parser(
        "ingest", help="transcribe audio files or folders into the library"
    )
    p.add_argument("paths", nargs="+")
    p.add_argument("--retry-failed", action="store_true")
    backend_args(p)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser(
        "watch", help="poll a folder and ingest memos once they stop changing"
    )
    p.add_argument("directory")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument(
        "--settle", type=float, default=5.0, help="seconds a file must stay unchanged"
    )
    p.add_argument("--once", action="store_true", help="single scan, then exit")
    backend_args(p)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("list", help="list memos")
    p.add_argument("--tag")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="show one memo with its transcript")
    p.add_argument("id", type=int)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("search", help="full-text search over transcripts")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("tag", help="add or remove a tag")
    p.add_argument("action", choices=["add", "rm"])
    p.add_argument("id", type=int)
    p.add_argument("tag")
    p.set_defaults(func=cmd_tag)

    p = sub.add_parser("rm", help="delete a memo, its transcript and its audio copy")
    p.add_argument("id", type=int)
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("retranscribe", help="transcribe a stored memo again")
    p.add_argument("id", type=int)
    backend_args(p)
    p.set_defaults(func=cmd_retranscribe)

    p = sub.add_parser(
        "retag", help="recompute automatic topic tags for the whole library"
    )
    p.add_argument("--embedder", choices=["lsa", "model2vec"], default="lsa")
    p.set_defaults(func=cmd_retag)

    p = sub.add_parser("export", help="export one memo")
    p.add_argument("id", type=int)
    p.add_argument("--format", choices=["md", "srt", "vtt", "json"], default="md")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser(
        "export-library", help="export every finished memo as one JSON file"
    )
    p.add_argument("output")
    p.set_defaults(func=cmd_export_library)

    p = sub.add_parser("serve", help="local web UI (loopback only by default)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--allow-remote", action="store_true", help="permit a non-loopback bind address"
    )
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser(
        "eval-tags", help="score topic clustering on a labelled JSON file"
    )
    p.add_argument("file")
    p.add_argument("--embedder", choices=["lsa", "model2vec"], default="lsa")
    p.set_defaults(func=cmd_eval_tags)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ModuleNotFoundError as exc:
        _say(f"missing dependency: {exc}")
        return 1
    except LanguageUnsupportedError as exc:
        _say(str(exc))
        return 1
    except Exception as exc:
        from voicememo.backends import ModelMissingError

        if isinstance(exc, ModelMissingError):
            _say(str(exc))
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
