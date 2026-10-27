"""Watch-folder auto-ingest by polling.

A file is ingested only after its (size, mtime) has stayed unchanged for ``settle_seconds``,
so half-copied memos are left alone. Temp and hidden files are ignored by name. Clock and sleep
are injected, so tests drive the loop with a fake clock and never sleep.

Restart safety comes from the library: ingest is keyed by content hash, so a rescan after a
restart finds the memo already stored and does no transcription.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from voicememo.audio import DecodeError
from voicememo.ingest import Engine, IngestResult, ingest_file
from voicememo.library import Library, LibraryError

log = logging.getLogger("voicememo.watch")

AUDIO_SUFFIXES = frozenset(
    {
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
        ".mp4",
    }
)
TEMP_SUFFIXES = frozenset(
    {".part", ".tmp", ".temp", ".crdownload", ".download", ".partial", ".swp"}
)

Signature = tuple[int, int]  # (size in bytes, mtime in ns)


def is_candidate(path: Path) -> bool:
    name = path.name
    if name.startswith((".", "~")) or name.endswith("~"):
        return False
    suffix = path.suffix.lower()
    return suffix in AUDIO_SUFFIXES and suffix not in TEMP_SUFFIXES


@dataclass
class Watcher:
    root: Path
    library: Library
    engine: Engine
    language: str | None = None
    settle_seconds: float = 5.0
    clock: Callable[[], float] = time.monotonic
    _first_seen: dict[Path, tuple[Signature, float]] = field(default_factory=dict)
    _done: dict[Path, Signature] = field(default_factory=dict)

    def scan(self) -> list[tuple[Path, IngestResult]]:
        """One pass over the folder. Returns the files that were handed to the pipeline."""
        now = self.clock()
        present: dict[Path, Signature] = {}
        for path in sorted(self.root.iterdir()):
            try:
                if path.is_symlink() or not path.is_file() or not is_candidate(path):
                    continue  # symlinks could point outside the watched folder
                stat = path.stat()
            except OSError as exc:  # deleted or locked between listing and stat
                log.warning("%s: skipped, cannot read (%s)", path.name, exc)
                continue
            present[path] = (stat.st_size, stat.st_mtime_ns)

        # Forget files that vanished so a re-created file starts its settle window again.
        for gone in [p for p in self._first_seen if p not in present]:
            del self._first_seen[gone]

        handled: list[tuple[Path, IngestResult]] = []
        for path, sig in present.items():
            previous = self._first_seen.get(path)
            if previous is None or previous[0] != sig:
                self._first_seen[path] = (
                    sig,
                    now,
                )  # new or still changing: restart the window
                continue
            if now - previous[1] < self.settle_seconds:
                continue
            if self._done.get(path) == sig:
                continue
            self._done[path] = (
                sig  # set first: a bad file is reported once, not on every scan
            )
            try:
                result = ingest_file(
                    self.library, path, self.engine, language=self.language
                )
            except (OSError, DecodeError, LibraryError) as exc:
                log.warning("%s: not ingested (%s)", path.name, exc)
                continue
            log.info("%s: %s %s", path.name, result.status, result.message)
            handled.append((path, result))
        return handled

    def run(
        self,
        interval: float = 2.0,
        *,
        max_scans: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> int:
        """Scan forever (or ``max_scans`` times). Returns the number of files handled."""
        handled = 0
        scans = 0
        while max_scans is None or scans < max_scans:
            handled += len(self.scan())
            scans += 1
            if max_scans is None or scans < max_scans:
                sleep(interval)
        return handled
