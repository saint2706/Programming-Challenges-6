"""JSON-lines access logger.

Every request that reaches a protected route -- successful directory
listings, downloads, and auth denials -- gets exactly one JSON object per
line, so "who fetched what and when" can be answered by reading (or tailing)
the log file after the fact, without a database. A single process-wide
`threading.Lock` serializes writes: FastAPI runs sync route handlers in a
thread pool, so concurrent requests can genuinely race to append a line at
the same time, and unsynchronized appends could interleave two writers'
bytes into one corrupt line.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class AccessLogger:
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        identity: str,
        ip: str,
        method: str,
        path: str,
        action: str,
        status: int,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "identity": identity,
            "ip": ip,
            "method": method,
            "path": path,
            "action": action,
            "status": status,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def tail(self, n: int = 200) -> list[dict[str, Any]]:
        """Most recent `n` entries, newest first. Empty list if no log yet."""
        if not self.log_path.exists():
            return []
        with self._lock:
            text = self.log_path.read_text(encoding="utf-8")
        entries: list[dict[str, Any]] = []
        for line in text.splitlines()[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a torn/partial line rather than fail the whole tail
        entries.reverse()
        return entries
