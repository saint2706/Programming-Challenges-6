"""Snapshots: a host's variables as fingerprints only, safe to copy to another machine.

``envdiff snapshot .env`` on each host (same passphrase) writes names, keyed fingerprints and coarse
flags. ``envdiff diff host-a.json host-b.json`` compares them later with no secret in either file.
The file records which key made it (an id derived from the key, not the key), so mixing snapshots
made with different passphrases is an error instead of a wall of false differences.
"""

import json
from pathlib import Path

from envdiff.fingerprint import Fingerprints
from envdiff.model import Entry, Source

MARKER = "envdiff_snapshot"
VERSION = 1


class SnapshotError(ValueError):
    pass


def snapshot_to_dict(source: Source) -> dict:
    return {
        MARKER: VERSION,
        "label": source.label,
        "key_id": source.key_id,
        "entries": {
            name: {
                "fp": None
                if entry.fingerprints is None
                else entry.fingerprints.as_list(),
                "empty": entry.empty,
                "placeholder": entry.placeholder,
                "edge_ws": entry.edge_whitespace,
                "shape": entry.shape,
            }
            for name, entry in sorted(source.entries.items())
        },
    }


def write_snapshot(source: Source, path: Path) -> None:
    path.write_text(
        json.dumps(snapshot_to_dict(source), indent=2) + "\n", encoding="utf-8"
    )


def looks_like_snapshot(text: str) -> bool:
    head = text.lstrip()[:200]
    return head.startswith("{") and f'"{MARKER}"' in text[:2000]


def read_snapshot(text: str, origin: str) -> Source:
    try:
        data = json.loads(text)
        if data.get(MARKER) != VERSION:
            raise SnapshotError(f"{origin}: unsupported snapshot version")
        source = Source(
            str(data.get("label") or origin),
            origin,
            "snapshot",
            key_id=data.get("key_id"),
        )
        for name, item in data["entries"].items():
            fp = item["fp"]
            fingerprints = None if fp is None else Fingerprints(*fp)
            source.entries[name] = Entry(
                fingerprints,
                bool(item["empty"]),
                bool(item["placeholder"]),
                bool(item["edge_ws"]),
                str(item["shape"]),
            )
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise SnapshotError(
            f"{origin}: not a valid envdiff snapshot ({type(exc).__name__})"
        ) from exc
    return source
