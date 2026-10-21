"""Download the Pizza&Chili benchmark texts into data/ (gitignored).

    uv run --no-project python scripts/fetch_corpus.py

Four real texts, each the 50 MB prefix published by the corpus authors: DNA (alphabet 16), English
(239), program sources (230) and protein sequences (27). Each download is checked against the
SHA-256 pinned in corpus.json, so a changed or truncated file fails loudly. Run it with
`--record` once to (re)write the pins after deliberately changing the list.

Source: http://pizzachili.dcc.uchile.cl/texts.html (P. Ferragina and G. Navarro).
"""

import gzip
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = ROOT / "corpus.json"
BASE = "https://pizzachili.dcc.uchile.cl/texts"
TEXTS = {
    "dna": "dna/dna.50MB.gz",
    "english": "nlang/english.50MB.gz",
    "sources": "code/sources.50MB.gz",
    "proteins": "protein/proteins.50MB.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    record = "--record" in sys.argv
    pins = {} if record or not MANIFEST.exists() else json.loads(MANIFEST.read_text("utf-8"))
    DATA.mkdir(exist_ok=True)
    out = {}
    for name, rel in TEXTS.items():
        target = DATA / name
        archive = DATA / f"{name}.gz"
        if not target.exists():
            url = f"{BASE}/{rel}"
            print(f"downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as f:
                shutil.copyfileobj(response, f)
            digest = sha256(archive)
            pinned = pins.get(name, {}).get("gz_sha256")
            if pinned and pinned != digest:
                archive.unlink()
                print(f"error: {name}: sha256 {digest} does not match the pinned {pinned}")
                return 1
            with gzip.open(archive, "rb") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            archive.unlink()
        else:
            digest = pins.get(name, {}).get("gz_sha256", "")
        out[name] = {"url": f"{BASE}/{rel}", "gz_sha256": digest, "bytes": target.stat().st_size}
        print(f"{name}: {out[name]['bytes']:,} bytes")
    if record or not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(out, indent=4) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
