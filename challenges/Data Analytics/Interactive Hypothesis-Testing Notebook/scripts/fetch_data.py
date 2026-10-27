"""Re-create sample_data/ from the public sources listed in sample_data/SOURCES.json.

Downloads three small tables (Palmer penguins, UCI iris, UCI wine quality) and exports three
historical tables from R's datasets package with Rscript (sleep, ToothGrowth, UCBAdmissions).
Run from the challenge folder:  uv run python scripts/fetch_data.py
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample_data"
RSCRIPT = Path(r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe")
UA = {"User-Agent": "Mozilla/5.0 (hypothesis-wizard fetch script)"}

URLS = {
    "penguins.csv": "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv",
    "iris.zip": "https://archive.ics.uci.edu/static/public/53/iris.zip",
    "wine_quality.zip": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
}


def download(url: str) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=60
    ).read()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sources: dict[str, dict] = {}

    penguins = download(URLS["penguins.csv"])
    (OUT / "penguins.csv").write_bytes(penguins)

    iris = zipfile.ZipFile(io.BytesIO(download(URLS["iris.zip"])))
    (OUT / "iris.csv").write_bytes(_iris_csv(iris.read("iris.data")))

    wine = zipfile.ZipFile(io.BytesIO(download(URLS["wine_quality.zip"])))
    (OUT / "wine_red.csv").write_bytes(
        wine.read("winequality-red.csv").replace(b";", b",")
    )

    script = ROOT / "scripts" / "export_r_datasets.R"
    subprocess.run([str(RSCRIPT), str(script), str(OUT)], check=True)

    for path in sorted(OUT.glob("*.csv")):
        sources[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / "SOURCES.json").write_text(
        json.dumps({"files": sources, "urls": URLS}, indent=2) + "\n", encoding="utf-8"
    )


def _iris_csv(raw: bytes) -> bytes:
    header = b"sepal_length,sepal_width,petal_length,petal_width,species\n"
    rows = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    return header + ("\n".join(rows) + "\n").encode("utf-8")


if __name__ == "__main__":
    main()
