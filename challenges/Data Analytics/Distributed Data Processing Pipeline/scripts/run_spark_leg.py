"""Run only the Spark legs (1m and 12m, Parquet) and merge them into results/results.json.

On Windows this needs HADOOP_HOME pointing at a folder with bin/winutils.exe and bin/hadoop.dll
(see the Spark section of the README). The other engines' rows are left as they are, so run it on
a quiet machine with the same repeats and warmup as the first matrix.
"""

from __future__ import annotations

import json

from dist_pipeline.bench import ALL_QUERIES, default_workdir, run_matrix
from dist_pipeline.paths import results_dir

NOTE = (
    "Spark legs rerun separately on the quiet machine with the same repeats and warmup after "
    "winutils.exe and hadoop.dll (Hadoop 3.3.6 build) were installed; the other engines were run "
    "in the first matrix."
)


def main() -> None:
    rows = run_matrix(
        ("1m", "12m"),
        ("spark",),
        ALL_QUERIES,
        repeats=3,
        warmup=1,
        workdir=default_workdir(),
        csv_scales=(),
        csv_engines=(),
        progress=lambda r: print(
            r["scale"], r["engine"], r["query"], r["status"], flush=True
        ),
    )
    path = results_dir() / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    old = payload["matrix"]
    first = next((i for i, r in enumerate(old) if r["engine"] == "spark"), len(old))
    kept = [r for r in old if r["engine"] != "spark"]
    kept[first:first] = json.loads(json.dumps(rows, default=str))
    payload["matrix"] = kept
    payload["spark_note"] = NOTE
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("merged", len(rows), "spark rows")


if __name__ == "__main__":
    main()
