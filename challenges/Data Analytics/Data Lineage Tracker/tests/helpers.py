"""Build small throwaway pipelines on disk."""

from __future__ import annotations

from pathlib import Path

from data_lineage.graph import build_graph
from data_lineage.project import load_project


def make_pipeline(
    root: Path,
    sources: dict[str, str],
    models: dict[str, str],
) -> Path:
    """``sources`` maps a table name to CSV text; ``models`` maps ``layer/name`` to SQL."""
    root.mkdir(parents=True, exist_ok=True)
    toml = []
    for name, csv in sources.items():
        (root / f"{name}.csv").write_text(csv.strip() + "\n", encoding="utf-8")
        toml.append(f'[sources.{name}]\npath = "{name}.csv"\n')
    (root / "sources.toml").write_text("\n".join(toml), encoding="utf-8")
    for path, sql in models.items():
        target = root / f"{path}.sql"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sql, encoding="utf-8")
    return root


def load(root: Path):
    project = load_project(root)
    return project, build_graph(project)


T_CSV = """
a,b,c,d
1,10,x,2020-01-01
2,20,y,2020-01-02
3,30,x,2020-01-03
4,40,z,2020-01-04
5,50,y,2020-01-05
6,60,x,2020-01-06
7,70,z,2020-01-07
8,80,y,2020-01-08
"""

U_CSV = """
k,v
1,100
2,200
3,300
"""
