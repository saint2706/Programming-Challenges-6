# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "networkx"]
# ///
"""Builds tests/fixtures: instances plus reference answers from code that shares nothing with ours.

* Assignment matrices: SciPy's `linear_sum_assignment` (Crouse's Jonker-Volgenant in C++). An entry
  of `x` is forbidden (+inf for SciPy); an infeasible instance is recorded as `infeasible`.
* Min-cost-flow networks: NetworkX's `network_simplex`, written as DIMACS `p min` files.

`expected.csv` has one line per instance: `kind,file,expected`. Run from the challenge directory:

    uv run --script scripts/make_fixtures.py
"""

from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SEED = 20261021


def matrix_text(c: np.ndarray, forbidden: np.ndarray) -> str:
    rows = []
    for i in range(c.shape[0]):
        rows.append(
            ",".join(
                "x" if forbidden[i, j] else str(int(c[i, j])) for j in range(c.shape[1])
            )
        )
    return "\n".join(rows) + "\n"


def assignment_fixtures(rng: np.random.Generator) -> list[tuple[str, str, str]]:
    shapes = [
        (1, 1),
        (5, 5),
        (10, 10),
        (25, 25),
        (40, 40),
        (6, 12),
        (12, 6),
        (30, 50),
        (50, 30),
    ]
    kinds = {
        "uniform": (0, 1000),
        "narrow": (0, 3),
        "signed": (-500, 500),
    }
    rows = []
    for (r, k), (name, (lo, hi)), percent in (
        (s, kind, p) for s in shapes for kind in kinds.items() for p in (0, 30, 70)
    ):
        c = rng.integers(lo, hi + 1, size=(r, k))
        forbidden = rng.random((r, k)) * 100 < percent
        # Keep most instances feasible by planting one full-size matching; leave some to chance.
        if rng.random() < 0.8:
            planted = rng.permutation(max(r, k))[: min(r, k)]
            for t, p_ in enumerate(planted):
                if r <= k:
                    forbidden[t, p_] = False
                else:
                    forbidden[p_, t] = False
        fname = f"asg_{name}_{r}x{k}_f{percent}.csv"
        (OUT / fname).write_text(matrix_text(c, forbidden), newline="\n")
        scipy_cost = np.where(forbidden, np.inf, c.astype(float))
        try:
            ri, ci = linear_sum_assignment(scipy_cost)
            expected = str(int(round(c[ri, ci].sum())))
        except ValueError:
            expected = "infeasible"
        rows.append(("matrix", fname, expected))
    return rows


def network_fixtures(rng: np.random.Generator) -> list[tuple[str, str, str]]:
    rows = []
    for idx, (nodes, arcs, negative) in enumerate(
        [
            (6, 12, False),
            (8, 20, True),
            (15, 50, False),
            (15, 50, True),
            (30, 120, True),
            (60, 300, False),
            (60, 300, True),
        ]
    ):
        for rep in range(3):
            g = nx.DiGraph()
            g.add_nodes_from(range(nodes))
            pairs = set()
            while len(pairs) < arcs:
                a, b = (int(x) for x in rng.integers(0, nodes, size=2))
                if a != b:
                    pairs.add((a, b))
            lo = -25 if negative else 0
            for a, b in sorted(pairs):
                g.add_edge(
                    a,
                    b,
                    capacity=int(rng.integers(0, 20)),
                    weight=int(rng.integers(lo, 100)),
                )
            supply = np.zeros(nodes, dtype=int)
            for _ in range(max(1, nodes // 5)):
                s, t = (int(x) for x in rng.choice(nodes, size=2, replace=False))
                amount = int(rng.integers(1, 8))
                supply[s] += amount
                supply[t] -= amount
            for v in range(nodes):
                g.nodes[v]["demand"] = -int(supply[v])  # NetworkX: demand = -supply
            try:
                cost, _ = nx.network_simplex(g)
                expected = str(int(cost))
            except nx.NetworkXUnfeasible:
                expected = "infeasible"
            lines = [f"p min {nodes} {g.number_of_edges()}"]
            lines += [
                f"n {v + 1} {int(supply[v])}" for v in range(nodes) if supply[v] != 0
            ]
            lines += [
                f"a {a + 1} {b + 1} 0 {d['capacity']} {d['weight']}"
                for a, b, d in sorted(g.edges(data=True))
            ]
            fname = f"net_{idx}_{rep}.min"
            (OUT / fname).write_text("\n".join(lines) + "\n", newline="\n")
            rows.append(("flow", fname, expected))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():
        old.unlink()
    rng = np.random.default_rng(SEED)
    rows = assignment_fixtures(rng) + network_fixtures(rng)
    (OUT / "expected.csv").write_text(
        "".join(f"{kind},{file},{expected}\n" for kind, file, expected in rows),
        newline="\n",
    )
    infeasible = sum(1 for r in rows if r[2] == "infeasible")
    print(f"{len(rows)} fixtures ({infeasible} infeasible) in {OUT}")


if __name__ == "__main__":
    main()
