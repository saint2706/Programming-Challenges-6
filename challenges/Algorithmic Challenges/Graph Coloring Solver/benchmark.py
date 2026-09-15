"""Chromatic-count comparison across graph families, and exact-solver runtime scaling.

    uv run python benchmark.py

Two tables:

1. How many colors each heuristic uses vs the true chromatic number, across
   graphs deliberately chosen to stress different things: cycles (trivial),
   complete graphs (trivial), the crown graph (breaks plain greedy on a
   *bad order*), Petersen and the Groetzsch graph (small named graphs with
   well-known structure), and queen graphs (the classic hard family for
   greedy-style heuristics -- queens attack along rows, columns, and both
   diagonals, so the graph is dense and highly symmetric).

2. How fast `exact_chromatic_number` runs as graphs grow, showing where NP-hardness
   actually starts to hurt in wall-clock terms for this implementation.
"""

from __future__ import annotations

import time

import networkx as nx
from coloring import (
    crown_graph,
    dsatur,
    exact_chromatic_number,
    from_networkx,
    greedy_color,
    num_colors,
    queen_graph,
    rlf,
    welsh_powell,
)


def _bench_chromatic_counts() -> None:
    families: list[tuple[str, dict]] = [
        ("cycle C7 (odd)", from_networkx(nx.cycle_graph(7))),
        ("cycle C8 (even)", from_networkx(nx.cycle_graph(8))),
        ("complete K8", from_networkx(nx.complete_graph(8))),
        ("petersen", from_networkx(nx.petersen_graph())),
        ("groetzsch (mycielski C5)", from_networkx(nx.mycielskian(nx.cycle_graph(5)))),
        ("crown n=10 (2-chromatic)", crown_graph(10)),
        ("queen4_4", queen_graph(4)),
        ("queen5_5", queen_graph(5)),
        ("queen6_6", queen_graph(6)),
        ("queen7_7", queen_graph(7)),
        ("random G(20, 0.5)", from_networkx(nx.gnp_random_graph(20, 0.5, seed=1))),
        ("random G(40, 0.5)", from_networkx(nx.gnp_random_graph(40, 0.5, seed=1))),
    ]

    print("Chromatic count: heuristics vs the true chromatic number")
    print(
        f"{'graph':<26}{'n':>5}{'greedy':>8}{'welsh-p':>9}{'dsatur':>8}{'rlf':>6}{'EXACT':>7}"
    )
    print("-" * 69)
    for name, graph in families:
        g = num_colors(greedy_color(graph))
        wp = num_colors(welsh_powell(graph))
        ds = num_colors(dsatur(graph))
        r = num_colors(rlf(graph))
        k, _ = exact_chromatic_number(graph)
        print(f"{name:<26}{len(graph):>5}{g:>8}{wp:>9}{ds:>8}{r:>6}{k:>7}")
    print()
    print(
        "Cycles, the complete graph, Petersen, and Groetzsch are all easy for "
        "every method here -- every heuristic already matches EXACT. The crown "
        "graph is the opposite extreme: default (interleaved-insertion-order) "
        "greedy and Welsh-Powell both land on 10 colors for a graph whose "
        "chromatic number is 2, because the graph is regular (every vertex has "
        "the same degree), so Welsh-Powell's degree sort provides no new "
        "information and its stable sort just preserves the same pathological "
        "order greedy started with. Queen graphs are the honest hard case: "
        "every heuristic here is at least 1 color above EXACT on queen6_6 and "
        "queen7_7, and on queen7_7, Welsh-Powell (12) is worse than plain "
        "greedy (10) -- a reminder that a 'smarter-sounding' static order is "
        "not a strict improvement without dynamic recomputation like DSATUR's."
    )


def _bench_exact_runtime() -> None:
    print()
    print("exact_chromatic_number runtime as the (queen graph) instance grows")
    print(f"{'graph':<12}{'n':>5}{'chromatic number':>18}{'time':>12}")
    print("-" * 47)
    for size in (4, 5, 6, 7):
        graph = queen_graph(size)
        start = time.perf_counter()
        k, _ = exact_chromatic_number(graph)
        t = time.perf_counter() - start
        print(f"queen{size}_{size:<6}{len(graph):>5}{k:>18}{t:11.3f}s")
    print()
    print(
        "queen8_8 (64 vertices) is deliberately not included here: it did not "
        "finish within a 45-second budget when checked manually, which is the "
        "real behavior of an NP-hard exact solver hitting a genuinely hard "
        "instance -- not a bug to paper over with a bigger timeout. Contrast "
        "with the random-graph runtimes below, which reach n=50 in a couple "
        "of seconds: 'how big a graph the exact solver can handle' depends "
        "enormously on structure, not just vertex count."
    )

    print()
    print("exact_chromatic_number runtime on random G(n, 0.5) graphs")
    print(f"{'n':>6}{'chromatic number':>18}{'time':>10}")
    print("-" * 34)
    for n in (30, 35, 40, 45, 50, 55):
        graph = from_networkx(nx.gnp_random_graph(n, 0.5, seed=1))
        start = time.perf_counter()
        k, _ = exact_chromatic_number(graph)
        t = time.perf_counter() - start
        print(f"{n:>6}{k:>18}{t:9.3f}s")


def main() -> None:
    _bench_chromatic_counts()
    _bench_exact_runtime()


if __name__ == "__main__":
    main()
