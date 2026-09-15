# Strongly Connected Components Finder

**Category:** Algorithmic Challenges
**Difficulty:** A (brief: "Tarjan vs Kosaraju, benchmark on large sparse graphs")

**Status:** Implemented (Python)

A strongly connected component (SCC) of a directed graph is a maximal set of
vertices where every vertex can reach every other vertex by following
directed edges. Three classic single-DFS-pass algorithms compute the same
partition with different bookkeeping:

| Method         | Idea                                        | Time     | Space |
| -------------- | ------------------------------------------- | -------- | ----- |
| `tarjan_scc`   | One DFS, a low-link value per vertex        | O(V + E) | O(V)  |
| `kosaraju_scc` | DFS finish order, then DFS the transpose    | O(V + E) | O(V)  |
| `gabow_scc`    | One DFS, two stacks, no low-link arithmetic | O(V + E) | O(V)  |

All three are implemented **iteratively** (an explicit stack standing in for
the call stack) -- the brief only asked for Tarjan and Kosaraju; Gabow's
path-based algorithm (2000) is added as the third point of comparison because
it removes the one non-obvious piece of Tarjan's algorithm (the low-link
minimum-taking) entirely, replacing it with a stack-popping rule, and is
mentioned far less often than the other two despite being competitive.

## The algorithms

### Tarjan: low-link values

Every vertex gets a discovery `index` (DFS pre-order number) and a `lowlink`:
the smallest index reachable from it via zero or more tree edges followed by
at most one back/cross edge into a vertex still on the DFS stack. A vertex
`v` is the **root** of its component exactly when `lowlink[v] == index[v]` --
nothing discovered under `v` can reach past it back up the stack, so `v` and
everything above it on the stack (down to `v` itself) is precisely one SCC.
Popping the stack down to and including `v` at that moment produces the
component. Components come out in **reverse topological order** of the
condensation DAG (see below) as a free side effect of DFS finishing order.

### Kosaraju/Sharir: two DFS passes, no low-link at all

1. DFS the graph once, recording each vertex's **finish time**.
2. Build the **transpose** graph (every edge reversed).
3. DFS the transpose, processing roots in **decreasing finish-time order**;
   each resulting tree is exactly one SCC.

The correctness argument: a DFS over the transpose, started at the
highest-finish-time unvisited vertex, can never leak out of the current
component into a component that finished earlier in step 1 -- an edge out of
the current component in the transpose graph would be an edge *into* it in
the original graph, and *that* edge would have forced the other component to
finish before this one did. No numeric low-link bookkeeping is needed; the
finish-time ordering alone enforces the invariant. The cost is a second full
graph traversal plus building the transpose (an extra O(V + E) pass and O(E)
memory) that Tarjan and Gabow avoid.

### Gabow (2000): path-based, two stacks

`S` holds every visited-but-unassigned vertex in DFS order. `P` holds a
*shrinking* subsequence of `S`: the current candidates for being a component
root. Whenever DFS finds an edge back to an already-visited, not-yet-assigned
vertex `w`, every candidate on `P` discovered *after* `w` gets popped off --
they're now known to close a cycle back to `w` and therefore share its
component, whether or not any of them individually looked like a root a
moment ago. When DFS finishes with a vertex `v` that is *still* on top of
`P`, nothing later can ever pull it into a bigger component: `v` is a root,
and popping `S` down to `v` gives its component -- exactly Tarjan's stack-pop
step, but the root test is "am I still on top of `P`?" instead of comparing
two integers. Gabow's paper frames this as removing the one part of Tarjan's
algorithm that requires derivation to trust at a glance (why does the
low-link minimum work?) in exchange for a second stack.

### Why iterative, not recursive

A textbook recursive Tarjan (`tarjan_scc_recursive` in `scc.py`, kept only
for this demo -- not exported as a real option) calls itself once per DFS
tree edge. Python's default recursion limit is ~1000 frames, and any graph
containing a directed path longer than that -- a random sparse graph well
under a hundred thousand nodes, or a real import/build dependency graph --
raises `RecursionError` before finishing:

```
$ uv run python scc.py --verify
...
Recursion-limit demo: recursive Tarjan on a long directed path
  RecursionError on a path of 1500 nodes (limit=1000), as expected
  iterative Tarjan handled the same graph fine: 1500 components (all size 1)
```

`sys.setrecursionlimit()` can paper over this for a while, but it doesn't fix
the underlying problem -- it trades a clean exception for a C stack overflow
(a hard crash, not a catchable Python error) once the new limit exceeds what
the OS thread stack can hold. All three algorithms here use an explicit
`(vertex, iterator-over-successors)` stack instead, which has no depth limit
beyond available heap memory.

## Condensation

Collapsing every SCC to a single node always produces a **DAG** -- if the
condensation had a cycle, the components on that cycle would all be mutually
reachable and therefore would have been one SCC to begin with. `condensation()`
builds this DAG from any component list:

```
>>> tarjan_scc(graph)
[[3, 2, 1], [7, 6], [5, 4], [8]]
>>> condensation(graph, _)
{0: set(), 1: {0}, 2: {0, 1}, 3: {1, 2}}
```

## Correctness

`test_scc.py` runs every hand-verified case (empty graph, single node,
self-loop, disconnected nodes, a pure DAG, a fully-connected digraph, the
CLRS textbook example, a node referenced only as a successor) against all
four algorithm variants (Tarjan, Kosaraju, Gabow, and the recursive Tarjan
used for the limit demo), plus 30 random-graph trials cross-checked against
`networkx.strongly_connected_components`, plus an all-pairs agreement sweep
between every algorithm combination:

```
$ uv run --with pytest --with numpy --with networkx pytest -q
407 passed in 1.37s
```

## Benchmarks

```
$ uv run python benchmark.py
Pure-Python Tarjan/Kosaraju/Gabow/networkx vs scipy (compiled) SCC
         n     edges    tarjan  kosaraju     gabow  networkx     scipy   scipy speedup
------------------------------------------------------------------------------------------------
     5,000    12,496    0.018s    0.004s    0.004s    0.018s   0.0036s            1.1x
    20,000    49,998    0.036s    0.023s    0.039s    0.107s   0.0108s            2.1x
    80,000   199,996    0.328s    0.348s    0.272s    0.902s   0.0524s            5.2x
   300,000   749,998    2.467s    3.051s    2.475s    9.218s   0.6564s            3.8x

Recursive Tarjan: fine until it isn't (RecursionError, not just slow)
   path length    iterative tarjan    recursive tarjan
------------------------------------------------------
           250             0.0005s             0.0004s
           500             0.0010s             0.0006s
           950             0.0015s             0.0009s
(sys.getrecursionlimit() == 1000 on this interpreter)
```

Two things the numbers show. First, **Gabow and Tarjan trade the lead** as
the graph grows (Kosaraju edges ahead at 5k, Gabow fastest at 80k and tied
with Tarjan at 300k) while **Kosaraju falls behind at scale** -- its second
full DFS pass plus building the transpose is a real, measurable cost once
the graph is big enough that constant factors stop dominating, exactly as
the complexity analysis predicts (same O(V+E), larger hidden constant).
`networkx`'s pure-Python implementation is consistently 3-10x slower than
all three of ours at every size, which is a generic-library-overhead story
(edge and attribute dict layers `scc.py`'s flat adjacency-list dicts skip),
not a comment on its correctness -- it's the reference oracle the test suite
checks against. Second, **scipy's compiled `connected_components`** (a
Tarjan variant with no per-edge Python object overhead) is 1x-5x faster
still than the fastest pure-Python entrant at every size tested -- the honest
ceiling for "how fast can this actually go," and a reminder that these
three algorithms' relative *ranking* is the real result of this challenge,
not their absolute wall-clock time against a compiled baseline they were
never going to beat.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Strongly Connected Components Finder"

uv run --with numpy python scc.py                      # demo on the CLRS textbook graph
uv run --with numpy --with networkx python scc.py --verify   # cross-check + recursion demo
uv run --with numpy --with networkx --with scipy python benchmark.py

uv run --with pytest --with numpy --with networkx pytest -q   # 407 tests
```

## Where this is used

**Compiler and build-system dependency analysis.** Detecting circular
imports/circular build targets is exactly SCC-finding on the dependency
graph; any SCC of size > 1 is a cycle that needs breaking (or, for mutually
recursive functions/modules, is expected and left alone).

**Deadlock detection.** A wait-for graph (process A waits on a resource held
by process B) has a deadlock if and only if it contains a cycle, i.e. an SCC
of size > 1 -- database engines and OS schedulers use SCC/cycle detection on
this graph for exactly this check.

**2-SAT solving.** The classic linear-time 2-SAT algorithm builds an
implication graph over literals and their negations, then checks that no
variable and its negation land in the same SCC -- if they do, the formula is
unsatisfiable. This is the textbook application that made Tarjan/Kosaraju
must-know algorithms rather than a graph-theory curiosity.

**Web graph and social network structure.** The "bowtie" model of the web
graph (Broder et al., 2000) is built directly from its single giant SCC (the
mutually-reachable core) plus the in-component and out-component hanging off
it -- SCC decomposition is the first step in that kind of structural
analysis.

## Sources

- [Tarjan, "Depth-First Search and Linear Graph Algorithms" (1972)](https://doi.org/10.1137/0201010) -- the original low-link algorithm.
- Sharir, M., "A strong-connectivity algorithm and its applications in data flow analysis," *Computers & Mathematics with Applications* 7(1):67-72, 1981 -- the earliest print citation of the algorithm attributed to Kosaraju's unpublished 1978 lecture notes.
- [Gabow, "Path-based depth-first search for strong and biconnected components" (2000)](https://doi.org/10.1016/S0020-0190(00)00051-X) -- the two-stack path-based algorithm.
- Cormen, Leiserson, Rivest & Stein, *Introduction to Algorithms* (CLRS), 3rd ed., §22.5 -- the textbook graph used in `test_scc.py` and the demo.
- [Broder et al., "Graph structure in the web" (2000)](https://doi.org/10.1016/S1389-1286(00)00083-9) -- the bowtie model, mentioned above under "where this is used."
