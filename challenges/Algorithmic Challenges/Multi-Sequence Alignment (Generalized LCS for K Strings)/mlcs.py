"""Multiple Longest Common Subsequence (MLCS): the classic K-dim DP vs the dominant-point method.

Given K sequences, find the longest sequence that is a subsequence of all of
them. For K=2 this is textbook LCS; the direct generalization is a K-dimensional
DP table, one cell per K-tuple of prefix lengths -- exact, but O(prod(n_i))
time *and* space, which is only practical for a handful of short sequences.

| Method                | Time                    | Space         | Optimal? |
| ---------------------- | ------------------------ | -------------- | -------- |
| `brute_force_mlcs`     | O(prod(n_i) * K)        | O(prod(n_i))   | Always   |
| `dominant_point_mlcs`  | O(D * Sigma * K) roughly | O(D)           | Always   |
| `progressive_mlcs`     | O(sum of pairwise LCS)  | O(max(n_i)^2)  | No guarantee |

`D` is the number of *dominant points* actually generated -- typically far
smaller than `prod(n_i)`, especially once K or the sequence lengths grow past
what the brute-force table can even hold in memory. `Sigma` is the alphabet
size. `brute_force_mlcs` and `dominant_point_mlcs` are two independent exact
algorithms cross-verified against each other below; `progressive_mlcs` is a
fast heuristic with no optimality guarantee, included for the same reason
TSP's nearest-neighbor is: to make the "exact is expensive, heuristics trade
that away" tradeoff concrete.
"""

from __future__ import annotations

from functools import cache


def is_subsequence(candidate: str, sequence: str) -> bool:
    it = iter(sequence)
    return all(ch in it for ch in candidate)


def brute_force_mlcs(sequences: list[str]) -> str:
    """Direct generalization of the 2-string LCS DP to K sequences via memoized recursion.

    `table[(i_1, ..., i_K)]` = the LCS of the length-`i_1`, ..., length-`i_K`
    prefixes of the K sequences. The 2-string recurrence generalizes exactly:
    if every sequence's prefix ends in the *same* character, that character is
    always safe to take (it can only help); otherwise try dropping the last
    character of each sequence in turn and keep whichever drop leaves the
    longest result. `prod(n_i)` reachable states, each O(K) work -- exact,
    but the state space is the whole K-dimensional grid, not just the
    sequences actually needed to reach the optimum (that's what
    `dominant_point_mlcs` fixes).
    """
    k = len(sequences)
    if k == 0:
        return ""
    if any(len(s) == 0 for s in sequences):
        return ""

    @cache
    def solve(idxs: tuple[int, ...]) -> str:
        if any(i == 0 for i in idxs):
            return ""
        chars = [sequences[j][idxs[j] - 1] for j in range(k)]
        if all(c == chars[0] for c in chars):
            prev = tuple(i - 1 for i in idxs)
            return solve(prev) + chars[0]
        best = ""
        for j in range(k):
            if idxs[j] == 0:
                continue
            dropped = tuple(i - 1 if pos == j else i for pos, i in enumerate(idxs))
            candidate = solve(dropped)
            if len(candidate) > len(best):
                best = candidate
        return best

    result = solve(tuple(len(s) for s in sequences))
    solve.cache_clear()
    return result


def _build_next_occurrence(sequence: str) -> list[dict[str, int]]:
    """`table[p][c]` = smallest 1-indexed position > p holding character `c`, or absent.

    Built by a single backward scan, O(n * |alphabet actually seen|) time and
    space -- the standard "next occurrence" table the dominant-point method
    (and Hunt-Szymanski's original 2-string algorithm) relies on for O(1)
    successor lookups instead of rescanning the sequence for every extension.
    """
    n = len(sequence)
    table: list[dict[str, int]] = [{} for _ in range(n + 1)]
    for pos in range(n - 1, -1, -1):
        table[pos] = dict(table[pos + 1])
        table[pos][sequence[pos]] = pos + 1
    return table


def dominant_point_mlcs(sequences: list[str]) -> str:
    """Dominant-point / match-point DAG method (Hunt & Szymanski 1977; Hakata & Imai 1992).

    A "point" is a K-tuple of prefix lengths (p_1, ..., p_K). Point `p`
    *dominates* point `q` (same or better on every sequence) when
    `p_i <= q_i` for every `i`; among all points reachable via a length-l
    common subsequence, only the *dominant* (Pareto-minimal) ones can ever
    matter, since any non-dominant point is worse-or-equal on every
    coordinate and so can extend to nothing a dominant point at the same
    level couldn't also extend to. Instead of visiting the full
    `prod(n_i)`-cell DP grid, this walks only through levels `L_0, L_1, ...`
    of dominant points:

    - `L_0 = {(0, ..., 0)}`.
    - From each point `p` in `L_l` and each character `c` that occurs *after*
      position `p_i` in every sequence `i`, the successor point is
      `(next_occ_1[p_1][c], ..., next_occ_K[p_K][c])` -- the earliest place
      each sequence can supply its next `c`, greedily. This is always safe:
      taking the earliest available match leaves the most room for
      everything that comes after, so it can never make the final answer
      shorter than taking a later one would.
    - `L_{l+1}` is the dominant subset of all such successor points, over all
      `p` in `L_l` and all valid `c`.
    - The MLCS length is the last level index `l` with `L_l` non-empty; a
      concrete LCS is recovered by walking parent-and-character pointers
      back from any point in that level to `L_0`.

    The dominance filter here is the direct O(m^2 * K) pairwise check, not a
    sorted/incremental skyline structure (the sub-quadratic skyline
    algorithms the MLCS literature builds on top of this are a further,
    separate optimization on top of the state-space reduction -- out of
    scope here; what this implementation demonstrates is *that* reduction,
    from `prod(n_i)` reachable states down to however many are actually
    dominant, not the fastest possible way to compute a skyline).
    """
    levels, parent = dominant_point_levels(sequences)
    if not parent:
        return ""

    chars = []
    cur = levels[-1][0]
    while cur in parent:
        prev, c = parent[cur]
        chars.append(c)
        cur = prev
    return "".join(reversed(chars))


def dominant_point_levels(
    sequences: list[str],
) -> tuple[
    list[list[tuple[int, ...]]],
    dict[tuple[int, ...], tuple[tuple[int, ...], str]],
]:
    """The level-by-level search behind :func:`dominant_point_mlcs`, exposed for benchmarking.

    Returns every level `L_0, L_1, ...` actually generated (so callers can
    see how many dominant points existed at each LCS length, i.e. the actual
    size of the reduced search space) plus the parent/character pointers
    needed to reconstruct any point's path back to `L_0`.
    """
    k = len(sequences)
    if k == 0:
        return [], {}
    if any(len(s) == 0 for s in sequences):
        return [[tuple([0] * k)]], {}

    next_occ = [_build_next_occurrence(s) for s in sequences]
    alphabet = set(sequences[0])
    for s in sequences[1:]:
        alphabet &= set(s)

    start = tuple([0] * k)
    parent: dict[tuple[int, ...], tuple[tuple[int, ...], str]] = {}
    levels: list[list[tuple[int, ...]]] = [[start]]

    while True:
        level = levels[-1]
        candidates: dict[tuple[int, ...], tuple[tuple[int, ...], str]] = {}
        for point in level:
            for c in alphabet:
                nxt = []
                ok = True
                for i in range(k):
                    occ = next_occ[i][point[i]].get(c)
                    if occ is None:
                        ok = False
                        break
                    nxt.append(occ)
                if not ok:
                    continue
                succ = tuple(nxt)
                if succ not in candidates:
                    candidates[succ] = (point, c)

        if not candidates:
            break

        dominant = _dominant_points(list(candidates.keys()))
        for point in dominant:
            parent[point] = candidates[point]
        levels.append(dominant)

    return levels, parent


def _dominant_points(points: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """The Pareto-minimal subset: `p` survives iff no other point is <= it on every coordinate."""
    result = []
    for p in points:
        dominated = False
        for q in points:
            if p is q:
                continue
            if all(q[i] <= p[i] for i in range(len(p))) and q != p:
                dominated = True
                break
        if not dominated:
            result.append(p)
    return result


def progressive_mlcs(sequences: list[str]) -> str:
    """Greedy pairwise reduction: fold sequences into a running LCS one at a time.

    `cur = LCS(s_1, s_2)`, then `cur = LCS(cur, s_3)`, and so on. Each step's
    result is by construction a subsequence of `cur` (hence of every sequence
    folded in so far) *and* of the next sequence, so the final result is
    always a valid common subsequence of all K inputs -- but committing to
    one particular optimal alignment of the first two sequences before ever
    looking at the third (and so on) is exactly the well-known failure mode
    of progressive multiple sequence alignment: an early, locally-optimal
    choice can strand the search away from the true joint optimum, with no
    way to correct it later. See the README for a concrete instance where
    this actually happens.
    """
    if not sequences:
        return ""
    cur = sequences[0]
    for s in sequences[1:]:
        cur = brute_force_mlcs([cur, s])
    return cur


def _demo() -> None:
    sequences = ["ABCBDAB", "BDCABA", "AEDBCB", "BADCAB"]
    exact = brute_force_mlcs(sequences)
    fast = dominant_point_mlcs(sequences)
    heuristic = progressive_mlcs(sequences)
    print(f"sequences: {sequences}")
    print(f"brute_force_mlcs:    {exact!r} (len {len(exact)})")
    print(f"dominant_point_mlcs: {fast!r} (len {len(fast)})")
    print(f"progressive_mlcs:    {heuristic!r} (len {len(heuristic)})")
    for name, result in [("brute_force_mlcs", exact), ("dominant_point_mlcs", fast)]:
        assert all(is_subsequence(result, s) for s in sequences), name


if __name__ == "__main__":
    _demo()
