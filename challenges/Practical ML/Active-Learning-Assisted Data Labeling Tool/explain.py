"""Why was this item suggested? Structured reasons, recomputed from the state (never cached)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Reason:
    idx: int
    top: list[tuple[int, float]]  # (class id, probability), most likely first
    margin: float  # p(top 1) - p(top 2)
    neighbors: list[
        tuple[int, int, float]
    ]  # (pool index, class id, cosine), nearest labeled first
    nearest_sim: float | None  # cosine to the nearest labeled item
    novel: bool  # among the items farthest from everything labeled
    novelty_pct: float
    cluster: (
        dict | None
    )  # {"id", "size", "labeled"} when the strategy clusters the pool


def _unit(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norm == 0, 1.0, norm)


def explain(state, selection, *, clusters=None, n_neighbors=3, novelty_pct=10.0):
    idx = np.asarray(selection.idx, dtype=np.int64)
    P = state.P[idx]
    order = np.argsort(-P, axis=1, kind="stable")[:, :3]
    ps = np.sort(P, axis=1)
    margin = ps[:, -1] - ps[:, -2]
    have = len(state.labeled) > 0
    if have:
        Xn = _unit(np.asarray(state.X, dtype=np.float64))
        L = Xn[state.labeled]
        sims = Xn[idx] @ L.T
        near = np.argsort(-sims, axis=1, kind="stable")[:, :n_neighbors]
        # "far" is relative: below the novelty_pct-th percentile of how close the unlabeled pool is
        cutoff = np.percentile((Xn[state.unlabeled()] @ L.T).max(axis=1), novelty_pct)
    out = []
    for r, i in enumerate(idx):
        neighbors, nearest, novel = [], None, False
        if have:
            neighbors = [
                (int(state.labeled[p]), int(state.y[p]), float(sims[r, p]))
                for p in near[r]
            ]
            nearest = neighbors[0][2]
            novel = bool(nearest < cutoff)
        cluster = None
        if clusters is not None:
            c = clusters[i]
            cluster = {
                "id": int(c),
                "size": int((clusters == c).sum()),
                "labeled": int((clusters[state.labeled] == c).sum()),
            }
        out.append(
            Reason(
                idx=int(i),
                top=[(int(c), float(P[r, c])) for c in order[r]],
                margin=float(margin[r]),
                neighbors=neighbors,
                nearest_sim=nearest,
                novel=novel,
                novelty_pct=novelty_pct,
                cluster=cluster,
            )
        )
    return out


def render(reason: Reason, classes, texts=None, width: int = 70) -> str:
    top = ", ".join(f"{classes[c]} {p:.0%}" for c, p in reason.top)
    lines = [f"model: {top} (margin {reason.margin:.2f})"]
    if reason.neighbors:
        for j, c, s in reason.neighbors:
            snippet = f" '{texts[j][:width]}'" if texts is not None else ""
            lines.append(f"near labeled #{j}{snippet} -> {classes[c]} (cosine {s:.2f})")
        if reason.novel:
            lines.append(
                f"novel: far from everything labeled (nearest cosine {reason.nearest_sim:.2f}, "
                f"in the farthest {reason.novelty_pct:.0f}% of the pool)"
            )
    else:
        lines.append("nothing labeled yet")
    if reason.cluster:
        c = reason.cluster
        lines.append(f"cluster {c['id']}: {c['size']} items, {c['labeled']} labeled")
    return "\n".join(lines)
