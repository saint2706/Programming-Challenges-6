"""Monotonic weight-of-evidence (WOE) binning, from scratch.

A scorecard feature is a handful of bins, each with a WOE = ln(P(bin | good) / P(bin | bad)). The
bins must be *monotone* in bad rate, otherwise a score goes up as a borrower gets riskier, which no
credit officer or regulator will accept.

The method:

1. Cut the non-missing, non-special values into fine bins (one per distinct value when there are at
   most 20, else quantiles).
2. Run pool-adjacent-violators (PAVA) on the bad rates, in both directions, and keep the direction
   that fits better. That gives the optimal monotone blocks.
3. Merge blocks that are too small, or have no goods or no bads, into the neighbour that changes the
   fit least (Ward cost), then merge down to ``max_bins``. Merging adjacent blocks of a monotone
   sequence keeps it monotone.
4. Special values (sentinel codes) and missing values never merge into a numeric bin: each gets its
   own bin, whose WOE is whatever the data says.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

SMOOTHING = 0.5  # added to every bin's good and bad count so no WOE is infinite


@dataclass
class Bin:
    label: str
    kind: str  # "range" | "special" | "missing"
    good: int
    bad: int
    woe: float = 0.0

    @property
    def count(self) -> int:
        return self.good + self.bad

    @property
    def bad_rate(self) -> float:
        return self.bad / self.count if self.count else float("nan")


@dataclass
class Binning:
    feature: str
    edges: list[float]  # right-closed cuts: range bin i is (edges[i-1], edges[i]]
    specials: list[float]
    bins: list[
        Bin
    ]  # range bins, then the special bin, then the missing bin (each if present)
    direction: str  # "ascending" | "descending" | "none" (bad rate vs value)

    @property
    def iv(self) -> float:
        """Information value: sum of (P(bin|good) - P(bin|bad)) * WOE."""
        good = np.array([b.good for b in self.bins], float)
        bad = np.array([b.bad for b in self.bins], float)
        pg, pb = _distributions(good, bad)
        return float(np.sum((pg - pb) * np.log(pg / pb)))

    # -- lookup ------------------------------------------------------------------------------
    def _slots(self) -> tuple[int, int]:
        """Bin index of the special and the missing bin; ``len(bins)`` (WOE 0) when absent."""
        none = len(self.bins)
        kinds = [b.kind for b in self.bins]
        return (
            kinds.index("special") if "special" in kinds else none,
            kinds.index("missing") if "missing" in kinds else none,
        )

    def index(self, x: np.ndarray) -> np.ndarray:
        """Bin index per value. A value whose bin had no training rows maps to ``len(bins)``."""
        x = np.asarray(x, dtype=float)
        special_slot, missing_slot = self._slots()
        out = np.searchsorted(np.asarray(self.edges), x, side="left")
        if self.specials:
            out = np.where(np.isin(x, self.specials), special_slot, out)
        return np.where(np.isnan(x), missing_slot, out)

    def woe_table(self) -> np.ndarray:
        """WOE per bin index, with a trailing 0.0 for values that fall in an unseen bin."""
        return np.array([b.woe for b in self.bins] + [0.0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.woe_table()[self.index(x)]

    # -- persistence -------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Binning:
        return cls(**{**d, "bins": [Bin(**b) for b in d["bins"]]})


def _distributions(good: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = len(good)
    pg = (good + SMOOTHING) / (good.sum() + SMOOTHING * k)
    pb = (bad + SMOOTHING) / (bad.sum() + SMOOTHING * k)
    return pg, pb


def _pava(
    rate: np.ndarray, weight: np.ndarray, increasing: bool
) -> list[tuple[int, int, float]]:
    """Weighted pool-adjacent-violators. Returns (start, end, mean) blocks, end inclusive."""
    sign = 1.0 if increasing else -1.0
    blocks: list[list[float]] = []  # [sum_w, sum_wr, start, end]
    for i, (r, w) in enumerate(zip(rate, weight, strict=True)):
        blocks.append([w, w * r, i, i])
        while (
            len(blocks) > 1
            and sign * (blocks[-2][1] / blocks[-2][0] - blocks[-1][1] / blocks[-1][0])
            > 0
        ):
            w2, wr2, _, end = blocks.pop()
            blocks[-1][0] += w2
            blocks[-1][1] += wr2
            blocks[-1][3] = end
    return [(int(b[2]), int(b[3]), b[1] / b[0]) for b in blocks]


def _sse(
    rate: np.ndarray, weight: np.ndarray, blocks: list[tuple[int, int, float]]
) -> float:
    return sum(
        float(np.sum(weight[s : e + 1] * (rate[s : e + 1] - mean) ** 2))
        for s, e, mean in blocks
    )


def _ward(a: list[int], b: list[int]) -> float:
    """Cost of merging two blocks of [good, bad]: weighted squared difference of bad rates."""
    na, nb = sum(a), sum(b)
    if na == 0 or nb == 0:
        return 0.0
    return na * nb / (na + nb) * (a[1] / na - b[1] / nb) ** 2


def _merge_blocks(
    blocks: list[list[int]],  # [good, bad] per block
    his: list[float],  # upper edge per block (last one is +inf)
    *,
    min_count: int,
    max_bins: int,
) -> tuple[list[list[int]], list[float]]:
    def undersized(b: list[int]) -> bool:
        return b[0] + b[1] < min_count or b[0] == 0 or b[1] == 0

    while len(blocks) > 1:
        small = [i for i, b in enumerate(blocks) if undersized(b)]
        if small:
            # Smallest first, merged with whichever neighbour hurts the fit least.
            i = min(small, key=lambda j: sum(blocks[j]))
            pairs = [p for p in (i - 1, i) if 0 <= p < len(blocks) - 1]
        elif len(blocks) > max_bins:
            pairs = list(range(len(blocks) - 1))
        else:
            break
        p = min(pairs, key=lambda q: _ward(blocks[q], blocks[q + 1]))
        blocks[p] = [blocks[p][0] + blocks[p + 1][0], blocks[p][1] + blocks[p + 1][1]]
        his[p] = his[p + 1]
        del blocks[p + 1], his[p + 1]
    return blocks, his


def _label(edges: list[float], i: int) -> str:
    if not edges:
        return "all values"
    if i == 0:
        return f"<= {edges[0]:.4g}"
    if i == len(edges):
        return f"> {edges[-1]:.4g}"
    return f"({edges[i - 1]:.4g}, {edges[i]:.4g}]"


def fit_binning(
    feature: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    specials: tuple[float, ...] | list[float] = (),
    max_bins: int = 8,
    min_bin_frac: float = 0.02,
    monotonic: str = "auto",
    max_fine_bins: int = 50,
) -> Binning:
    """Fit monotonic WOE bins for one feature.

    ``y`` is 1 for bad, 0 for good. ``monotonic`` is "auto" (best direction), "ascending",
    "descending" or "none" (bad-rate order is not enforced; only size rules apply).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    specials = list(specials)
    is_missing = np.isnan(x)
    is_special = (
        np.isin(x, specials) & ~is_missing if specials else np.zeros(len(x), bool)
    )
    regular = ~(is_missing | is_special)
    xr, yr = x[regular], y[regular]
    min_count = math.ceil(min_bin_frac * len(x))

    edges: list[float] = []
    blocks: list[list[int]] = []
    direction = "none"
    if len(xr):
        uniq = np.unique(xr)
        if len(uniq) <= 20:
            cuts = uniq[:-1]
        else:
            cuts = np.unique(
                np.quantile(xr, np.linspace(0, 1, max_fine_bins + 1)[1:-1])
            )
            cuts = cuts[cuts < uniq[-1]]
        idx = np.searchsorted(cuts, xr, side="left")
        n_fine = len(cuts) + 1
        bad = np.bincount(idx, weights=yr, minlength=n_fine)
        cnt = np.bincount(idx, minlength=n_fine).astype(float)
        keep = (
            cnt > 0
        )  # quantile cuts can leave an empty bin; drop it so PAVA sees real data
        starts = np.flatnonzero(keep)
        fine_his = [float(cuts[i]) if i < len(cuts) else math.inf for i in starts]
        bad, cnt = bad[keep], cnt[keep]
        rate = bad / cnt
        if monotonic == "none":
            groups = [(i, i, rate[i]) for i in range(len(rate))]
        else:
            fits = {
                d: _pava(rate, cnt, increasing=(d == "ascending"))
                for d in (
                    ("ascending", "descending") if monotonic == "auto" else (monotonic,)
                )
            }
            direction = min(fits, key=lambda d: _sse(rate, cnt, fits[d]))
            groups = fits[direction]
        blocks = [
            [
                int(cnt[s : e + 1].sum() - bad[s : e + 1].sum()),
                int(bad[s : e + 1].sum()),
            ]
            for s, e, _ in groups
        ]
        his = [fine_his[e] for _, e, _ in groups]
        blocks, his = _merge_blocks(blocks, his, min_count=min_count, max_bins=max_bins)
        edges = [h for h in his[:-1]]

    bins = [Bin(_label(edges, i), "range", g, b) for i, (g, b) in enumerate(blocks)]
    for kind, mask, label in (
        (
            "special",
            is_special,
            f"special {[int(s) if s == int(s) else s for s in specials]}",
        ),
        ("missing", is_missing, "missing"),
    ):
        if mask.any():
            bad_n = int(y[mask].sum())
            bins.append(Bin(label, kind, int(mask.sum()) - bad_n, bad_n))

    good = np.array([b.good for b in bins], float)
    bad_a = np.array([b.bad for b in bins], float)
    pg, pb = _distributions(good, bad_a)
    for b, w in zip(bins, np.log(pg / pb), strict=True):
        b.woe = float(w)
    return Binning(feature, edges, specials, bins, direction)
