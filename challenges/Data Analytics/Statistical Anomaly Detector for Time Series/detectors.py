"""Statistical anomaly detectors for a univariate time series (no ML, no training).

Every detector takes a 1-D float array and returns a `Detection`: a boolean
`flags` array, a per-point `scores` array, and the threshold that turned one
into the other (where such a threshold exists).

Point detectors (ignore time order):  zscore, iqr, mad, gesd
Windowed detectors (look back only):  rolling-z, rolling-mad
Decomposition detectors:              stl-mad, stl-z, stl-iqr, stl-gesd
                                      (detect on the residual after removing trend + seasonality)

Also here: automatic seasonal-period detection (periodogram candidates refined
and ranked by the autocorrelation function) and Rosner's Generalized ESD test,
implemented from the definition rather than wrapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

METHODS = (
    "zscore",
    "iqr",
    "mad",
    "gesd",
    "rolling-z",
    "rolling-mad",
    "stl-mad",
    "stl-z",
    "stl-iqr",
    "stl-gesd",
)
STL_METHODS = tuple(m for m in METHODS if m.startswith("stl-"))

# 1 / Phi^-1(0.75): rescales a MAD so it estimates the standard deviation of a
# normal distribution, and the constant in Iglewicz & Hoaglin's modified z-score.
_MAD_TO_Z = 0.6745
# Fallback scale when the MAD is 0 (more than half the points tie at the median):
# mean absolute deviation * 1.253314 estimates sigma for a normal (IBM/Iglewicz).
_MEANAD_TO_SIGMA = 1.253314


class DetectorError(ValueError):
    """The input can't be analysed (too short, wrong shape, missing period...)."""


@dataclass
class Params:
    """Tunable thresholds; the defaults are the textbook ones."""

    z_threshold: float = 3.0  # |z| > 3
    iqr_k: float = 1.5  # Tukey fences: Q1 - 1.5 IQR, Q3 + 1.5 IQR
    mad_threshold: float = 3.5  # Iglewicz & Hoaglin: |modified z| > 3.5
    window: int = 100  # rolling look-back, in points
    alpha: float = 0.05  # Generalized ESD significance level
    max_outliers: int | None = None  # Generalized ESD upper bound (None: 5% of n)
    # STL: statsmodels' default seasonal smoother (7) leaves a heavy-tailed
    # residual under robust=True (see README); 25 is calibrated much better.
    stl_seasonal: int = 25
    stl_robust: bool = True


@dataclass
class Detection:
    method: str
    flags: np.ndarray
    scores: np.ndarray
    threshold: float | None = None
    params: dict = field(default_factory=dict)
    # STL detectors also expose the decomposition, for plotting.
    components: dict[str, np.ndarray] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return int(self.flags.sum())


def _as_series(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise DetectorError("expected a 1-D series")
    if len(arr) < 3:
        raise DetectorError("need at least 3 points")
    if not np.all(np.isfinite(arr)):
        raise DetectorError("series contains NaN/inf; clean or drop them first")
    return arr


# ---------------------------------------------------------------- point scores
# Each `*_scores` helper returns a non-negative "how extreme" number per point
# in units where the classic threshold applies, so `flags = scores > threshold`.


def z_scores(x: np.ndarray) -> np.ndarray:
    """|x - mean| / sample std. Returns zeros for a constant series."""
    sd = x.std(ddof=1)
    if sd == 0:
        return np.zeros_like(x)
    return np.abs(x - x.mean()) / sd


def iqr_scores(x: np.ndarray) -> np.ndarray:
    """Distance outside the interquartile box, in IQR units (0 inside the box).

    Tukey's fence rule "x < Q1 - k*IQR or x > Q3 + k*IQR" is exactly
    `iqr_scores(x) > k`. Quartiles use numpy's default linear interpolation.
    """
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        # More than half the points tie in the middle: any deviation from the
        # tied value is infinitely many IQRs away, which is the honest reading.
        return np.where((x < q1) | (x > q3), np.inf, 0.0)
    return np.maximum(np.maximum(q1 - x, x - q3), 0.0) / iqr


def modified_z_scores(x: np.ndarray) -> np.ndarray:
    """Iglewicz-Hoaglin |0.6745 (x - median) / MAD|, with a mean-AD fallback."""
    med = np.median(x)
    dev = np.abs(x - med)
    mad = np.median(dev)
    if mad > 0:
        return _MAD_TO_Z * dev / mad
    mean_ad = dev.mean()
    if mean_ad == 0:
        return np.zeros_like(x)
    return dev / (_MEANAD_TO_SIGMA * mean_ad)


# ------------------------------------------------------------------------ GESD


@dataclass
class GesdResult:
    n_outliers: int
    indices: np.ndarray  # positions of the flagged points, most extreme first
    r: np.ndarray  # test statistic R_i for i = 1..max_outliers
    lam: np.ndarray  # critical value lambda_i for i = 1..max_outliers


def gesd_critical_values(n: int, max_outliers: int, alpha: float) -> np.ndarray:
    """Rosner (1983) critical values lambda_i, i = 1..max_outliers.

    lambda_i = (n-i) t_{p, n-i-1} / sqrt((n-i-1 + t^2)(n-i+1)),
    p = 1 - alpha / (2 (n-i+1)).
    """
    i = np.arange(1, max_outliers + 1)
    p = 1.0 - alpha / (2.0 * (n - i + 1))
    dof = n - i - 1
    t = stats.t.ppf(p, dof)
    return (n - i) * t / np.sqrt((dof + t**2) * (n - i + 1))


def generalized_esd(x, max_outliers: int, alpha: float = 0.05) -> GesdResult:
    """Rosner's Generalized Extreme Studentized Deviate test.

    Repeatedly removes the point farthest from the mean of what's left and
    records R_i = max|x - mean| / s. The number of outliers is the *largest* i
    with R_i > lambda_i, so a masked outlier (one that only looks extreme once
    its neighbours are gone) is still caught. Assumes the non-outliers are
    approximately normal; `max_outliers` only needs to be an upper bound.
    """
    x = _as_series(x)
    n = len(x)
    if max_outliers < 1:
        raise DetectorError("max_outliers must be >= 1")
    if max_outliers > n - 3:
        raise DetectorError(f"max_outliers must be <= n - 3 (n={n})")
    if not 0 < alpha < 1:
        raise DetectorError("alpha must be in (0, 1)")

    lam = gesd_critical_values(n, max_outliers, alpha)
    alive = np.ones(n, dtype=bool)
    removed: list[int] = []
    r = np.zeros(max_outliers)
    for i in range(max_outliers):
        rest = x[alive]
        sd = rest.std(ddof=1)
        if sd == 0:  # everything left is identical: nothing more to remove
            r[i:] = 0.0
            break
        dev = np.where(alive, np.abs(x - rest.mean()), -np.inf)
        j = int(np.argmax(dev))
        r[i] = dev[j] / sd
        removed.append(j)
        alive[j] = False
    exceeds = np.nonzero(r > lam)[0]
    n_out = int(exceeds[-1]) + 1 if len(exceeds) else 0
    return GesdResult(n_out, np.array(removed[:n_out], dtype=int), r, lam)


def _default_max_outliers(n: int, requested: int | None) -> int:
    k = requested if requested is not None else max(1, n // 20)
    return int(min(k, n - 3))


def _gesd_detection(x: np.ndarray, p: Params, method: str) -> Detection:
    if len(x) < 6:
        raise DetectorError("Generalized ESD needs at least 6 points")
    k = _default_max_outliers(len(x), p.max_outliers)
    res = generalized_esd(x, k, p.alpha)
    flags = np.zeros(len(x), dtype=bool)
    flags[res.indices] = True
    return Detection(
        method,
        flags,
        z_scores(x),  # plain z for plotting; the *decision* uses R_i > lambda_i
        threshold=None,
        params={"alpha": p.alpha, "max_outliers": k, "n_outliers": res.n_outliers},
        notes=["decision uses Rosner's R_i > lambda_i, not a fixed |z| cutoff"],
    )


# --------------------------------------------------------- point-detector API


def zscore(x, threshold: float = 3.0) -> Detection:
    x = _as_series(x)
    s = z_scores(x)
    return Detection("zscore", s > threshold, s, threshold, {"threshold": threshold})


def iqr(x, k: float = 1.5) -> Detection:
    x = _as_series(x)
    s = iqr_scores(x)
    return Detection("iqr", s > k, s, k, {"k": k})


def mad(x, threshold: float = 3.5) -> Detection:
    x = _as_series(x)
    s = modified_z_scores(x)
    return Detection("mad", s > threshold, s, threshold, {"threshold": threshold})


def gesd(x, params: Params | None = None) -> Detection:
    return _gesd_detection(_as_series(x), params or Params(), "gesd")


# -------------------------------------------------------------------- rolling


def rolling_scores(x: np.ndarray, window: int, robust: bool) -> np.ndarray:
    """Score each point against the `window` points *before* it (never itself).

    robust=False: |x - mean| / std of the window. robust=True: modified z from
    the window's median and MAD. The first `window` points have no full window
    and get score 0 (not flagged) rather than a noisy small-sample score.
    """
    n = len(x)
    if window < 3:
        raise DetectorError("window must be >= 3")
    if window >= n:
        raise DetectorError(f"window ({window}) must be smaller than the series ({n})")
    scores = np.zeros(n)
    # windows[t] = x[t : t + window]; the point being scored is x[t + window].
    win = sliding_window_view(x, window)[:-1]
    target = x[window:]
    if robust:
        med = np.median(win, axis=1)
        mad_ = np.median(np.abs(win - med[:, None]), axis=1)
        dev = np.abs(target - med)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(
                mad_ > 0, _MAD_TO_Z * dev / mad_, np.where(dev > 0, np.inf, 0.0)
            )
    else:
        mean = win.mean(axis=1)
        sd = win.std(axis=1, ddof=1)
        dev = np.abs(target - mean)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(sd > 0, dev / sd, np.where(dev > 0, np.inf, 0.0))
    scores[window:] = s
    return scores


def rolling_z(x, window: int = 100, threshold: float = 3.0) -> Detection:
    x = _as_series(x)
    s = rolling_scores(x, window, robust=False)
    return Detection(
        "rolling-z",
        s > threshold,
        s,
        threshold,
        {"window": window, "threshold": threshold},
    )


def rolling_mad(x, window: int = 100, threshold: float = 3.5) -> Detection:
    x = _as_series(x)
    s = rolling_scores(x, window, robust=True)
    return Detection(
        "rolling-mad",
        s > threshold,
        s,
        threshold,
        {"window": window, "threshold": threshold},
    )


# ------------------------------------------------------------------------ STL


def stl_decompose(
    x,
    periods: list[int] | int,
    *,
    robust: bool = True,
    seasonal: int = 25,
) -> dict[str, np.ndarray]:
    """STL (one period) or MSTL (several) -> trend / seasonal / resid.

    `robust=True` downweights points with large residuals in the loess fits, so
    the anomalies we're hunting don't bend the trend and seasonal components
    toward themselves and hide. It has a price: see the README on the shape of
    the residual it leaves behind. `seasonal` is the loess window (in cycles)
    of the seasonal smoother; larger means a more nearly periodic seasonal shape.
    """
    from statsmodels.tsa.seasonal import MSTL, STL

    x = _as_series(x)
    plist = [periods] if isinstance(periods, int) else list(periods)
    if not plist or any(p < 2 for p in plist):
        raise DetectorError("periods must be integers >= 2")
    if seasonal < 3 or seasonal % 2 == 0:
        raise DetectorError("STL seasonal window must be an odd integer >= 3")
    if len(x) < 2 * max(plist):
        raise DetectorError(
            f"series of {len(x)} points is shorter than two cycles of period {max(plist)}"
        )
    if len(plist) == 1:
        fit = STL(x, period=plist[0], seasonal=seasonal, robust=robust).fit()
        seas = np.asarray(fit.seasonal)
    else:
        # MSTL picks its own per-period seasonal windows; only `robust` is passed on.
        fit = MSTL(x, periods=sorted(plist), stl_kwargs={"robust": robust}).fit()
        seas = np.asarray(fit.seasonal)
        seas = seas.sum(axis=1) if seas.ndim == 2 else seas
    return {
        "trend": np.asarray(fit.trend),
        "seasonal": seas,
        "resid": np.asarray(fit.resid),
    }


def stl_detect(
    x,
    periods: list[int] | int,
    scorer: str = "mad",
    params: Params | None = None,
) -> Detection:
    """Decompose, then run a point detector on the residual.

    scorer: "mad" (modified z; the default because the residual of a robust STL
    still carries the anomalies, and MAD is the scale that ignores them),
    "z", "iqr" or "gesd".
    """
    p = params or Params()
    x = _as_series(x)
    comp = stl_decompose(x, periods, robust=p.stl_robust, seasonal=p.stl_seasonal)
    resid = comp["resid"]
    name = f"stl-{scorer}"
    if scorer == "gesd":
        det = _gesd_detection(resid, p, name)
    elif scorer == "mad":
        s = modified_z_scores(resid)
        det = Detection(name, s > p.mad_threshold, s, p.mad_threshold)
    elif scorer == "z":
        s = z_scores(resid)
        det = Detection(name, s > p.z_threshold, s, p.z_threshold)
    elif scorer == "iqr":
        s = iqr_scores(resid)
        det = Detection(name, s > p.iqr_k, s, p.iqr_k)
    else:
        raise DetectorError(f"unknown STL scorer {scorer!r}")
    det.components = comp
    det.params = {
        **det.params,
        "periods": [periods] if isinstance(periods, int) else list(periods),
        "robust": p.stl_robust,
        "seasonal": p.stl_seasonal,
    }
    return det


# ------------------------------------------------------------------- dispatch


def run_method(
    method: str,
    x,
    *,
    periods: list[int] | None = None,
    params: Params | None = None,
) -> Detection:
    """Run one named detector. STL detectors need `periods`."""
    p = params or Params()
    if method == "zscore":
        return zscore(x, p.z_threshold)
    if method == "iqr":
        return iqr(x, p.iqr_k)
    if method == "mad":
        return mad(x, p.mad_threshold)
    if method == "gesd":
        return gesd(x, p)
    if method == "rolling-z":
        return rolling_z(x, p.window, p.z_threshold)
    if method == "rolling-mad":
        return rolling_mad(x, p.window, p.mad_threshold)
    if method in STL_METHODS:
        if not periods:
            raise DetectorError(
                f"{method} needs a seasonal period (none given or detected)"
            )
        return stl_detect(x, periods, method.split("-", 1)[1], p)
    raise DetectorError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")


# --------------------------------------------------------- seasonal period


@dataclass
class PeriodEstimate:
    period: int
    acf: float  # autocorrelation at that lag of the detrended series
    candidates: list[tuple[int, float]]  # (period, acf) for every candidate tried


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocorrelation for lags 0..max_lag (via FFT, O(n log n))."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * np.conj(f), size)[: max_lag + 1]
    if ac[0] == 0:
        return np.zeros(max_lag + 1)
    return ac / ac[0]


def _winsorize(x: np.ndarray, k: float = 6.0) -> np.ndarray:
    """Clip to median +/- k robust sigmas so spikes don't dominate the spectrum."""
    med = np.median(x)
    scale = np.median(np.abs(x - med)) / _MAD_TO_Z
    if scale == 0:
        return x
    return np.clip(x, med - k * scale, med + k * scale)


def _comb(ac: np.ndarray, lag: int, max_multiples: int = 6) -> float:
    """Mean ACF at lag, 2*lag, 3*lag, ... (as many as fit, at most `max_multiples`)."""
    m = max(1, min(max_multiples, (len(ac) - 1) // lag))
    return float(np.mean(ac[lag * np.arange(1, m + 1)]))


def detect_period(
    x,
    min_period: int = 2,
    max_period: int | None = None,
    min_acf: float = 0.3,
    n_candidates: int = 6,
    min_prominence: float = 0.1,
) -> PeriodEstimate | None:
    """Find the dominant seasonal period, or None if there isn't one.

    1. Winsorize outliers and remove a linear trend.
    2. Take the strongest periodogram peaks as candidate periods (n / k for FFT
       bin k). The periodogram is coarse at long periods (n/k is not an
       integer), so
    3. each candidate is refined to the best *integer* lag within +/-10% by the
       autocorrelation function, which is exact for whole-number periods.
    4. Among candidates whose ACF is close to the best, the *smallest* period
       wins: a period-24 signal also peaks at lags 48, 72, ..., and the
       fundamental is the one we want.

    A candidate must be a local ACF maximum that rises at least `min_prominence`
    above the lowest ACF value before it, and clear `min_acf` and also 4/sqrt(n)
    (white noise produces ACF peaks of that size by chance); otherwise the
    series is called non-seasonal.
    """
    x = _as_series(x)
    n = len(x)
    max_period = max_period or n // 3  # want at least three full cycles
    if max_period < min_period + 1:
        return None
    t = np.arange(n)
    y = _winsorize(x)
    y = y - np.polyval(np.polyfit(t, y, 1), t)

    spec = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    ks = np.arange(len(spec))
    with np.errstate(divide="ignore"):
        periods = np.where(ks > 0, n / np.maximum(ks, 1), np.inf)
    ok = (ks > 0) & (periods >= min_period) & (periods <= max_period)
    # local maxima of the periodogram inside the allowed period range
    is_peak = np.zeros(len(spec), dtype=bool)
    is_peak[1:-1] = (spec[1:-1] > spec[:-2]) & (spec[1:-1] >= spec[2:])
    cand_bins = ks[ok & is_peak]
    if len(cand_bins) == 0:
        return None
    cand_bins = cand_bins[np.argsort(spec[cand_bins])[::-1][:n_candidates]]

    ac = autocorrelation(y, max(max_period + 1, n // 2))
    ac = ac * n / (n - np.arange(len(ac)))  # undo the biased estimator's (n-L)/n taper
    tried: dict[int, float] = {}
    for k in cand_bins:
        p0 = n / k
        lo = max(min_period, int(np.floor(p0 * 0.9)))
        hi = min(max_period, int(np.ceil(p0 * 1.1)))
        if hi < lo:
            continue
        # A smooth signal has high ACF at tiny lags simply because neighbours
        # resemble each other, which shows up as a maximum stuck to the edge of
        # the search window; only a hump strictly inside it is a period.
        hump = lo + int(np.argmax(ac[lo : hi + 1]))
        if hump in (lo, hi) and hump != max_period:
            continue
        # A long period's ACF peak is nearly flat (1 - cos(2 pi / P) ~ 2 pi^2 / P^2),
        # so noise moves the argmax by a lag or two, and STL is unforgiving of a
        # period that is off by one. Score each lag by the *comb* of ACF values at
        # its multiples instead: an error of 1 at lag L is an error of j at lag j*L.
        comb = np.array([_comb(ac, lag) for lag in range(lo, hi + 1)])
        lag = lo + int(np.argmax(comb))
        # ...and it has to *stand out*: a smooth but non-seasonal series (an AR(1)
        # process, a slow drift) has an ACF that only decays, so any wiggle in
        # it is a peak with no depth. A real cycle dips well below its own peak
        # on the way there (a sine's ACF goes negative at half a period).
        if ac[lag] - ac[1:lag].min() < min_prominence:
            continue
        tried[lag] = float(ac[lag])
    if not tried:
        return None
    # Neighbouring bins refine to neighbouring lags (47, 48, 50 for a true 48):
    # collapse each run to its best-ACF lag, or the smallest one would win by luck.
    peaks: dict[int, float] = {}
    run: list[int] = []
    for lag in sorted(tried):
        if run and lag - run[-1] > max(2, round(0.06 * lag)):
            best_lag = max(run, key=tried.__getitem__)
            peaks[best_lag] = tried[best_lag]
            run = []
        run.append(lag)
    best_lag = max(run, key=tried.__getitem__)
    peaks[best_lag] = tried[best_lag]

    cutoff = max(min_acf, 4.0 / np.sqrt(n))
    best = max(peaks.values())
    if best < cutoff:
        return None
    strong = sorted(p for p, a in peaks.items() if a >= 0.8 * best and a >= cutoff)
    period = strong[0]
    return PeriodEstimate(period, peaks[period], sorted(peaks.items()))
