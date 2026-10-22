"""Cox proportional hazards regression, written from the partial likelihood.

The hazard of subject i is ``h0(t) exp(x_i . beta)`` and ``beta`` maximises the partial likelihood,
a product over event times of "this subject failed, given who was at risk". Features:

* **ties**: Efron (default; what R and most packages use) or Breslow;
* **late entry**: the risk set at ``t`` is ``entry < t <= time``;
* **strata**: a separate baseline hazard per stratum, one shared ``beta``;
* **ridge penalty** on standardised coefficients, for collinear or sparse designs;
* Schoenfeld residuals and the Grambsch-Therneau test of the proportional-hazards assumption;
* Breslow baseline hazard, survival curves for new covariates, Harrell's concordance.

Optimisation is Newton-Raphson with step halving on standardised covariates. Risk-set sums come
from cumulative sums over subjects sorted by exit time (minus those who enter later), so a fit costs
``O(n p^2)`` per iteration plus ``O(K p^2)`` memory for ``K`` events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, stats

from survival_toolkit.kaplan_meier import fit as fit_km
from survival_toolkit.types import Survival, SurvivalError

TIES = ("efron", "breslow")
TRANSFORMS = ("km", "rank", "identity", "log")


@dataclass
class _Block:
    """Index bookkeeping for one stratum: who is at risk when, and which events tie."""

    idx: np.ndarray  # subject indices (into the full data) in this stratum
    time_order: np.ndarray  # positions within idx sorted by exit time
    time_pos: (
        np.ndarray
    )  # per distinct event time: first sorted position with time >= u
    entry_order: np.ndarray | None
    entry_pos: np.ndarray | None
    times: np.ndarray  # distinct event times u (ascending)
    death_idx: np.ndarray  # subject indices of the events, ordered by time
    death_local: np.ndarray  # the same as positions within idx
    group: np.ndarray  # per event: index of its distinct time
    start: (
        np.ndarray
    )  # per distinct time: offset of its first event in the event arrays
    count: np.ndarray  # per distinct time: number of events
    frac: np.ndarray  # per event: Efron fraction l/d (zeros for Breslow)


def _block(data: Survival, idx: np.ndarray, efron: bool) -> _Block | None:
    t = data.time[idx]
    ev = data.event[idx]
    if not ev.any():
        return None
    order = np.argsort(t, kind="stable")
    times = np.unique(t[ev])
    time_pos = np.searchsorted(t[order], times, side="left")
    entry_order = entry_pos = None
    if data.entry is not None:
        e = data.entry[idx]
        entry_order = np.argsort(e, kind="stable")
        entry_pos = np.searchsorted(e[entry_order], times, side="left")
    deaths = np.flatnonzero(ev)
    deaths = deaths[np.argsort(t[deaths], kind="stable")]
    group = np.searchsorted(times, t[deaths])
    count = np.bincount(group, minlength=times.size)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    within = np.arange(deaths.size) - start[group]
    frac = within / count[group] if efron else np.zeros(deaths.size)
    return _Block(
        idx, order, time_pos, entry_order, entry_pos, times, idx[deaths], deaths, group, start,
        count, frac,
    )  # fmt: skip


def _reverse_cumsum(a: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Sum of ``a[p:]`` for each p in ``positions`` (0 past the end)."""
    rc = np.cumsum(a[::-1], axis=0)[::-1]
    pad = np.zeros((1, *a.shape[1:]))
    return np.concatenate([rc, pad])[positions]


@dataclass(frozen=True)
class CoxModel:
    """A fitted model; coefficients and covariance are on the original covariate scale."""

    names: list[str]
    beta: np.ndarray
    cov: np.ndarray
    loglik: float
    loglik_null: float
    score_statistic: float
    ties: str
    penalizer: float
    n: int
    n_events: int
    iterations: int
    converged: bool
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    data: Survival
    x: np.ndarray
    strata: np.ndarray
    _blocks: list = field(repr=False, default_factory=list)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def z(self) -> np.ndarray:
        return self.beta / self.se

    @property
    def p(self) -> np.ndarray:
        return 2 * stats.norm.sf(np.abs(self.z))

    @property
    def hazard_ratio(self) -> np.ndarray:
        return np.exp(self.beta)

    def hazard_ratio_ci(self) -> tuple[np.ndarray, np.ndarray]:
        half = stats.norm.ppf(1 - self.alpha / 2) * self.se
        return np.exp(self.beta - half), np.exp(self.beta + half)

    def likelihood_ratio(self) -> tuple[float, int, float]:
        stat = 2 * (self.loglik - self.loglik_null)
        df = len(self.beta)
        return stat, df, float(stats.chi2.sf(stat, df))

    def wald(self) -> tuple[float, int, float]:
        stat = float(self.beta @ np.linalg.solve(self.cov, self.beta))
        return stat, len(self.beta), float(stats.chi2.sf(stat, len(self.beta)))

    def score(self) -> tuple[float, int, float]:
        df = len(self.beta)
        return self.score_statistic, df, float(stats.chi2.sf(self.score_statistic, df))

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * len(self.beta)

    def linear_predictor(self, x: np.ndarray | None = None) -> np.ndarray:
        x = self.x if x is None else np.atleast_2d(x)
        return (x - self.mean) @ self.beta

    def summary(self) -> list[dict]:
        lo, hi = self.hazard_ratio_ci()
        return [
            {
                "name": n,
                "coef": float(b),
                "hr": float(h),
                "se": float(s),
                "z": float(z),
                "p": float(p),
                "hr_lower": float(a),
                "hr_upper": float(c),
            }
            for n, b, h, s, z, p, a, c in zip(
                self.names, self.beta, self.hazard_ratio, self.se, self.z, self.p, lo, hi,
                strict=True,
            )
        ]  # fmt: skip

    def concordance(self) -> dict:
        """Harrell's C within strata: of the comparable pairs, how many does the model order right.

        A pair is comparable when the earlier time is an event (an event beats a censoring at the
        same time). Equal risk scores count one half.
        """
        risk = self.linear_predictor()
        d = self.data
        concordant = tied = total = 0.0
        for stratum in np.unique(self.strata):
            idx = np.flatnonzero(self.strata == stratum)
            t, ev, r = d.time[idx], d.event[idx], risk[idx]
            events = np.flatnonzero(ev)
            for lo in range(0, events.size, 512):
                chunk = events[lo : lo + 512]
                later = (t[None, :] > t[chunk, None]) | (
                    (t[None, :] == t[chunk, None]) & ~ev[None, :]
                )
                diff = r[chunk, None] - r[None, :]
                total += later.sum()
                concordant += (later & (diff > 0)).sum()
                tied += (later & (diff == 0)).sum()
        c = (concordant + 0.5 * tied) / total if total else float("nan")
        return {
            "c": float(c),
            "concordant": int(concordant),
            "tied": int(tied),
            "pairs": int(total),
        }

    def baseline_hazard(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Cumulative baseline hazard per stratum: (event times, H0 at those times).

        The baseline is the hazard of a subject at the covariate means. Breslow's estimator adds
        ``d / S0`` at each event time; with Efron ties each tied event gets its own adjusted risk
        set, ``sum_l 1 / (S0 - l/d D0)``, which is what R's ``survfit`` uses.
        """
        z = (self.x - self.mean) / self.scale
        eta = z @ (self.beta * self.scale)
        theta = np.exp(eta - eta.max())
        scale = np.exp(-eta.max())  # theta above is the true risk times this
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for blk in self._blocks:
            s0, s1, d0, d1 = _sums(blk, theta[blk.idx], z[blk.idx], hessian=False)[:4]
            a0, _ = _event_terms(blk, s0, s1, d0, d1)
            per_event = 1.0 / a0 if self.ties == "efron" else None
            if per_event is None:
                increments = blk.count / s0
            else:
                increments = np.add.reduceat(per_event, blk.start)
            out[int(self.strata[blk.idx[0]])] = (
                blk.times,
                np.cumsum(increments * scale),
            )
        return out

    def predict_survival(
        self, x: np.ndarray, times: np.ndarray, *, stratum: int = 0
    ) -> np.ndarray:
        """S(t | x) = exp(-H0(t) exp(x . beta)), one row per covariate row, one column per time."""
        base = self.baseline_hazard()
        if stratum not in base:
            raise SurvivalError(
                f"stratum {stratum} has no events, so it has no baseline hazard"
            )
        u, h0 = base[stratum]
        idx = np.searchsorted(u, np.asarray(times, dtype=float), side="right") - 1
        h = np.where(idx < 0, 0.0, h0[np.maximum(idx, 0)])
        return np.exp(-np.outer(np.exp(self.linear_predictor(x)), h))

    def schoenfeld(self) -> dict:
        """Schoenfeld residuals: each event's covariates minus the risk-set average at its time.

        With Efron ties the average is taken over the tied events' own Efron weights. They sum
        to zero at the fitted coefficients, which is the likelihood's score equation.
        """
        z = (self.x - self.mean) / self.scale
        zb = z @ (self.beta * self.scale)
        eta = zb - zb.max()
        theta = np.exp(eta)
        res, times, strata = [], [], []
        for blk in self._blocks:
            th = theta[blk.idx]
            zz = z[blk.idx]
            s0, s1, d0, d1 = _sums(blk, th, zz, hessian=False)[:4]
            a0, a1 = _event_terms(blk, s0, s1, d0, d1)
            expect = a1 / a0[:, None]
            # average over each tie group (identical for Breslow)
            mean = np.add.reduceat(expect, blk.start, axis=0) / blk.count[:, None]
            r = z[blk.death_idx] - mean[blk.group]
            res.append(r * self.scale)
            times.append(blk.times[blk.group])
            strata.append(self.strata[blk.death_idx])
        return {
            "residuals": np.vstack(res),
            "time": np.concatenate(times),
            "stratum": np.concatenate(strata),
        }

    def scaled_schoenfeld(self) -> tuple[np.ndarray, np.ndarray]:
        """Event times and scaled Schoenfeld residuals ``beta + d V r`` (events x covariates).

        Plotted against time, a flat band around ``beta`` is what proportional hazards looks like;
        a slope is a coefficient that changes over time.
        """
        sch = self.schoenfeld()
        r = sch["residuals"]
        return sch["time"], self.beta + r.shape[0] * (r @ self.cov)

    def proportional_hazards_test(self, transform: str = "km") -> dict:
        """Score test that each coefficient is constant in time (R's ``cox.zph``).

        Each covariate is interacted with ``g(t)`` and the test asks whether that interaction
        coefficient is zero, evaluated at the fitted model (so no refit). A significant covariate has
        an effect that grows or fades. ``g`` is ``km`` (1 - Kaplan-Meier, R's default), ``rank``,
        ``identity`` or ``log``. The global test adds all interactions at once.
        """
        if transform not in TRANSFORMS:
            raise SurvivalError(f"transform must be one of {', '.join(TRANSFORMS)}")
        t, var, resid = self._event_pieces()
        g = _transform(transform, self.data, t)
        g = g - g.mean()
        u = np.einsum("k,kp->p", g, resid)
        i_bb = var.sum(axis=0)
        i_bg = np.einsum("k,kpq->pq", g, var)
        i_gg = np.einsum("k,kpq->pq", g * g, var)
        # score test for the interaction block, with beta profiled out: u' (Igg - Igb Ibb^-1 Ibg)^-1 u
        schur = i_gg - i_bg @ np.linalg.solve(i_bb, i_bg)
        p = len(self.beta)
        terms = []
        for j, name in enumerate(self.names):
            chi2 = float(u[j] ** 2 / schur[j, j])
            terms.append(
                {"name": name, "chi2": chi2, "p": float(stats.chi2.sf(chi2, 1))}
            )
        glob = float(u @ np.linalg.solve(schur, u))
        return {
            "transform": transform,
            "terms": terms,
            "global": {"chi2": glob, "df": p, "p": float(stats.chi2.sf(glob, p))},
        }

    def _event_pieces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per event: its time, the risk-set covariance of the covariates, and x - E[x | risk set]."""
        z = (self.x - self.mean) / self.scale
        eta = z @ (self.beta * self.scale)
        theta = np.exp(eta - eta.max())
        times, variances, resid = [], [], []
        for blk in self._blocks:
            th, zz = theta[blk.idx], z[blk.idx]
            s0, s1, d0, d1, s2, d2 = _sums(blk, th, zz, hessian=True)
            a0, a1, a2 = _event_terms(blk, s0, s1, d0, d1, s2, d2)
            m = a1 / a0[:, None]
            variances.append(a2 / a0[:, None, None] - m[:, :, None] * m[:, None, :])
            resid.append(z[blk.death_idx] - m)
            times.append(blk.times[blk.group])
        return np.concatenate(times), np.concatenate(variances), np.vstack(resid)


def _transform(name: str, data: Survival, t: np.ndarray) -> np.ndarray:
    if name == "identity":
        return t
    if name == "log":
        return np.log(t)
    if name == "rank":  # ranks among every subject's exit time, censored ones included
        uniq, inverse = np.unique(data.time, return_inverse=True)
        per_time = np.zeros(uniq.size)
        per_time[inverse] = stats.rankdata(data.time, method="average")
        return per_time[np.searchsorted(uniq, t)]
    # km: 1 - S(t-) over the pooled sample, ignoring strata as R does
    km = fit_km(data)
    idx = np.searchsorted(km.time, t, side="left") - 1
    return 1.0 - np.where(idx < 0, 1.0, km.survival[np.maximum(idx, 0)])


def _sums(blk: _Block, th: np.ndarray, z: np.ndarray, *, hessian: bool):
    """Risk-set sums S0, S1, S2 and tied-event sums D0, D1, D2 at each distinct event time."""
    order, pos = blk.time_order, blk.time_pos
    s0 = _reverse_cumsum(th[order], pos)
    s1 = _reverse_cumsum((th[:, None] * z)[order], pos)
    s2 = None
    if hessian:
        outer = th[:, None, None] * z[:, :, None] * z[:, None, :]
        s2 = _reverse_cumsum(outer[order], pos)
    if blk.entry_order is not None:
        eo, ep = blk.entry_order, blk.entry_pos
        s0 = s0 - _reverse_cumsum(th[eo], ep)
        s1 = s1 - _reverse_cumsum((th[:, None] * z)[eo], ep)
        if hessian:
            s2 = s2 - _reverse_cumsum(outer[eo], ep)
    dl = blk.death_local
    thd, zd = th[dl], z[dl]
    d0 = np.add.reduceat(thd, blk.start)
    d1 = np.add.reduceat(thd[:, None] * zd, blk.start, axis=0)
    d2 = None
    if hessian:
        d2 = np.add.reduceat(
            thd[:, None, None] * zd[:, :, None] * zd[:, None, :], blk.start, axis=0
        )
    return s0, s1, d0, d1, s2, d2


def _event_terms(blk, s0, s1, d0, d1, s2=None, d2=None):
    """Per event: the Efron-adjusted risk-set sums A0, A1 (and A2)."""
    g, f = blk.group, blk.frac
    a0 = s0[g] - f * d0[g]
    a1 = s1[g] - f[:, None] * d1[g]
    if s2 is None:
        return a0, a1
    return a0, a1, s2[g] - f[:, None, None] * d2[g]


def _evaluate(blocks, z, beta, penalizer, *, hessian=True):
    """Penalised log partial likelihood, its gradient and the information matrix."""
    eta = z @ beta
    eta = eta - eta.max()  # cancels in every ratio and in the likelihood
    theta = np.exp(eta)
    p = beta.size
    ll = 0.0
    grad = np.zeros(p)
    info = np.zeros((p, p))
    for blk in blocks:
        th, zz = theta[blk.idx], z[blk.idx]
        s0, s1, d0, d1, s2, d2 = _sums(blk, th, zz, hessian=hessian)
        if hessian:
            a0, a1, a2 = _event_terms(blk, s0, s1, d0, d1, s2, d2)
        else:
            a0, a1 = _event_terms(blk, s0, s1, d0, d1)
        zd = z[blk.death_idx]
        ll += float(eta[blk.death_idx].sum() - np.log(a0).sum())
        m = a1 / a0[:, None]
        grad += (zd - m).sum(axis=0)
        if hessian:
            info += (a2 / a0[:, None, None]).sum(axis=0) - m.T @ m
    ll -= 0.5 * penalizer * float(beta @ beta)
    grad -= penalizer * beta
    info += penalizer * np.eye(p)
    return ll, grad, info


def fit(
    data: Survival,
    x: np.ndarray,
    names: list[str] | None = None,
    *,
    strata: np.ndarray | None = None,
    ties: str = "efron",
    penalizer: float = 0.0,
    alpha: float = 0.05,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> CoxModel:
    """Fit a Cox model by Newton-Raphson on the (penalised) partial likelihood."""
    if ties not in TIES:
        raise SurvivalError(f"ties must be one of {', '.join(TIES)}")
    if penalizer < 0:
        raise SurvivalError("penalizer must be >= 0")
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != len(data):
        raise SurvivalError("x needs one row per subject")
    if not np.all(np.isfinite(x)):
        raise SurvivalError(
            "covariates have missing or infinite values; drop those rows first"
        )
    p = x.shape[1]
    names = names or [f"x{i + 1}" for i in range(p)]
    if len(names) != p:
        raise SurvivalError("names needs one entry per covariate column")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    flat = [n for n, s in zip(names, scale, strict=True) if s == 0]
    if flat:
        raise SurvivalError(
            f"covariate {flat[0]!r} is constant, so its effect cannot be estimated"
        )
    z = (x - mean) / scale
    if np.linalg.matrix_rank(z) < p:
        _, r, piv = linalg.qr(z, mode="economic", pivoting=True)
        rank = int(np.sum(np.abs(np.diag(r)) > 1e-10 * abs(r[0, 0])))
        redundant = ", ".join(repr(names[i]) for i in sorted(piv[rank:]))
        raise SurvivalError(
            f"covariates are collinear (rank {rank} of {p}); drop or merge: {redundant}"
        )

    codes = (
        np.zeros(len(data), dtype=int)
        if strata is None
        else np.unique(np.asarray(strata).astype(str), return_inverse=True)[1]
    )
    blocks = [
        b
        for s in np.unique(codes)
        if (b := _block(data, np.flatnonzero(codes == s), ties == "efron")) is not None
    ]
    if not blocks:
        raise SurvivalError("no events were observed, so nothing can be estimated")

    beta = np.zeros(p)
    ll0, grad0, info0 = _evaluate(blocks, z, beta, penalizer)
    try:
        score = float(grad0 @ np.linalg.solve(info0, grad0))
    except np.linalg.LinAlgError as exc:
        raise SurvivalError(
            "the information matrix is singular: collinear covariates?"
        ) from exc
    ll, grad, info = ll0, grad0, info0
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError as exc:
            raise SurvivalError(
                "the information matrix is singular: collinear covariates?"
            ) from exc
        scale_step = 1.0
        for _ in range(30):
            with np.errstate(over="ignore", invalid="ignore"):
                cand = _evaluate(blocks, z, beta + scale_step * step, penalizer)
            if np.isfinite(cand[0]) and cand[0] >= ll - 1e-12:
                break
            scale_step /= 2
        else:
            break  # no step improves the likelihood: stuck at (numerical) optimum
        beta = beta + scale_step * step
        gain = cand[0] - ll
        ll, grad, info = cand
        if (
            abs(gain) < tol * (abs(ll) + 0.1)
            and np.max(np.abs(scale_step * step)) < 1e-6
        ):
            converged = True
            break
    if not converged and np.max(np.abs(grad)) < 1e-6 * max(1.0, np.max(np.abs(info))):
        converged = True
    cov_z = np.linalg.inv(info)
    beta_x = beta / scale
    cov_x = cov_z / np.outer(scale, scale)
    return CoxModel(
        names=names,
        beta=beta_x,
        cov=cov_x,
        loglik=float(ll),
        loglik_null=float(ll0),
        score_statistic=score,
        ties=ties,
        penalizer=penalizer,
        n=len(data),
        n_events=data.n_events,
        iterations=iterations,
        converged=converged,
        alpha=alpha,
        mean=mean,
        scale=scale,
        data=data,
        x=x,
        strata=codes,
        _blocks=blocks,
    )
