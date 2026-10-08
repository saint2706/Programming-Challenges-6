"""Learned heads on frozen embeddings (three framings x linear/MLP) and the TF-IDF baselines.

* classification: 5 logits, cross-entropy;
* regression: a heteroscedastic Gaussian, ``mu`` and ``sigma = SIGMA_MIN + softplus(raw)``, trained
  on the Gaussian NLL (reported "up to a constant": the ``0.5 * log(2 pi)`` term is dropped);
* ordinal: one score and 4 increasing thresholds (cumulative link, proportional odds).

Linear heads are fit full-batch with L-BFGS, so each is the exact optimum of a smooth penalized
objective ``mean NLL + (l2 / 2) |W|^2`` (for classification that is scikit-learn's multinomial
logistic regression with ``C = 1 / (l2 * n)``, which a test checks). MLP heads (one hidden layer)
train with AdamW and keep the weights of their best validation epoch. Every head starts from the
marginal distribution of the labels, so a split with a single star value still trains.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from torch import nn

from review_stars.config import FRAMINGS, Config
from review_stars.probs import K, Pred, softmax

SIGMA_MIN = 0.05
CAPACITIES = ("linear", "mlp")
PREDICT_BATCH = 8192
_OUT_DIM = {"classification": K, "regression": 2, "ordinal": 1}


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-6)
    return x + float(np.log(-np.expm1(-x)))


class Net(nn.Module):
    def __init__(self, d: int, framing: str, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        self.framing, self.hidden = framing, hidden
        self.trunk = (
            nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
            if hidden
            else nn.Identity()
        )
        self.out = nn.Linear(hidden or d, _OUT_DIM[framing])
        if framing == "ordinal":
            self.raw_theta = nn.Parameter(torch.zeros(K - 1))
            self.out.bias.requires_grad_(
                False
            )  # the score's offset is the thresholds' job

    def forward(self, x):
        return self.out(self.trunk(x))

    def theta(self) -> torch.Tensor:
        steps = F.softplus(self.raw_theta[1:]) + 1e-3
        return torch.cat([self.raw_theta[:1], steps]).cumsum(0)


def _check_features(X: np.ndarray, what: str = "features") -> None:
    if not np.isfinite(X).all():
        raise ValueError(f"{what} contain non-finite values (NaN or inf)")


def _check_labels(framing: str, y: np.ndarray) -> None:
    if framing != "regression" and (
        y.min() < 1 or y.max() > K or not np.all(y == np.round(y))
    ):
        raise ValueError(
            f"star labels must be whole numbers between 1 and {K} for {framing}"
        )


def _tensors(framing: str, X: np.ndarray, y: np.ndarray):
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    if framing == "regression":
        return Xt, torch.from_numpy(np.asarray(y, dtype=np.float32))
    return Xt, torch.from_numpy(np.asarray(y, dtype=np.int64) - 1)


def nll_loss(net: Net, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of the framing's own model."""
    if net.framing == "classification":
        return F.cross_entropy(out, y)
    if net.framing == "regression":
        sigma = SIGMA_MIN + F.softplus(out[:, 1])
        return (torch.log(sigma) + 0.5 * ((y - out[:, 0]) / sigma) ** 2).mean()
    cum = torch.sigmoid(net.theta()[None, :] - out[:, :1])
    ones, zeros = torch.ones_like(cum[:, :1]), torch.zeros_like(cum[:, :1])
    full = torch.cat([zeros, cum, ones], dim=1)
    p = (full[:, 1:] - full[:, :-1]).gather(1, y[:, None]).squeeze(1)
    return -torch.log(p.clamp_min(1e-12)).mean()


def _init_from_labels(net: Net, framing: str, y: np.ndarray) -> None:
    """Start at the marginal label distribution: a one-star training set is then just a prior."""
    with torch.no_grad():
        net.out.weight.mul_(0.1 if net.hidden else 0.0)
        if framing == "classification":
            counts = np.bincount(np.asarray(y, dtype=np.int64) - 1, minlength=K) + 1.0
            net.out.bias.copy_(torch.from_numpy(np.log(counts / counts.sum())).float())
        elif framing == "regression":
            net.out.bias[0] = float(np.mean(y))
            net.out.bias[1] = _inv_softplus(float(np.std(y)) - SIGMA_MIN + 0.1)
        else:
            net.out.bias.zero_()
            cum = np.clip([(y <= k).mean() for k in range(1, K)], 0.01, 0.99)
            theta = np.maximum.accumulate(logit(cum) + np.arange(K - 1) * 1e-2)
            gaps = np.maximum(np.diff(theta) - 1e-3, 1e-3)
            raw = [theta[0]] + [_inv_softplus(g) for g in gaps]
            net.raw_theta.copy_(torch.tensor(raw, dtype=torch.float32))


@dataclass
class Head:
    framing: str
    capacity: str
    net: Net
    l2: float = 0.0
    dropout: float = 0.0
    history: dict = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> Pred:
        _check_features(X)
        self.net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), PREDICT_BATCH):
                xb = torch.from_numpy(
                    np.ascontiguousarray(X[i : i + PREDICT_BATCH], dtype=np.float32)
                )
                outs.append(self.net(xb))
            out = torch.cat(outs) if outs else torch.zeros((0, _OUT_DIM[self.framing]))
            theta = (
                self.net.theta().numpy().astype(np.float64)
                if self.framing == "ordinal"
                else None
            )
        out = out.numpy().astype(np.float64)
        if self.framing == "classification":
            return Pred("classification", logits=out)
        if self.framing == "regression":
            sigma = (
                SIGMA_MIN
                + np.log1p(np.exp(-np.abs(out[:, 1])))
                + np.maximum(out[:, 1], 0)
            )
            return Pred("regression", mu=out[:, 0], sigma=sigma)
        return Pred("ordinal", score=out[:, 0], theta=theta)

    def save(self, path) -> None:
        torch.save(
            {
                "framing": self.framing,
                "capacity": self.capacity,
                "d": int(
                    self.net.out.in_features
                    if not self.net.hidden
                    else self.net.trunk[0].in_features
                ),
                "hidden": self.net.hidden,
                "dropout": self.dropout,
                "l2": self.l2,
                "state": self.net.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path) -> Head:
        blob = torch.load(Path(path), weights_only=True)
        net = Net(blob["d"], blob["framing"], blob["hidden"], blob["dropout"])
        net.load_state_dict(blob["state"])
        return cls(
            blob["framing"], blob["capacity"], net.eval(), blob["l2"], blob["dropout"]
        )


def val_nll(head: Head, X: np.ndarray, y: np.ndarray) -> float:
    Xt, yt = _tensors(head.framing, X, y)
    head.net.eval()
    with torch.no_grad():
        return float(nll_loss(head.net, head.net(Xt), yt))


def _prepare(framing: str, X, y, Xv, yv):
    if framing not in FRAMINGS:
        raise ValueError(f"framing must be one of {FRAMINGS}, not {framing!r}")
    if len(X) == 0 or len(X) != len(y):
        raise ValueError(
            "training features and labels must be non-empty and the same length"
        )
    if len(Xv) == 0 or len(Xv) != len(yv):
        raise ValueError(
            "the validation set must be non-empty (it picks hyperparameters and epochs)"
        )
    for arr, what in ((X, "training features"), (Xv, "validation features")):
        _check_features(arr, what)
    _check_labels(framing, np.asarray(y))
    _check_labels(framing, np.asarray(yv))


def fit_linear(framing: str, X, y, Xv, yv, l2: float, max_iter: int = 400) -> Head:
    """Exact (L-BFGS) fit of a linear head with an L2 penalty ``l2`` on the weights."""
    _prepare(framing, X, y, Xv, yv)
    torch.manual_seed(0)
    net = Net(X.shape[1], framing)
    _init_from_labels(net, framing, np.asarray(y))
    Xt, yt = _tensors(framing, X, y)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=max_iter,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-12,
    )

    def closure():
        opt.zero_grad()
        loss = nll_loss(net, net(Xt), yt) + 0.5 * l2 * (net.out.weight**2).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("training diverged: the loss is not finite")
        loss.backward()
        return loss

    initial = float(opt.step(closure).detach())
    with torch.no_grad():
        final = float(nll_loss(net, net(Xt), yt) + 0.5 * l2 * (net.out.weight**2).sum())
    return Head(
        framing,
        "linear",
        net.eval(),
        l2=l2,
        history={"initial_loss": initial, "final_loss": final},
    )


def select_linear(framing: str, X, y, Xv, yv, grid, max_iter: int = 400):
    """Fit one head per ``l2`` in ``grid``; keep the one with the lowest validation NLL."""
    best, table = None, []
    for l2 in grid:
        head = fit_linear(framing, X, y, Xv, yv, l2=l2, max_iter=max_iter)
        v = val_nll(head, Xv, yv)
        table.append({"l2": float(l2), "val_nll": v})
        if best is None or v < best[0]:
            best = (v, head)
    return best[1], table


def fit_mlp(
    framing: str,
    X,
    y,
    Xv,
    yv,
    *,
    hidden: int,
    dropout: float,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    batch: int,
    weight_decay: float,
) -> Head:
    """One-hidden-layer head, AdamW, best-validation-epoch weights (early stopping)."""
    _prepare(framing, X, y, Xv, yv)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = Net(X.shape[1], framing, hidden, dropout)
    _init_from_labels(net, framing, np.asarray(y))
    Xt, yt = _tensors(framing, X, y)
    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    head = Head(
        framing, "mlp", net, dropout=dropout, history={"train_loss": [], "val_nll": []}
    )
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), batch):
            idx = order[i : i + batch]
            opt.zero_grad()
            loss = nll_loss(net, net(Xt[idx]), yt[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("training diverged: the loss is not finite")
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        v = val_nll(head, Xv, yv)
        head.history["train_loss"].append(total / len(Xt))
        head.history["val_nll"].append(v)
        if v < best - 1e-6:
            best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return head


def select_mlp(framing: str, X, y, Xv, yv, *, dropouts, hidden: int, seed: int, **kw):
    """Train one MLP per dropout rate (same seed); keep the lowest validation NLL."""
    best, table = None, []
    for p in dropouts:
        head = fit_mlp(framing, X, y, Xv, yv, hidden=hidden, dropout=p, seed=seed, **kw)
        v = min(head.history["val_nll"])
        table.append({"dropout": float(p), "val_nll": v})
        if best is None or v < best[0]:
            best = (v, head)
    return best[1], table


# ---------------------------------------------------------------- TF-IDF baselines


@dataclass
class TfidfBaseline:
    vectorizer: TfidfVectorizer
    clf: LogisticRegression | None
    ridge: Ridge
    sigma: float
    only_star: int | None
    report: dict

    def _logits(self, A) -> np.ndarray:
        n = A.shape[0]
        if (
            self.clf is None
        ):  # a single star in training: a constant (vanishing alternatives)
            logits = np.full((n, K), -30.0)
            logits[:, self.only_star - 1] = 0.0
            return logits
        dec = self.clf.decision_function(A)
        if dec.ndim == 1:  # binary problem: scikit-learn returns one column
            dec = np.stack([np.zeros(n), dec], axis=1)
        logits = np.full((n, K), -30.0)
        logits[:, self.clf.classes_ - 1] = dec
        return logits

    def predict(self, texts) -> dict[str, Pred]:
        A = self.vectorizer.transform(list(texts))
        mu = self.ridge.predict(A).astype(np.float64)
        return {
            "classification": Pred("classification", logits=self._logits(A)),
            "regression": Pred("regression", mu=mu, sigma=np.full(len(mu), self.sigma)),
        }


def fit_tfidf(texts, y, texts_val, y_val, cfg: Config) -> TfidfBaseline:
    """Word 1-2 gram TF-IDF with logistic regression (C by validation NLL) and ridge regression
    (alpha by validation RMSE) whose residual sigma is estimated on the validation set."""
    y, y_val = np.asarray(y), np.asarray(y_val)
    _check_labels("classification", y)
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=cfg.tfidf_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    A = vec.fit_transform(list(texts))
    Av = vec.transform(list(texts_val))
    report: dict = {"features": int(A.shape[1])}
    clf, only = None, None
    if len(np.unique(y)) < 2:
        only = int(y[0])
        report["c"] = None
    else:
        best = None
        for C in cfg.tfidf_c_grid:
            cand = LogisticRegression(C=C, max_iter=300).fit(A, y)
            tmp = TfidfBaseline(vec, cand, None, 0.0, None, {})
            P = softmax(tmp._logits(Av))
            v = float(
                -np.log(
                    np.clip(P[np.arange(len(y_val)), y_val - 1], 1e-12, None)
                ).mean()
            )
            if best is None or v < best[0]:
                best = (v, C, cand)
        report["val_nll"], report["c"], clf = best[0], best[1], best[2]
    best_r = None
    for alpha in cfg.ridge_alpha_grid:
        cand = Ridge(alpha=alpha).fit(A, y.astype(np.float64))
        rmse = float(np.sqrt(np.mean((cand.predict(Av) - y_val) ** 2)))
        if best_r is None or rmse < best_r[0]:
            best_r = (rmse, alpha, cand)
    report["val_rmse"], report["alpha"], ridge = best_r
    resid = ridge.predict(Av) - y_val
    sigma = max(float(np.std(resid)), SIGMA_MIN)
    report["sigma"] = sigma
    return TfidfBaseline(vec, clf, ridge, sigma, only, report)
