import numpy as np
import pytest
from helpers import make_gaussian_data, make_ordinal_data, make_review_data, make_texts
from review_stars import heads, probs
from review_stars.config import Config
from review_stars.probs import Pred
from sklearn.linear_model import LogisticRegression

FAST = {"epochs": 12, "patience": 3, "lr": 3e-3, "batch": 256, "weight_decay": 1e-4}


def split(n=3000, seed=0, **kw):
    X, y = make_review_data(n=n, seed=seed, **kw)
    a, b = int(n * 0.6), int(n * 0.8)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def nll_of(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y - 1], 1e-12, None)).mean())


# ---------------------------------------------------------------- linear heads


@pytest.mark.parametrize("framing", ["classification", "regression", "ordinal"])
def test_a_linear_head_beats_the_prior_on_held_out_reviews(framing):
    (X, y), (Xv, yv), (Xt, yt) = split()
    head = heads.fit_linear(framing, X, y, Xv, yv, l2=1e-4)
    P = head.predict(Xt).proba()
    prior = np.bincount(y, minlength=6)[1:] / len(y)
    prior_nll = nll_of(np.tile(prior, (len(yt), 1)), yt)
    assert P.shape == (len(yt), 5) and np.allclose(P.sum(axis=1), 1.0)
    assert nll_of(P, yt) < prior_nll - 0.1
    assert head.history["final_loss"] < head.history["initial_loss"]


def test_the_linear_classification_head_matches_scikit_learns_logistic_regression():
    (X, y), (Xv, yv), (Xt, _) = split(n=2400)
    l2 = 1e-3
    ours = heads.fit_linear("classification", X, y, Xv, yv, l2=l2).predict(Xt).proba()
    # mean cross-entropy + (l2 / 2) |W|^2  <=>  sklearn C = 1 / (l2 * n)
    sk = LogisticRegression(C=1.0 / (l2 * len(y)), max_iter=2000, tol=1e-8).fit(X, y)
    theirs = sk.predict_proba(Xt)
    # measured: max 0.0002, mean 0.00003; a penalty mapping that is off by 2x moves max to ~0.07
    assert np.abs(ours - theirs).max() < 0.003 and np.abs(ours - theirs).mean() < 0.001


def test_the_linear_regression_head_recovers_a_known_noise_level_and_the_mean_function():
    X, y, w = make_gaussian_data(n=6000, sigma=0.5)
    head = heads.fit_linear(
        "regression", X[:4800], y[:4800], X[4800:], y[4800:], l2=1e-6
    )
    pred = head.predict(X[4800:])
    assert abs(float(pred.sigma.mean()) - 0.5) < 0.05
    assert float(np.abs(pred.mu - (3.0 + X[4800:] @ w)).mean()) < 0.05
    inside = np.abs(y[4800:] - pred.mu) <= pred.sigma
    assert 0.62 < inside.mean() < 0.74  # about 68% of a Gaussian lies within one sigma
    assert (pred.sigma >= heads.SIGMA_MIN).all()


def test_the_linear_ordinal_head_recovers_the_thresholds_of_a_proportional_odds_model():
    X, y, _, theta = make_ordinal_data(n=20_000)
    head = heads.fit_linear(
        "ordinal", X[:16_000], y[:16_000], X[16_000:], y[16_000:], l2=1e-7
    )
    pred = head.predict(X[16_000:])
    assert np.all(np.diff(pred.theta) > 0)
    assert np.abs(pred.theta - theta).max() < 0.08  # measured 0.046
    assert np.allclose(pred.proba().sum(axis=1), 1.0)


def test_l2_selection_picks_the_value_with_the_lowest_validation_nll():
    (X, y), (Xv, yv), _ = split(n=1200, signal=1.2)
    head, table = heads.select_linear(
        "classification", X, y, Xv, yv, grid=(1e-6, 1e-3, 1e2)
    )
    best = min(table, key=lambda r: r["val_nll"])
    assert head.l2 == best["l2"] and len(table) == 3
    assert head.l2 != 1e2  # an absurdly strong penalty cannot win


# ---------------------------------------------------------------- MLP heads


@pytest.mark.parametrize("framing", ["classification", "regression", "ordinal"])
def test_an_mlp_head_learns_and_returns_its_best_validation_epoch(framing):
    (X, y), (Xv, yv), (Xt, yt) = split()
    head = heads.fit_mlp(framing, X, y, Xv, yv, hidden=32, dropout=0.1, seed=0, **FAST)
    hist = head.history["val_nll"]
    assert len(hist) >= 2 and min(hist) < hist[0]
    # early stopping keeps the best weights, not the last ones
    assert abs(heads.val_nll(head, Xv, yv) - min(hist)) < 1e-4
    P = head.predict(Xt).proba()
    prior = np.bincount(y, minlength=6)[1:] / len(y)
    assert nll_of(P, yt) < nll_of(np.tile(prior, (len(yt), 1)), yt) - 0.05


def test_an_mlp_is_reproducible_for_a_seed_and_differs_across_seeds():
    (X, y), (Xv, yv), (Xt, _) = split(n=900)
    kw = {"hidden": 16, "dropout": 0.1, **FAST, "epochs": 4}
    a = heads.fit_mlp("classification", X, y, Xv, yv, seed=3, **kw).predict(Xt).logits
    b = heads.fit_mlp("classification", X, y, Xv, yv, seed=3, **kw).predict(Xt).logits
    c = heads.fit_mlp("classification", X, y, Xv, yv, seed=4, **kw).predict(Xt).logits
    assert np.allclose(a, b, atol=1e-5) and not np.allclose(a, c, atol=1e-3)


def test_dropout_is_chosen_on_validation_nll_and_never_active_at_prediction_time():
    (X, y), (Xv, yv), (Xt, _) = split(n=1200)
    head, table = heads.select_mlp(
        "classification", X, y, Xv, yv, dropouts=(0.0, 0.5), hidden=16, seed=0, **FAST
    )
    assert {r["dropout"] for r in table} == {0.0, 0.5}
    assert head.dropout == min(table, key=lambda r: r["val_nll"])["dropout"]
    assert np.array_equal(
        head.predict(Xt).logits, head.predict(Xt).logits
    )  # eval mode: no noise


# ---------------------------------------------------------------- awkward inputs


@pytest.mark.parametrize("framing", ["classification", "regression", "ordinal"])
@pytest.mark.parametrize("capacity", ["linear", "mlp"])
def test_a_training_split_with_a_single_star_value_still_trains_and_predicts(
    framing, capacity
):
    (X, _), (Xv, yv), (Xt, _) = split(n=600)
    y = np.full(len(X), 5, dtype=np.int64)  # every review is five stars
    if capacity == "linear":
        head = heads.fit_linear(framing, X, y, Xv, yv, l2=1e-3)
    else:
        head = heads.fit_mlp(
            framing, X, y, Xv, yv, hidden=8, dropout=0.0, seed=0, **FAST
        )
    P = head.predict(Xt).proba()
    assert np.isfinite(P).all() and np.allclose(P.sum(axis=1), 1.0)
    assert P[:, 4].mean() > 0.5  # it learned that five stars is overwhelmingly likely


def test_missing_stars_in_the_training_set_do_not_break_the_classifier():
    (X, y), (Xv, yv), (Xt, _) = split(n=900)
    keep = y <= 3
    head = heads.fit_linear("classification", X[keep], y[keep], Xv, yv, l2=1e-3)
    P = head.predict(Xt).proba()
    assert np.isfinite(P).all() and np.allclose(P.sum(axis=1), 1.0)
    assert (
        P[:, 4].mean() < 0.2
    )  # stars never seen get little mass (the prior smoothing, no more)


def test_non_finite_features_are_rejected_before_training():
    (X, y), (Xv, yv), _ = split(n=300)
    X = X.copy()
    X[3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        heads.fit_linear("classification", X, y, Xv, yv, l2=1e-3)
    with pytest.raises(ValueError, match="non-finite"):
        heads.fit_mlp(
            "classification", X, y, Xv, yv, hidden=4, dropout=0.0, seed=0, **FAST
        )


def test_labels_outside_one_to_five_are_rejected():
    (X, y), (Xv, yv), _ = split(n=300)
    bad = y.copy()
    bad[0] = 6
    with pytest.raises(ValueError, match="1 and 5"):
        heads.fit_linear("classification", X, bad, Xv, yv, l2=1e-3)


def test_an_empty_validation_set_is_an_error():
    (X, y), _, _ = split(n=300)
    with pytest.raises(ValueError, match="validation"):
        heads.fit_mlp(
            "classification", X, y, X[:0], y[:0], hidden=4, dropout=0.0, seed=0, **FAST
        )


def test_a_head_round_trips_through_a_file_without_pickle(tmp_path):
    (X, y), (Xv, yv), (Xt, _) = split(n=600)
    for framing, capacity in (("ordinal", "linear"), ("regression", "mlp")):
        head = (
            heads.fit_linear(framing, X, y, Xv, yv, l2=1e-3)
            if capacity == "linear"
            else heads.fit_mlp(
                framing, X, y, Xv, yv, hidden=8, dropout=0.1, seed=1, **FAST
            )
        )
        path = tmp_path / f"{framing}-{capacity}.pt"
        head.save(path)
        again = heads.Head.load(path)
        assert np.allclose(
            again.predict(Xt).proba(), head.predict(Xt).proba(), atol=1e-6
        )
        assert (again.framing, again.capacity) == (framing, capacity)


def test_predicting_in_batches_gives_the_same_answer_as_all_at_once():
    (X, y), (Xv, yv), (Xt, _) = split(n=900)
    head = heads.fit_linear("regression", X, y, Xv, yv, l2=1e-3)
    big = np.concatenate([Xt] * 70)  # more rows than one prediction batch
    assert len(big) > heads.PREDICT_BATCH
    assert np.allclose(head.predict(big).mu[: len(Xt)], head.predict(Xt).mu, atol=1e-5)


# ---------------------------------------------------------------- TF-IDF baselines


def test_the_tfidf_baselines_learn_sentiment_words():
    texts, stars = make_texts(900)
    cfg = Config(
        tfidf_c_grid=(1.0, 8.0), ridge_alpha_grid=(0.3, 3.0), tfidf_features=2000
    )
    base = heads.fit_tfidf(
        texts[:600], stars[:600], texts[600:750], stars[600:750], cfg
    )
    out = base.predict(texts[750:])
    P = out["classification"].proba()
    assert set(out) == {"classification", "regression"}
    assert (probs.predicted_star(P) == stars[750:]).mean() > 0.4  # chance is about 0.2
    reg = out["regression"]
    assert np.corrcoef(reg.mu, stars[750:])[0, 1] > 0.7
    assert np.allclose(reg.sigma, reg.sigma[0]) and reg.sigma[0] >= heads.SIGMA_MIN
    assert base.report["c"] in (1.0, 8.0) and base.report["alpha"] in (0.3, 3.0)


def test_the_tfidf_classifier_gives_unseen_stars_a_vanishing_probability():
    texts, stars = make_texts(600)
    keep = stars <= 3
    cfg = Config(tfidf_c_grid=(1.0,), ridge_alpha_grid=(1.0,), tfidf_features=2000)
    kept = [t for t, k in zip(texts, keep, strict=True) if k]
    base = heads.fit_tfidf(kept, stars[keep], texts[:100], stars[:100], cfg)
    P = base.predict(texts[:50])["classification"].proba()
    assert P.shape == (50, 5) and np.allclose(P.sum(axis=1), 1.0)
    assert P[:, 3:].max() < 1e-6


def test_tfidf_copes_with_empty_and_non_english_reviews():
    texts, stars = make_texts(300)
    cfg = Config(tfidf_c_grid=(1.0,), ridge_alpha_grid=(1.0,), tfidf_features=500)
    base = heads.fit_tfidf(
        texts[:200], stars[:200], texts[200:250], stars[200:250], cfg
    )
    out = base.predict(["", "   ", "非常好", "\U0001f44d"])
    for pred in out.values():
        assert np.isfinite(pred.proba()).all() and len(pred) == 4


def test_pred_objects_are_the_shared_currency():
    (X, y), (Xv, yv), (Xt, _) = split(n=600)
    pred = heads.fit_linear("classification", X, y, Xv, yv, l2=1e-3).predict(Xt)
    assert isinstance(pred, Pred) and pred.logits.dtype == np.float64
