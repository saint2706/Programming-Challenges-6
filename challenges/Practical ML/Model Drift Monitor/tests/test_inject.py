import numpy as np
import pytest
from drift_monitor import inject, model
from drift_monitor.data import FEATURES
from helpers import synth

START = 600


def _df(n=1200):
    return synth(n, 0)


def test_covariate_shifts_only_the_chosen_feature_after_the_start():
    df = _df()
    out = inject.covariate(df, "vicdemand", 2.0, START)
    assert out[:START].equals(df[:START])
    std = df["vicdemand"].std()
    diff = (out["vicdemand"] - df["vicdemand"]).to_numpy()
    assert np.allclose(diff[START:], 2.0 * std) and np.allclose(diff[:START], 0)
    for col in (*[f for f in FEATURES if f != "vicdemand"], "label"):
        assert out[col].equals(df[col])


def test_covariate_rejects_a_categorical_feature():
    with pytest.raises(ValueError, match="numeric"):
        inject.covariate(_df(), "day", 1.0, START)


def test_prior_moves_the_label_rate_after_the_start_and_keeps_the_head_and_height():
    df = _df(3000)
    out = inject.prior(df, 0.8, 1000, seed=1)
    assert out.height == df.height and out[:1000].equals(df[:1000])
    assert out[1000:]["label"].mean() == pytest.approx(0.8, abs=0.001)
    assert inject.prior(df, 0.8, 1000, seed=1).equals(out)
    with pytest.raises(ValueError, match="rate"):
        inject.prior(df, 1.0, 1000)


def test_concept_flips_labels_only_after_the_start_and_leaves_features_alone():
    df = _df(4000)
    out = inject.concept(df, 0.3, 1000, seed=2)
    assert out[:1000].equals(df[:1000])
    assert out.select(FEATURES).equals(df.select(FEATURES))
    flipped = (out["label"] != df["label"]).to_numpy()
    assert not flipped[:1000].any()
    assert flipped[1000:].mean() == pytest.approx(0.3, abs=0.03)


def test_noisy_classifier_changes_only_rows_from_the_start_and_stays_a_probability():
    df = _df()
    clf = model.fit(df, seed=0)
    noisy = inject.NoisyClassifier(clf, sd=0.2, start_row=START, seed=3)
    x = df.select(FEATURES).to_numpy().astype(float)
    a, b = clf.predict_proba(x), noisy.predict_proba(x)
    np.testing.assert_allclose(a[:START], b[:START])
    assert np.abs(a[START:, 1] - b[START:, 1]).mean() > 0.05
    assert (b >= 0).all() and (b <= 1).all() and np.allclose(b.sum(axis=1), 1)
    np.testing.assert_allclose(
        b, inject.NoisyClassifier(clf, 0.2, START, 3).predict_proba(x)
    )
