import numpy as np
import pytest
from helpers import synthetic
from inbox_sorter.explain import READABLE, contributions, group_contributions, reasons
from inbox_sorter.models import fit_models, time_split


@pytest.fixture(scope="module")
def setup():
    df = synthetic(900)
    train, val, test = time_split(df)
    models = fit_models(train, val, seed=0)
    x = models.matrix(test, with_text=False)
    return models, test, x


def test_contributions_plus_base_value_reproduce_the_raw_margin(setup):
    models, _, x = setup
    contribs, base = contributions(models.lgbm_meta, x)
    margin = models.lgbm_meta.booster_.predict(x, raw_score=True)
    np.testing.assert_allclose(contribs.sum(axis=1) + base, margin, atol=1e-5)


def test_the_planted_feature_is_the_top_reason_for_a_high_scoring_message(setup):
    models, test, x = setup
    scores = models.predict("lgbm_meta", test)
    best = int(np.argmax(scores))
    out = reasons(models.lgbm_meta, x[[best]], models.meta_names, k=3)[0]
    assert out[0][0] == READABLE["sender_prior_acted_rate"]
    assert out[0][1] > 0  # pushes the score up


def test_reasons_are_sorted_by_magnitude_and_limited_to_k(setup):
    models, _, x = setup
    out = reasons(models.lgbm_meta, x[:5], models.meta_names, k=2)
    assert len(out) == 5
    for row in out:
        assert len(row) == 2
        assert abs(row[0][1]) >= abs(row[1][1])


def test_svd_components_are_merged_into_one_wording_reason(setup):
    models, test, _ = setup
    x = models.matrix(test, with_text=True)
    contribs, _ = contributions(models.lgbm_meta_text, x)
    grouped, names = group_contributions(contribs, models.meta_text_names)
    assert names.count(READABLE["svd"]) == 1
    assert grouped.shape[1] == len(names) < contribs.shape[1]
    np.testing.assert_allclose(grouped.sum(axis=1), contribs.sum(axis=1), atol=1e-9)


def test_every_metadata_feature_has_a_readable_label():
    from inbox_sorter.features import FEATURE_COLUMNS

    assert set(FEATURE_COLUMNS) <= set(READABLE)
    assert len(set(READABLE.values())) == len(READABLE)
