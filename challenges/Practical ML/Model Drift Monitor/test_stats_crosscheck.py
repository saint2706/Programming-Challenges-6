"""Our statistics against Evidently's, on the same data (dev dependency)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently")
from evidently.legacy.calculations.stattests.jensenshannon import (
    _jensenshannon,
)
from evidently.legacy.calculations.stattests.ks_stattest import (
    _ks_stat_test,
)
from evidently.legacy.calculations.stattests.psi import _psi
from evidently.legacy.calculations.stattests.wasserstein_distance_norm import (
    _wasserstein_distance_norm,
)
from evidently.legacy.core import ColumnType
from scipy import stats as sps

from stats import Baseline

RNG = np.random.default_rng(5)


def test_categorical_psi_and_js_equal_evidently():
    ref = RNG.integers(1, 8, 5000)
    live = RNG.choice(np.arange(1, 8), 336, p=[0.3, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15])
    ours = Baseline.fit({"day": ref}, categorical=["day"]).window_stats("day", live)
    r, c = pd.Series(ref), pd.Series(live)
    psi_ev, _ = _psi(r, c, ColumnType.Categorical, 0.1)
    js_ev, _ = _jensenshannon(r, c, ColumnType.Categorical, 0.1, base=2)
    assert ours["psi"] == pytest.approx(psi_ev, rel=1e-3)
    assert ours["js"] == pytest.approx(js_ev, rel=1e-3)


def test_wasserstein_equals_evidently_normalized_by_the_reference_std():
    ref, live = RNG.normal(0, 1, 4000), RNG.normal(0.7, 1.3, 336)
    ours = Baseline.fit({"x": ref}).window_stats("x", live)["wasserstein"]
    ev, _ = _wasserstein_distance_norm(
        pd.Series(ref), pd.Series(live), ColumnType.Numerical, 0.1
    )
    assert ours == pytest.approx(ev, rel=1e-6)


def test_ks_statistic_has_the_p_value_evidently_reports():
    ref, live = RNG.normal(0, 1, 4000), RNG.normal(0.3, 1, 336)
    stat = Baseline.fit({"x": ref}).window_stats("x", live)["ks"]
    p_ev, _ = _ks_stat_test(pd.Series(ref), pd.Series(live), ColumnType.Numerical, 0.05)
    p_ours = sps.ks_2samp(ref, live).pvalue
    assert p_ev == pytest.approx(p_ours, rel=1e-6)
    assert stat == pytest.approx(sps.ks_2samp(ref, live).statistic)
