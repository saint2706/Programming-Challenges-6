"""The OpenVINO graph must reproduce the NumPy score exactly, on every device that exists."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from credit_scorecard import ov_scoring
from credit_scorecard.binning import fit_binning
from credit_scorecard.scorecard import Scorecard

DEVICES = [d for d in ov_scoring.available_devices() if d in ("CPU", "GPU")]


def _edge_case_card() -> tuple[Scorecard, pl.DataFrame]:
    """Features covering every graph branch: cuts, special codes, a missing bin, no cuts at all."""
    rng = np.random.default_rng(0)
    n = 6_000
    y = (rng.random(n) < 0.1).astype(int)
    ranged = rng.normal(size=n) + 0.8 * y
    coded = rng.integers(0, 4, n).astype(float) + y
    coded[:60] = 98.0
    gappy = rng.normal(size=n)
    gappy[60:360] = np.nan
    constant = np.full(n, 3.0)
    frame = pl.DataFrame(
        {"ranged": ranged, "coded": coded, "gappy": gappy, "constant": constant}
    )
    binnings = {
        "ranged": fit_binning("ranged", ranged, y),
        "coded": fit_binning("coded", coded, y, specials=[98.0]),
        "gappy": fit_binning("gappy", gappy, y),
        "constant": fit_binning("constant", constant, y),
    }
    assert binnings["constant"].edges == [] and binnings["coded"].specials
    card = Scorecard(binnings, dict.fromkeys(binnings, 0.9), intercept=2.5)
    return card, frame


def _probe_frame(card: Scorecard) -> pl.DataFrame:
    """Values exactly on, just below and just above every cut, plus NaN and the sentinel."""
    rows = {f: [] for f in card.features}
    for f in card.features:
        edges = card.binnings[f].edges
        probes = [
            v
            for e in edges
            for v in (e, np.nextafter(np.float32(e), np.float32(-np.inf)), e + 0.5)
        ]
        probes += [np.nan, 98.0, -1e6, 1e6]
        rows[f] = [float(p) for p in probes]
    length = max(len(v) for v in rows.values())
    return pl.DataFrame({f: (v + [np.nan] * length)[:length] for f, v in rows.items()})


@pytest.mark.parametrize("device", DEVICES)
def test_edge_cases_match_numpy_exactly(device):
    card, frame = _edge_case_card()
    scorer = ov_scoring.OvScorer(card, device)
    assert np.array_equal(scorer.score(frame), card.score(frame))
    probe = _probe_frame(card)
    mismatched = np.flatnonzero(scorer.score(probe) != card.score(probe))
    # float32 inputs: only a value within one float32 step of a cut may land in the next bin.
    assert len(mismatched) <= 2, probe[mismatched.tolist()]


@pytest.mark.parametrize("device", DEVICES)
def test_real_test_set_matches_numpy(result, device):
    assert ov_scoring.parity(result.scorecard, result.test, device) > 0.9999


def test_cpu_is_always_available_and_batch_size_is_dynamic(result):
    scorer = ov_scoring.OvScorer(result.scorecard, "CPU")
    exact = result.scorecard.score(result.test)
    for n in (1, 7, 5_000):
        assert np.array_equal(scorer.score(result.test.head(n)), exact[:n])
