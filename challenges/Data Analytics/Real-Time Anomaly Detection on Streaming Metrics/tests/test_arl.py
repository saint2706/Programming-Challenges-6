import pytest
from spc_stream import arl


def test_shewhart_closed_form():
    assert arl.shewhart_exact(0) == pytest.approx(370.4, abs=0.1)
    assert arl.shewhart_exact(1) == pytest.approx(43.9, abs=0.1)


def test_simulated_arl_matches_published(monkeypatch):
    monkeypatch.setattr(arl, "SHIFTS", (0.0, 1.0, 2.0))
    res = arl.simulate(reps=500, seed=7)
    checked = 0
    for name, by_shift in res.items():
        for d, (mean, se) in by_shift.items():
            ref = (
                arl.shewhart_exact(d)
                if name == "shewhart 3 sigma"
                else arl.PUBLISHED[name].get(d)
            )
            if ref is None:
                continue
            assert abs(mean - ref) < 4 * se + 0.05 * ref, (name, d, mean, ref)
            checked += 1
    assert checked == 10
