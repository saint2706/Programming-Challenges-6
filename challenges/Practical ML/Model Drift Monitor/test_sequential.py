import numpy as np
import pytest

from sequential import SeqParams, calibrate_sequential, make_detectors, run

REF = np.random.default_rng(0).normal(5.0, 2.0, 4500)


@pytest.fixture(scope="module")
def params():
    return calibrate_sequential(REF, reps=6, seed=0)


def test_calibration_returns_parameters_with_the_reference_location_and_scale(params):
    assert isinstance(params, SeqParams)
    assert params.loc == pytest.approx(5.0, abs=0.2) and params.scale == pytest.approx(
        2.0, abs=0.2
    )
    assert 0 < params.adwin_delta < 1 and params.ph_threshold > 0


def test_both_detectors_stay_quiet_on_a_fresh_stationary_stream(params):
    stream = np.random.default_rng(11).normal(5.0, 2.0, 3000)
    for name, det in make_detectors(params).items():
        assert run(det, stream) == [], name


def test_both_detectors_alarm_soon_after_a_step_change(params):
    rng = np.random.default_rng(12)
    stream = np.concatenate([rng.normal(5, 2, 1500), rng.normal(5 + 1.5 * 2, 2, 1500)])
    for name, det in make_detectors(params).items():
        alarms = run(det, stream)
        assert alarms, name
        assert 1500 <= alarms[0] <= 1500 + 300, (name, alarms[0])


def test_run_returns_sorted_indices_and_reset_clears_state(params):
    rng = np.random.default_rng(13)
    stream = np.concatenate([rng.normal(5, 2, 800), rng.normal(11, 2, 800)])
    det = make_detectors(params)["adwin"]
    first = run(det, stream)
    assert first == sorted(first)
    det.reset()
    assert run(det, stream) == first  # a fresh detector reproduces the same alarms


def test_calibrated_false_alarms_on_the_reference_are_within_target(params):
    # the contract of calibration: on resampled reference data, <= alpha per 336-row window
    rng = np.random.default_rng(14)
    total, n_alarms = 0, 0
    for _ in range(6):
        s = rng.choice(REF, size=len(REF))
        total += len(s)
        n_alarms += sum(len(run(d, s)) for d in make_detectors(params).values())
    windows = total / 336
    assert n_alarms / (2 * windows) <= 0.02


def test_a_constant_reference_does_not_divide_by_zero():
    p = calibrate_sequential(np.full(1000, 3.0), reps=2, seed=0)
    assert p.scale == 1.0
    assert run(make_detectors(p)["page_hinkley"], np.full(200, 3.0)) == []


def test_block_calibration_is_quiet_on_the_reference_in_its_own_order():
    # a slowly wandering reference: the ordered stream is the honest null
    rng = np.random.default_rng(21)
    ref = np.cumsum(rng.normal(0, 0.02, 90)).repeat(50) + rng.normal(0, 1, 4500)
    p = calibrate_sequential(ref, reps=4, seed=0, mode="blocks")
    for name, det in make_detectors(p).items():
        assert run(det, ref) == [], name


def test_block_calibration_is_never_more_sensitive_than_iid_on_wandering_data():
    rng = np.random.default_rng(22)
    ref = np.cumsum(rng.normal(0, 0.02, 90)).repeat(50) + rng.normal(0, 1, 4500)
    iid = calibrate_sequential(ref, reps=4, seed=0, mode="iid")
    blocks = calibrate_sequential(ref, reps=4, seed=0, mode="blocks")
    assert (
        blocks.ph_threshold >= iid.ph_threshold
        and blocks.adwin_delta <= iid.adwin_delta
    )


def test_unknown_sequential_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        calibrate_sequential(REF, reps=1, mode="nope")
