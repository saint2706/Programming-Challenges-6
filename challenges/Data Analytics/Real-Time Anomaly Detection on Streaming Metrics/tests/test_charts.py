import math

import pytest
from spc_stream.charts import CUSUM, EWMA, Shewhart, make_chart


def first_alarm(chart, zs):
    for i, z in enumerate(zs):
        p = chart.update(z)
        if p.alarm:
            return i, p.alarm
    return None


@pytest.mark.parametrize(
    ("zs", "rule", "at"),
    [
        ([0, 0, 3.5], "WE1", 2),
        ([0, 2.5, 0, 2.5], "WE2", 3),
        ([-2.5, -2.5], "WE2", 1),
        ([1.5, 1.5, 0, 1.5, 1.5], "WE3", 4),
        ([0.2] * 8, "WE4", 7),
        ([-0.2] * 8, "WE4", 7),
    ],
)
def test_western_electric_rules(zs, rule, at):
    i, alarm = first_alarm(Shewhart(), zs)
    assert alarm.startswith(rule)
    assert i == at


def test_two_of_three_needs_same_side():
    assert first_alarm(Shewhart(), [2.5, -2.5, 0.5]) is None


def test_rule_subset_and_validation():
    assert first_alarm(Shewhart(rules=(1,)), [0.2] * 20) is None
    with pytest.raises(ValueError):
        Shewhart(rules=(5,))


def test_shewhart_resets_after_alarm():
    c = Shewhart()
    assert first_alarm(c, [0.2] * 8)[0] == 7
    assert c.update(0.2).alarm is None  # the run starts over


def test_ewma_limits_grow_to_asymptote():
    c = EWMA(lam=0.2, L=3)
    widths = [c.update(0).ucl for _ in range(200)]
    assert widths[0] == pytest.approx(3 * 0.2)  # first point: sigma_z = lambda
    assert widths == sorted(widths)
    assert widths[-1] == pytest.approx(3 * math.sqrt(0.2 / 1.8))
    assert EWMA(lam=0.2, L=3, exact_limits=False).update(0).ucl == pytest.approx(
        widths[-1]
    )


def test_ewma_detects_small_shift_and_validates():
    i, alarm = first_alarm(EWMA(), [1.0] * 50)
    assert 5 < i < 20
    assert "EWMA" in alarm
    with pytest.raises(ValueError):
        EWMA(lam=0)


def test_cusum_both_directions_and_reset():
    c = CUSUM()
    i, alarm = first_alarm(c, [1.5] * 20)
    assert alarm == "CUSUM upward shift"
    assert i == 5  # 6 * (1.5 - 0.5) = 6 > 5
    assert c.update(0).stat == 0
    assert first_alarm(CUSUM(), [-1.5] * 20)[1] == "CUSUM downward shift"


def test_cusum_stat_sign():
    c = CUSUM()
    assert c.update(2).stat == pytest.approx(1.5)
    c.reset()
    assert c.update(-2).stat == pytest.approx(-1.5)


def test_make_chart():
    assert isinstance(make_chart("cusum"), CUSUM)
    with pytest.raises(ValueError, match="unknown chart"):
        make_chart("xbar")
