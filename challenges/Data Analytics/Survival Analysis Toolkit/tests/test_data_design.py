"""Loading, cleaning and the design matrix."""

import numpy as np
import polars as pl
import pytest
from survival_toolkit import analysis
from survival_toolkit.data import DATASETS, load_builtin, prepare
from survival_toolkit.design import encode
from survival_toolkit.types import SurvivalError


def test_bundled_datasets_load_with_expected_shapes():
    expected = {"telco": (7032, 1869), "lung": (228, 165), "gbsg": (686, 299)}
    for name, (n, events) in expected.items():
        _, prepared = load_builtin(name)
        assert (len(prepared.survival), prepared.survival.n_events) == (n, events)
    _, telco = load_builtin("telco")
    assert telco.dropped["no exposure (time <= entry or <= 0)"] == 11
    assert "No internet service" not in telco.frame["TechSupport"].unique().to_list()


def test_unknown_dataset_and_missing_file():
    with pytest.raises(SurvivalError, match="unknown dataset"):
        load_builtin("nope")
    with pytest.raises(SurvivalError, match="neither a bundled dataset"):
        analysis.resolve("no_such_file.csv", time="t", event="e")
    with pytest.raises(SurvivalError, match="--time and --event"):
        analysis.resolve(str(__file__))


def test_prepare_drops_and_reports_unusable_rows():
    frame = pl.DataFrame(
        {
            "t": [5.0, 0.0, None, 3.0, 2.0],
            "e": ["x", "x", "x", None, "y"],
            "w": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    out = prepare(frame, time="t", event="e", event_value="x")
    assert len(out.survival) == 2  # only (5, x) and (2, y) are usable
    assert out.survival.event.tolist() == [True, False]
    assert out.dropped["missing time, event or entry"] == 2
    assert out.dropped["no exposure (time <= entry or <= 0)"] == 1


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 1, 1], [False, True, True]),
        ([1, 2, 2], [False, True, True]),
        ([True, False, True], [True, False, True]),
    ],
)
def test_event_coding_is_inferred(values, expected):
    frame = pl.DataFrame({"t": [1.0, 2.0, 3.0], "e": values})
    assert prepare(frame, time="t", event="e").survival.event.tolist() == expected


def test_ambiguous_event_column_needs_a_value():
    frame = pl.DataFrame({"t": [1.0, 2.0], "e": ["dead", "alive"]})
    with pytest.raises(SurvivalError, match="--event-value"):
        prepare(frame, time="t", event="e")
    with pytest.raises(SurvivalError, match="no column"):
        prepare(frame, time="zzz", event="e")


def test_late_entry_column_is_validated():
    frame = pl.DataFrame({"t": [5.0, 4.0], "e": [1, 1], "s": [1.0, 4.0]})
    out = prepare(frame, time="t", event="e", entry="s")
    assert (
        len(out.survival) == 1
        and out.dropped["no exposure (time <= entry or <= 0)"] == 1
    )


def test_encode_numeric_boolean_and_text_columns():
    frame = pl.DataFrame(
        {
            "age": [30.0, 40.0, None, 50.0, 60.0],
            "flag": [True, False, True, True, False],
            "plan": ["b", "a", "a", "c", "a"],
        }
    )
    d = encode(frame, ["age", "flag", "plan"])
    assert d.keep.tolist() == [True, True, False, True, True]
    assert d.names == [
        "age",
        "flag",
        "plan[b]",
        "plan[c]",
    ]  # 'a' is the most common: the reference
    assert d.levels["plan"] == ["a", "b", "c"]
    assert d.x.shape == (4, 4)
    np.testing.assert_array_equal(d.x[:, 2], [1, 0, 0, 0])
    named = encode(frame, ["plan"], reference={"plan": "c"})
    assert named.names == ["plan[a]", "plan[b]"]


def test_encode_errors():
    frame = pl.DataFrame({"a": ["x", "x"], "b": [1.0, 2.0]})
    with pytest.raises(SurvivalError, match="single level"):
        encode(frame, ["a"])
    with pytest.raises(SurvivalError, match="reference level"):
        encode(frame, ["a"], reference={"a": "zz"})
    with pytest.raises(SurvivalError, match="no column"):
        encode(frame, ["nope"])


def test_group_labels_bins_and_limits():
    frame = pl.DataFrame(
        {
            "x": list(range(100)),
            "g": [f"g{i}" for i in range(100)],
            "c": ["a", "b"] * 50,
        }
    )
    labels = analysis.group_labels(frame, "x", bins=4)
    assert len(set(labels)) == 4 and min(set(labels)).startswith("Q1")
    assert set(analysis.group_labels(frame, "c")) == {"a", "b"}
    with pytest.raises(SurvivalError, match="more than"):
        analysis.group_labels(frame, "g")
    with pytest.raises(SurvivalError, match="numeric"):
        analysis.group_labels(frame, "c", bins=3)
    with pytest.raises(SurvivalError, match="no column"):
        analysis.group_labels(frame, "zz")


def test_every_spec_points_at_real_columns():
    for name, spec in DATASETS.items():
        _, prepared = load_builtin(name)
        for col in (*spec.covariates, spec.group, spec.time, spec.event):
            assert col in prepared.frame.columns, (name, col)
        assert spec.strata is None or spec.strata in prepared.frame.columns
