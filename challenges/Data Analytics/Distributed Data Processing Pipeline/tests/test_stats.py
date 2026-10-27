from dist_pipeline.stats import summarise


def test_empty_sample_has_no_median():
    assert summarise([])["median"] is None


def test_iqr_is_withheld_below_four_samples():
    s = summarise([1.0, 2.0, 3.0])
    assert s["median"] == 2.0
    assert s["iqr"] is None


def test_iqr_and_median_for_five_samples():
    s = summarise([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["median"] == 3.0
    assert s["iqr"] == 2.0
    assert (s["min"], s["max"], s["n"]) == (1.0, 5.0, 5)
