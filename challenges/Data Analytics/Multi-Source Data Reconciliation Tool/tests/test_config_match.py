from pathlib import Path

import polars as pl
import pytest
from data_reconciler import config
from data_reconciler.config import ConfigError, Key
from data_reconciler.load import ROW, resolve
from data_reconciler.match import STATUSES, match, normalize_key

ROOT = Path(__file__).resolve().parents[1]

# --- config -----------------------------------------------------------------------------------


def raw(**override):
    base = {
        "left": {"name": "L", "path": "l.csv"},
        "right": {"name": "R", "path": "r.csv"},
        "keys": [{"left": "a", "right": "b"}],
        "fields": [{"name": "f", "left": "a", "right": "b", "compare": "text"}],
    }
    base.update(override)
    return base


def test_the_bundled_airports_job_parses():
    cfg = config.load(ROOT / "airports.toml")
    assert [k.left for k in cfg.keys] == ["iata", "icao"]
    assert {f.compare for f in cfg.fields} == {
        "code",
        "text",
        "crosswalk",
        "geo",
        "number",
    }
    assert cfg.identity == ("name", "location", "country") and cfg.identity_min == 3
    assert cfg.left.null_values == ("\\N",)


def test_a_minimal_job_parses_with_defaults():
    cfg = config.parse(raw())
    assert cfg.fields[0].near == 0.85 and cfg.left.header is True and cfg.identity == ()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(extra=1), r"unknown key\(s\) \['extra'\]"),
        (lambda r: r.pop("keys"), "needs a `keys` section"),
        (lambda r: r.update(keys=[]), "at least one"),
        (lambda r: r.update(keys=[{"left": "a"}]), "needs both"),
        (lambda r: r["left"].update(bogus=1), r"\[left\] has unknown"),
        (lambda r: r["left"].pop("path"), "needs `path`"),
        (lambda r: r["left"].update(header=False), "needs `columns`"),
        (lambda r: r["fields"][0].update(compare="fuzzy"), "compare must be one of"),
        (lambda r: r["fields"][0].update(typo=1), "unknown key"),
        (lambda r: r["fields"][0].update(compare="geo"), "takes 2 column"),
        (lambda r: r["fields"][0].update(near=2), "near must be between"),
        (lambda r: r["fields"][0].update(abs_tolerance=-1), "cannot be negative"),
        (lambda r: r["fields"][0].update(min_share=0), "min_share"),
        (lambda r: r["fields"].append(dict(r["fields"][0])), "must be unique"),
        (lambda r: r.update(identity=["nope"]), "unknown field"),
        (lambda r: r.update(identity=["f"], identity_min=2), "identity_min"),
    ],
)
def test_bad_configs_say_what_is_wrong(mutate, message):
    r = raw()
    mutate(r)
    with pytest.raises(ConfigError, match=message):
        config.parse(r)


def test_invalid_toml_is_a_config_error(tmp_path):
    (tmp_path / "bad.toml").write_text("left = [", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        config.load(tmp_path / "bad.toml")


def test_data_placeholder_follows_the_available_folder(home, tmp_path):
    assert resolve("{data}/x.csv", home) == home / "sample_data" / "x.csv"
    assert resolve("rel/x.csv", home) == home / "rel" / "x.csv"
    assert resolve(str(tmp_path / "x.csv"), home) == tmp_path / "x.csv"


# --- matching ---------------------------------------------------------------------------------


def frame(**cols):
    n = len(next(iter(cols.values())))
    return pl.DataFrame({**cols, ROW: list(range(n))})


def statuses(result, side):
    return dict(
        result.rows.filter(pl.col("side") == side).select("row", "status").iter_rows()
    )


def test_keys_match_after_trimming_and_upper_casing_and_empty_is_no_key():
    left = frame(k=[" lhr", "CDG", ""])
    right = frame(k=["LHR", "cdg ", "   "])
    result = match(left, right, (Key("k", "k"),))
    assert sorted(result.pairs.select("left_row", "right_row").iter_rows()) == [
        (0, 0),
        (1, 1),
    ]
    assert (
        statuses(result, "left")[2] == "unkeyed"
        and statuses(result, "right")[2] == "unkeyed"
    )
    assert pl.DataFrame({"k": [" a ", ""]}).select(normalize_key("k"))[
        "k"
    ].to_list() == ["A", None]


def test_unmatched_rows_are_only_left_or_only_right():
    result = match(frame(k=["A", "B"]), frame(k=["A", "C"]), (Key("k", "k"),))
    assert statuses(result, "left") == {0: "matched", 1: "only_left"}
    assert statuses(result, "right") == {0: "matched", 1: "only_right"}


def test_a_key_on_several_rows_is_never_guessed():
    left = frame(k=["A", "A", "B"])
    right = frame(k=["A", "B"])
    result = match(left, right, (Key("k", "k"),))
    assert statuses(result, "left") == {
        0: "duplicate_key",
        1: "duplicate_key",
        2: "matched",
    }
    assert statuses(result, "right") == {0: "duplicate_key", 1: "matched"}
    assert result.pairs.height == 1
    dups = result.rows.filter(pl.col("status") == "duplicate_key")
    assert set(dups["key"]) == {"A"}


def test_a_duplicate_with_no_counterpart_still_reports_its_key():
    result = match(frame(k=["A", "A"]), frame(k=["Z"]), (Key("k", "k"),))
    dups = result.rows.filter(
        (pl.col("side") == "left") & (pl.col("status") == "duplicate_key")
    )
    assert dups.height == 2 and set(dups["key"]) == {"A"}


def test_the_second_key_only_sees_rows_the_first_left_over():
    left = frame(iata=["AAA", None, "ZZZ"], icao=["XAAA", "XBBB", "XCCC"])
    right = frame(iata=["AAA", None, None], icao=["XAAA", "XBBB", "XNONE"])
    result = match(left, right, (Key("iata", "iata"), Key("icao", "icao")))
    pairs = {
        (r["left_row"], r["right_row"]): r["stage"]
        for r in result.pairs.iter_rows(named=True)
    }
    assert pairs == {(0, 0): 0, (1, 1): 1}
    assert statuses(result, "left")[2] == "only_left"


def test_a_row_that_was_ambiguous_is_not_retried_with_the_next_key():
    left = frame(iata=["AAA", "AAA"], icao=["X1", "X2"])
    right = frame(iata=["AAA"], icao=["X1"])
    result = match(left, right, (Key("iata", "iata"), Key("icao", "icao")))
    assert result.pairs.height == 0  # X1 would have paired row 0 by guesswork
    assert set(statuses(result, "left").values()) == {"duplicate_key"}


def test_a_right_row_cannot_be_paired_twice_across_stages():
    left = frame(a=["K1", "K2"], b=["P", "P"])
    right = frame(a=["K1"], b=["P"])
    result = match(left, right, (Key("a", "a"), Key("b", "b")))
    assert (
        result.pairs.height == 1
    )  # K1 took the right row; P is then ambiguous only among unresolved rows
    assert statuses(result, "left")[1] == "only_left"


def test_every_row_ends_with_exactly_one_known_status():
    left = frame(k=["A", "A", "B", None, "C", " d"])
    right = frame(k=["a", "B", "B", "E", None])
    result = match(left, right, (Key("k", "k"),))
    for side, f in (("left", left), ("right", right)):
        rows = result.rows.filter(pl.col("side") == side)
        assert sorted(rows["row"]) == list(range(f.height))
        assert set(rows["status"]) <= set(STATUSES)
    matched_left = set(result.pairs["left_row"])
    assert (
        len(matched_left) == result.pairs.height == len(set(result.pairs["right_row"]))
    )
