"""The ctypes boundary: text that is not ASCII, awkward CSV, many tables, bad requests."""

import gc

import pytest
from conftest import cells
from pivot_tool.engine import Table
from pivot_tool.ffi import EngineError, engine


def test_non_ascii_labels_and_quoted_fields_survive_the_round_trip():
    csv = 'city,note,n\nZürich,"likes, commas",1\n東京,"two\nlines",2\n"say ""hi""",plain,3\n😀,x,4\n'.encode()
    with Table(csv) as t:
        assert t.rows == 4
        r = t.pivot(
            {
                "rows": [{"field": "city"}],
                "values": [{"field": "n"}],
                "options": {"totals_row": False, "totals_col": False},
            }
        )
        labels = [h["labels"][0] for h in r["rows"]]
        assert set(labels) == {"Zürich", "東京", 'say "hi"', "😀"}
        d = t.drill({"rows": [{"field": "city"}]}, ["東京"], [], 5)
        assert d["rows"] == [["東京", "two\nlines", "2"]]


def test_a_file_with_a_byte_order_mark_and_crlf_line_endings_loads():
    with Table("﻿a,b\r\n1,2\r\n3,4\r\n".encode()) as t:
        assert t.names == ["a", "b"] and t.rows == 2


def test_semicolon_files_are_detected_and_an_explicit_delimiter_wins():
    with Table(b"a;b\n1;2\n") as t:
        assert t.names == ["a", "b"]
    with Table(b"a;b,c\n1;2,3\n", delimiter=",") as t:
        assert t.names == ["a;b", "c"]


def test_many_tables_at_once_and_dropping_them():
    tables = [Table(b"x\n1\n2\n") for _ in range(50)]
    assert len({t.id for t in tables}) == 50
    for t in tables:
        t.close()
    with pytest.raises(EngineError, match="no table with id"):
        engine().call("describe", {"table": tables[0].id or 1_000_000})
    gc.collect()


def test_a_table_is_dropped_when_it_goes_out_of_scope():
    t = Table(b"x\n1\n")
    table_id = t.id
    del t
    gc.collect()
    with pytest.raises(EngineError, match="no table with id"):
        engine().call("describe", {"table": table_id})


def test_a_big_file_loads_and_pivots():
    rows = "\n".join(f"g{i % 7},{i % 100},{i}" for i in range(200_000))
    with Table(f"g,k,v\n{rows}\n".encode()) as t:
        r = t.pivot(
            {
                "rows": [{"field": "g"}],
                "cols": [{"field": "k", "group": {"kind": "bin", "width": 50}}],
                "values": [{"field": "v"}],
            }
        )
        total = cells(r)[((0, ()), (0, ()))]
        assert total == sum(range(200_000))


def test_bad_requests_come_back_as_errors():
    e = engine()
    with pytest.raises(EngineError, match="bad request"):
        e.call("pivot", {"table": 1})
    with pytest.raises(EngineError, match="bad request"):
        e.call("pivot", {"table": 1, "spec": {"rows": [{"field": "a", "nope": 1}]}})
    with pytest.raises(EngineError, match="fields, but the header has"):
        Table(b"a,b\n1\n")
    with pytest.raises(EngineError, match="UTF-8"):
        Table(bytes([0xFF, 0xFE, 0x00]))
    assert e.call("version")["engine"] == "pivot-core"


def test_values_lists_the_distinct_values_in_display_order(adult):
    v = adult.values("income")
    assert v == {"total": 2, "values": ["<=50K", ">50K"]}
    v = adult.values("education", limit=3)
    assert v["total"] == 16 and len(v["values"]) == 3
