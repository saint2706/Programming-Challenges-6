import pytest
from pdftoolkit.errors import PdfToolError
from pdftoolkit.ranges import parse_groups, parse_pages


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("1", [0]),
        ("2-4", [1, 2, 3]),
        ("3-", [2, 3, 4]),
        ("5-2", [4, 3, 2, 1]),
        ("1,3-4,1", [0, 2, 3, 0]),
        ("end", [4]),
        ("LAST-3", [4, 3, 2]),
        ("odd", [0, 2, 4]),
        ("even", [1, 3]),
        ("all", [0, 1, 2, 3, 4]),
        (" 2 - 3 , 5 ", [1, 2, 4]),
    ],
)
def test_parse_pages(spec, expected):
    assert parse_pages(spec, 5) == expected


@pytest.mark.parametrize(
    "spec", ["", " , ", "0", "6", "2-9", "a", "1-2-3", "-3", "1..3", "2,x"]
)
def test_bad_specs(spec):
    with pytest.raises(PdfToolError):
        parse_pages(spec, 5)


def test_groups_keep_one_list_per_token():
    assert parse_groups("1-2,4,5-", 6) == [[0, 1], [3], [4, 5]]
