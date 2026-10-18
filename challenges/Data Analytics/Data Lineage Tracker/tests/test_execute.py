import pytest
from data_lineage import execute

COLUMNS = {"id": "BIGINT", "s": "VARCHAR", "d": "DATE", "f": "DOUBLE", "b": "BOOLEAN"}


@pytest.fixture
def con():
    c = execute.connect()
    c.execute(
        "CREATE TABLE base AS SELECT i AS id, 'row' || i AS s, DATE '2020-01-01' + i::INT AS d, "
        "i * 1.5 AS f, i % 2 = 0 AS b FROM range(1, 22) t(i)"
    )
    yield c
    c.close()


def perturbed(con, column, strategy):
    return con.execute(
        execute.perturbed_select("base", COLUMNS, column, strategy)
    ).fetchall()


def base_rows(con):
    return con.execute("SELECT * FROM base ORDER BY id").fetchall()


def index(column):
    return list(COLUMNS).index(column)


@pytest.mark.parametrize("strategy", execute.STRATEGIES)
@pytest.mark.parametrize("column", list(COLUMNS))
def test_every_strategy_keeps_shape_and_the_other_columns(con, column, strategy):
    out, base = perturbed(con, column, strategy), base_rows(con)
    assert len(out) == len(base)
    i = index(column)
    for new_row, old_row in zip(out, base, strict=True):
        assert [v for j, v in enumerate(new_row) if j != i] == [
            v for j, v in enumerate(old_row) if j != i
        ]


def test_shift_is_type_aware(con):
    base = base_rows(con)
    assert [r[0] for r in perturbed(con, "id", "shift")] == [r[0] * 3 + 7 for r in base]
    assert [r[1] for r in perturbed(con, "s", "shift")] == [r[1] + "~" for r in base]
    assert [
        (r[2] - o[2]).days
        for r, o in zip(perturbed(con, "d", "shift"), base, strict=True)
    ] == [11] * 21
    assert [r[3] for r in perturbed(con, "f", "shift")] == [r[3] * 3 + 7 for r in base]
    assert [r[4] for r in perturbed(con, "b", "shift")] == [not r[4] for r in base]


def test_shuffle_keeps_the_values_but_not_their_rows_and_is_repeatable(con):
    first = [r[0] for r in perturbed(con, "id", "shuffle")]
    assert sorted(first) == list(range(1, 22)) and first != list(range(1, 22))
    assert first == [r[0] for r in perturbed(con, "id", "shuffle")]


def test_nullify_blanks_every_seventh_row(con):
    out = perturbed(con, "s", "nullify")
    assert [i + 1 for i, r in enumerate(out) if r[1] is None] == [7, 14, 21]


def test_constant_collapses_to_the_minimum(con):
    assert {r[0] for r in perturbed(con, "id", "constant")} == {1}
    assert {r[1] for r in perturbed(con, "s", "constant")} == {"row1"}


def test_bad_arguments(con):
    with pytest.raises(ValueError):
        execute.perturbed_select("base", COLUMNS, "id", "explode")
    with pytest.raises(KeyError):
        execute.perturbed_select("base", COLUMNS, "nope", "shift")


def test_identifiers_are_quoted():
    assert execute.qi('we"ird') == '"we""ird"'


def test_unique_key_skips_floats_nulls_and_repeats(con):
    assert execute.unique_key(con, "base") == "id"
    con.execute(
        "CREATE TABLE t2 AS SELECT * FROM (VALUES (1.5, 'a'), (2.5, 'a')) v(f, s)"
    )
    assert execute.unique_key(con, "t2") is None  # f is unique but a float, s repeats
    con.execute(
        "CREATE TABLE t3 AS SELECT * FROM (VALUES (1, 'a'), (NULL, 'b')) v(k, s)"
    )
    assert execute.unique_key(con, "t3") == "s"  # k has a NULL
    con.execute("CREATE TABLE t4 AS SELECT 1 AS k WHERE false")
    assert execute.unique_key(con, "t4") is None


def test_plain_fingerprint_ignores_row_order_aligned_does_not(con):
    con.execute(
        "CREATE TABLE a AS SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) v(k, x)"
    )
    con.execute(
        "CREATE TABLE b AS SELECT * FROM (VALUES (3, 30), (1, 10), (2, 20)) v(k, x)"
    )
    con.execute(
        "CREATE TABLE c AS SELECT * FROM (VALUES (1, 20), (2, 10), (3, 30)) v(k, x)"
    )
    fa, fb, fc = (execute.fingerprints(con, t, "k") for t in "abc")
    assert fa == fb  # same rows in another order
    assert fa.plain["x"] == fc.plain["x"]  # same values...
    assert fa.aligned["x"] != fc.aligned["x"]  # ...on different rows
    assert execute.fingerprints(con, "a").aligned == {}


def test_float_noise_below_four_decimals_is_ignored(con):
    con.execute("CREATE TABLE a AS SELECT 0.1 + 0.2 AS x")
    con.execute("CREATE TABLE b AS SELECT 0.3 AS x")
    assert execute.fingerprints(con, "a").plain == execute.fingerprints(con, "b").plain


def test_nulls_and_empty_strings_hash_differently(con):
    con.execute("CREATE TABLE a AS SELECT * FROM (VALUES (NULL::VARCHAR), ('x')) v(s)")
    con.execute("CREATE TABLE b AS SELECT * FROM (VALUES (''), ('x')) v(s)")
    assert execute.fingerprints(con, "a").plain != execute.fingerprints(con, "b").plain
