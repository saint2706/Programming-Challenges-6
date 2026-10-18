import pytest
from data_lineage.analyze import AnalysisError
from data_lineage.graph import build_graph, label, parse_node, source_schemas
from data_lineage.project import ProjectError, Source, load_project
from helpers import T_CSV, U_CSV, load, make_pipeline


def pipeline(tmp_path, models, sources=None):
    return make_pipeline(tmp_path, sources or {"t": T_CSV, "u": U_CSV}, models)


# -- project loading ----------------------------------------------------------------------------
def test_loads_models_with_layers_and_names(tmp_path):
    root = pipeline(
        tmp_path, {"staging/stg_t": "SELECT a FROM t", "marts/m": "SELECT a FROM stg_t"}
    )
    project = load_project(root)
    assert {(m.name, m.layer) for m in project.models} == {
        ("stg_t", "staging"),
        ("m", "marts"),
    }
    assert [s.name for s in project.sources] == ["t", "u"]


def test_trailing_semicolon_and_whitespace_are_stripped(tmp_path):
    project = load_project(pipeline(tmp_path, {"m": "\n  SELECT a FROM t ;\n"}))
    assert project.model("m").sql == "SELECT a FROM t"


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda p: p / "nope", "not found"),
        (lambda p: p, "sources.toml"),
    ],
)
def test_missing_pieces(tmp_path, setup, message):
    with pytest.raises(ProjectError, match=message):
        load_project(setup(tmp_path))


def test_source_file_must_exist(tmp_path):
    (tmp_path / "sources.toml").write_text('[sources.t]\npath = "missing.csv"\n')
    (tmp_path / "m.sql").write_text("SELECT 1")
    with pytest.raises(ProjectError, match="file not found"):
        load_project(tmp_path)


def test_duplicate_model_and_source_names_are_rejected(tmp_path):
    root = pipeline(tmp_path, {"a/m": "SELECT a FROM t", "b/m": "SELECT a FROM t"})
    with pytest.raises(ProjectError, match="defined twice"):
        load_project(root)
    root = pipeline(tmp_path / "x", {"t": "SELECT a FROM u"})
    with pytest.raises(ProjectError, match="defined twice"):
        load_project(root)


def test_model_names_must_be_identifiers(tmp_path):
    with pytest.raises(ProjectError, match="plain identifier"):
        load_project(pipeline(tmp_path, {"my-model": "SELECT a FROM t"}))


def test_empty_model_and_no_models(tmp_path):
    with pytest.raises(ProjectError, match="is empty"):
        load_project(pipeline(tmp_path, {"m": "  ;"}))
    with pytest.raises(ProjectError, match="no .sql models"):
        load_project(pipeline(tmp_path / "y", {}))


def test_unsupported_source_type(tmp_path):
    (tmp_path / "x.xlsx").write_bytes(b"")
    (tmp_path / "sources.toml").write_text('[sources.x]\npath = "x.xlsx"\n')
    (tmp_path / "m.sql").write_text("SELECT 1")
    project = load_project(tmp_path)
    with pytest.raises(ProjectError, match="unsupported file type"):
        project.sources[0].reader  # noqa: B018


def test_source_path_with_a_quote_is_escaped(tmp_path):
    src = Source("t", tmp_path / "it's.csv")
    assert "it''s.csv" in src.relation()


# -- graph --------------------------------------------------------------------------------------
def test_source_schema_comes_from_the_file(tmp_path):
    project = load_project(pipeline(tmp_path, {"m": "SELECT a FROM t"}))
    schema = source_schemas(project)
    assert list(schema["t"]) == ["a", "b", "c", "d"] and schema["t"]["a"] == "BIGINT"
    assert schema["t"]["d"] == "DATE"


def test_models_are_built_in_dependency_order(tmp_path):
    _, g = load(
        pipeline(
            tmp_path,
            {
                "z_last": "SELECT a FROM mid",
                "mid": "SELECT a FROM t",
                "a_first": "SELECT k FROM u",
            },
        )
    )
    order = g.models
    assert order.index("mid") < order.index("z_last")
    assert g.rank == {"t": 0, "u": 0, "mid": 1, "a_first": 1, "z_last": 2}


def test_cycle_is_reported_with_the_models_involved(tmp_path):
    root = pipeline(tmp_path, {"p": "SELECT a FROM q", "q": "SELECT a FROM p"})
    with pytest.raises(AnalysisError, match="circular.*p, q"):
        load(root)


def test_unknown_table_and_self_reference(tmp_path):
    with pytest.raises(AnalysisError, match="m reads ghost"):
        load(pipeline(tmp_path, {"m": "SELECT a FROM ghost"}))
    with pytest.raises(AnalysisError, match="reads itself"):
        load(pipeline(tmp_path / "s", {"m": "SELECT a FROM m"}))


def test_closures_follow_direct_edges_then_indirect_ones(tmp_path):
    _, g = load(
        pipeline(
            tmp_path,
            {
                "s1": "SELECT a, b AS bb FROM t WHERE c = 'x'",
                "s2": "SELECT a * 2 AS a2 FROM s1",
            },
        )
    )
    a = ("t", "a")
    assert g.downstream(a, indirect=False) == {("s1", "a"), ("s2", "a2")}
    # t.c filters s1, so it reaches both columns of s1 and, through them, s2.a2
    c = ("t", "c")
    assert g.downstream(c, indirect=False) == set()
    assert g.downstream(c, indirect=True) == {("s1", "a"), ("s1", "bb"), ("s2", "a2")}
    assert g.upstream(("s2", "a2"), indirect=False) == {("s1", "a"), ("t", "a")}
    assert g.upstream(("s2", "a2"), indirect=True) == {
        ("s1", "a"),
        ("t", "a"),
        ("t", "c"),
    }
    assert g.source_columns(("s2", "a2"), indirect=True) == {("t", "a"), ("t", "c")}


def test_has_node_and_label_roundtrip(tmp_path):
    _, g = load(pipeline(tmp_path, {"m": "SELECT a FROM t"}))
    assert (
        g.has_node(("m", "a"))
        and not g.has_node(("m", "zzz"))
        and not g.has_node(("q", "a"))
    )
    assert parse_node(label(("m", "a"))) == ("m", "a")
    for bad in ("nodot", ".x", "x."):
        with pytest.raises(ValueError):
            parse_node(bad)


def test_to_json_has_every_edge_and_indirect_link(tmp_path):
    _, g = load(pipeline(tmp_path, {"m": "SELECT a, b + a AS s FROM t WHERE c = 'x'"}))
    data = g.to_json()
    assert {tuple(e["dst"]) for e in data["edges"]} == {("m", "a"), ("m", "s")}
    assert len(data["edges"]) == 3
    assert data["indirect"] == [
        {"src": ["t", "c"], "model": "m", "reasons": ["filter"]}
    ]
    assert [m["name"] for m in data["models"]] == ["m"]


# -- the real pipeline --------------------------------------------------------------------------
def test_real_pipeline_shape(real_graph):
    assert len(real_graph.models) == 10 and len(real_graph.sources) == 2
    assert len(real_graph.all_nodes()) == 77
    assert real_graph.rank["mart_rfm"] == 5
    assert all(not ml.warnings for ml in real_graph.lineage.values())


def test_real_pipeline_money_traces_to_quantity_and_price(real_graph):
    sources = real_graph.source_columns(("mart_rfm", "monetary"), indirect=False)
    assert ("raw_transactions", "quantity") in sources and (
        "raw_transactions",
        "price",
    ) in sources
    assert ("raw_transactions", "country") not in sources


def test_real_pipeline_stock_code_also_shapes_columns_it_does_not_feed(real_graph):
    # stock_code is copied into product_code and also decides which lines exist at all
    node = ("raw_transactions", "stock_code")
    by_value = real_graph.downstream(node, indirect=False)
    with_rows = real_graph.downstream(node, indirect=True)
    assert ("stg_lines", "product_code") in by_value
    assert ("stg_lines", "line_amount") not in by_value
    assert ("stg_lines", "line_amount") in with_rows
    assert len(with_rows) > 3 * len(by_value)


def test_real_pipeline_using_join_depends_on_both_customer_id_columns(real_graph):
    parents = {p for p, _, _ in real_graph.direct_up[("mart_customers", "customer_id")]}
    assert parents == {
        ("int_customer_orders", "customer_id"),
        ("int_refunds", "customer_id"),
    }


def test_real_pipeline_static_columns_match_what_duckdb_builds(real_project):
    graph = build_graph(real_project)
    assert [c.name for c in graph.lineage["mart_customers"].columns][:3] == [
        "customer_id",
        "n_orders",
        "gross_revenue",
    ]
