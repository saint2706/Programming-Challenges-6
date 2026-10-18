import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from data_lineage.cli import app
from data_lineage.graph import parse_node
from data_lineage.report import build_html, embed_json
from data_lineage.verify import verify
from helpers import T_CSV, U_CSV, load, make_pipeline
from typer.testing import CliRunner

JS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "data_lineage"
    / "assets"
    / "report.js"
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def clean(text: str) -> str:
    """Strip colour codes and Rich box drawing so assertions survive GitHub Actions' styling."""
    text = ANSI.sub("", text)
    text = re.sub(r"[─-╿]", " ", text)
    return re.sub(r"\s+", " ", text)


def cli(*args):
    result = CliRunner().invoke(app, [str(a) for a in args])
    return result.exit_code, clean(result.output)


def embedded(page: str) -> dict:
    m = re.search(
        r'<script id="lineage-data" type="application/json">(.*?)</script>',
        page,
        re.DOTALL,
    )
    return json.loads(m.group(1))


# -- the HTML file -------------------------------------------------------------------------------
def test_report_is_self_contained(real_graph):
    page = build_html(real_graph, None, "Real")
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", page))
    assert urls <= {
        "http://www.w3.org/2000/svg"
    }  # an XML namespace name, never fetched
    assert '<link rel="stylesheet"' not in page and 'src="http' not in page


def test_embedded_data_is_the_graph_plus_title_and_verification(tmp_path):
    root = make_pipeline(tmp_path, {"t": T_CSV}, {"m": "SELECT a, b + 1 AS b1 FROM t"})
    project, graph = load(root)
    v = verify(project, graph, only_sources={("t", "a")}, strategies=("shift",))
    data = embedded(build_html(graph, v, "My pipeline"))
    assert data["title"] == "My pipeline"
    assert {k: data[k] for k in ("sources", "models", "edges", "indirect")} == {
        k: graph.to_json()[k] for k in ("sources", "models", "edges", "indirect")
    }
    assert data["verification"]["runs"] == v.runs
    assert embedded(build_html(graph))["verification"] is None


def test_hostile_names_cannot_break_out_of_the_page(tmp_path):
    evil = "a</script><img src=x onerror=alert(1)>"
    root = make_pipeline(
        tmp_path,
        {"t": f'"{evil}",b\n1,2\n3,4'},
        {"m": f'SELECT "{evil}" AS "<b>x</b>&amp;" FROM t'},
    )
    _, graph = load(root)
    page = build_html(graph, None, "</title><script>alert(1)</script>")
    assert "<img src=x" not in page
    assert (
        page.count("</script") == 2
    )  # the data block and the viewer, nothing from the data
    assert "&lt;/title&gt;&lt;script&gt;" in page
    data = embedded(page)  # and the data survives the escaping unchanged
    assert data["sources"][0]["columns"][0]["name"] == evil
    assert data["models"][0]["columns"][0]["name"] == "<b>x</b>&amp;"


def test_embed_json_escapes_the_javascript_line_separators():
    text = embed_json({"s": "a b c</script>&"})
    assert " " not in text and " " not in text and "</script>" not in text
    assert json.loads(text)["s"] == "a b c</script>&"


def test_viewer_builds_dom_without_innerhtml():
    source = JS.read_text(encoding="utf-8")
    assert (
        "innerHTML" not in source
        and "outerHTML" not in source
        and "insertAdjacentHTML" not in source
    )
    assert "document.write" not in source and ("ev" + "al(") not in source


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_viewer_closures_equal_the_python_graph_for_every_column(real_graph, tmp_path):
    data = tmp_path / "graph.json"
    data.write_text(json.dumps(real_graph.to_json()), encoding="utf-8")
    script = tmp_path / "check.js"
    script.write_text(
        """
const fs = require("fs");
const L = require(process.argv[2]);
const data = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const idx = L.buildIndex(data);
const out = {};
for (const [k, v] of idx.columns) {
  const name = v.table + "." + v.column;
  out[name] = {};
  for (const ind of [false, true]) {
    const f = (set) => [...set].map((s) => s.replace("\\u0000", ".")).sort();
    out[name][ind] = { up: f(L.upstream(idx, v.table, v.column, ind)), down: f(L.downstream(idx, v.table, v.column, ind)) };
  }
}
process.stdout.write(JSON.stringify(out));
""",
        encoding="utf-8",
    )
    run = subprocess.run(
        ["node", str(script), str(JS), str(data)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    js = json.loads(run.stdout)
    assert len(js) == 77
    for name, modes in js.items():
        node = parse_node(name)
        for ind in (False, True):
            want_up = sorted(f"{t}.{c}" for t, c in real_graph.upstream(node, ind))
            want_down = sorted(f"{t}.{c}" for t, c in real_graph.downstream(node, ind))
            assert modes[str(ind).lower()]["up"] == want_up, (name, ind, "up")
            assert modes[str(ind).lower()]["down"] == want_down, (name, ind, "down")


# -- the command line ----------------------------------------------------------------------------
def test_models_lists_everything_in_order():
    code, out = cli("models")
    assert code == 0 and "raw_transactions source" in out
    assert out.index("stg_lines") < out.index("int_invoices") < out.index("mart_rfm")


def test_build_writes_json_and_html(tmp_path):
    code, out = cli("build", "--out", tmp_path, "--title", "Retail")
    assert code == 0 and "2 sources, 10 models, 77 columns" in out
    assert json.loads((tmp_path / "lineage.json").read_text(encoding="utf-8"))["models"]
    assert "<title>Retail</title>" in (tmp_path / "lineage.html").read_text(
        encoding="utf-8"
    )
    assert not (tmp_path / "verification.json").exists()


def test_trace_prints_the_hops_and_the_source_columns():
    code, out = cli("trace", "mart_rfm.monetary")
    assert code == 0 and "int_invoices.invoice_amount" in out
    assert "values computed from: raw_transactions.customer_id" not in out
    assert "raw_transactions.price" in out and "raw_transactions.quantity" in out
    code, out = cli("trace", "mart_rfm.monetary", "--indirect")
    assert "rows shaped by:" in out and "raw_transactions.stock_code" in out


def test_impact_groups_by_model_in_dependency_order():
    code, out = cli("impact", "raw_transactions.price")
    assert code == 0 and "reaches 18 columns in 9 models" in out
    assert out.index("stg_lines:") < out.index("mart_rfm:")
    code, out = cli("impact", "raw_transactions.stock_code", "--indirect")
    assert "reaches 65 columns in 9 models" in out


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["trace", "mart_rfm"], "expected table.column"),
        (["trace", "mart_rfm.nope"], "no column mart_rfm.nope"),
        (["impact", "ghost.x"], "no column ghost.x"),
        (["models", "--pipeline", "nowhere"], "pipeline folder not found"),
        (["verify", "--strategy", "explode"], "unknown strategy explode"),
    ],
)
def test_bad_input_is_a_clear_error_not_a_traceback(args, message):
    code, out = cli(*args)
    assert code == 2 and message in out


def test_verify_one_column_one_strategy_writes_the_json(tmp_path):
    code, out = cli(
        "verify",
        "--out",
        tmp_path,
        "--source",
        "raw_transactions.price",
        "--strategy",
        "shift",
    )
    assert code == 0 and "confirmed" in out and "0 pipeline" not in out
    data = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    assert data["runs"] == 3 and data["missed"] == []


def test_verify_exits_nonzero_when_the_lineage_missed_something(tmp_path):
    root = make_pipeline(
        tmp_path / "p",
        {"t": T_CSV, "u": U_CSV},
        {"m": "SELECT a, COLUMNS('^b$') FROM t"},
    )
    code, out = cli(
        "verify", "--pipeline", root, "--out", tmp_path / "o", "--strategy", "shift"
    )
    assert code == 1 and "schema mismatch" in out and "not in the lineage" in out
    code, out = cli("build", "--pipeline", root, "--out", tmp_path / "o2", "--verify")
    assert code == 1 and "warning: m: uses COLUMNS" in out
    assert (tmp_path / "o2" / "lineage.html").exists()


def test_the_committed_blind_spot_example_is_caught_by_the_data_check(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "blind_spots"
    code, out = cli("verify", "--pipeline", example, "--out", tmp_path)
    assert code == 1
    assert "orders.net_amount -> amounts_doubled.net_amount" in out
    assert "orders.tax_amount -> amounts_doubled.tax_amount" in out
