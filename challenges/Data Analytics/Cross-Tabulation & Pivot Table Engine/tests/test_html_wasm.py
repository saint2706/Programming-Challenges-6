"""The exported page, and the committed WebAssembly module against the native engine."""

import base64
import gzip
import json
import re
import shutil
import subprocess

import pytest
from conftest import ADULT, ROOT, UCB
from pivot_tool.engine import Table, read_bytes
from pivot_tool.html_export import WASM_NAME, build_html, embed_json, web_asset
from pivot_tool.presets import ADULT as ADULT_SET
from pivot_tool.presets import UCB as UCB_SET

WEB = ROOT / "src" / "pivot_tool" / "web"


def block(page: str, name: str) -> str:
    return re.search(
        rf'<script id="{name}" type="[^"]+">(.*?)</script>', page, re.DOTALL
    ).group(1)


def test_page_is_self_contained_and_its_payloads_decode():
    csv = read_bytes(UCB)
    page = build_html(csv, "Berkeley", presets=UCB_SET.presets)
    assert not re.findall(r"https?://", page), (
        "the page must not reference anything outside itself"
    )
    assert '<link rel="stylesheet"' not in page and 'src="http' not in page
    assert gzip.decompress(base64.b64decode(block(page, "data-b64"))) == csv
    wasm = gzip.decompress(base64.b64decode(block(page, "wasm-b64")))
    assert wasm[:4] == b"\0asm" and wasm == (WEB / WASM_NAME).read_bytes()
    config = json.loads(block(page, "config"))
    assert config["title"] == "Berkeley" and len(config["presets"]) == 3


def test_hostile_titles_and_preset_names_cannot_break_out_of_the_page():
    evil = "</script><img src=x onerror=alert(1)>"
    page = build_html(
        b"a,b\n1,2\n",
        "</title><script>alert(1)</script>",
        presets=[{"name": evil, "description": evil, "spec": {}}],
    )
    assert "<img src=x" not in page and "<script>alert" not in page
    assert (
        page.count("</script") == 5
    )  # config, wasm, data, engine, app: nothing from the data
    assert "&lt;/title&gt;" in page
    config = json.loads(block(page, "config"))
    assert config["presets"][0]["name"] == evil  # the escaping is lossless


def test_embed_json_escapes_script_breakers_and_line_separators():
    text = embed_json({"s": "a" + chr(0x2028) + "b" + chr(0x2029) + "</script>&"})
    assert (
        chr(0x2028) not in text
        and chr(0x2029) not in text
        and "</script>" not in text
        and "&" not in text
    )
    assert json.loads(text)["s"] == "a" + chr(0x2028) + "b" + chr(0x2029) + "</script>&"


@pytest.mark.parametrize("asset", ["app.js", "engine.js"])
def test_page_code_never_writes_html_from_data(asset):
    source = web_asset(asset).decode("utf-8")
    for banned in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "ev" + "al(",
        "new Function",
    ):
        assert banned not in source, banned


def test_presets_all_run_on_their_own_dataset():
    for ds in (ADULT_SET, UCB_SET):
        with Table.from_path(ds.path) as t:
            for preset in ds.presets:
                if preset.get("tab") == "crosstab":
                    spec = {
                        "row": preset["spec"]["rows"][0],
                        "col": preset["spec"]["cols"][0],
                    }
                    if preset.get("layer"):
                        spec["layer"] = {"field": preset["layer"]}
                    assert t.crosstab(spec)["n"] > 0
                else:
                    result = t.pivot(preset["spec"])
                    assert result["rows"] and result["cells"], preset["name"]


# -- the browser's engine is the native engine ------------------------------------------------------
SPECS = [
    {
        "rows": [{"field": "education"}],
        "cols": [{"field": "sex"}],
        "values": [{"field": "hours_per_week", "agg": "mean"}, {"agg": "count"}],
    },
    {
        "rows": [
            {"field": "workclass"},
            {"field": "occupation", "top": {"n": 3, "measure": "count"}},
        ],
        "cols": [{"field": "income"}],
        "values": [
            {"agg": "count", "show_as": {"kind": "pct_row"}},
            {"field": "age", "agg": "median"},
        ],
        "options": {"subtotals": True},
    },
    {
        "rows": [{"field": "age", "group": {"kind": "bin", "width": 10}}],
        "values": [
            {"field": "capital_gain", "agg": "std"},
            {"field": "age", "agg": "distinct"},
        ],
    },
    {
        "rows": [{"field": "marital_status"}],
        "values": [
            {"agg": "count", "label": "n"},
            {"field": "hours_per_week", "agg": "sum", "label": "h"},
            {"calc": "h / n", "label": "per"},
        ],
    },
]
CROSSTAB = {
    "row": {"field": "sex"},
    "col": {"field": "income"},
    "layer": {"field": "marital_status"},
}
NODE_SCRIPT = """
const fs = require("fs"), zlib = require("zlib");
const P = require(process.argv[2]);
(async () => {
  const eng = await P.Engine.load(fs.readFileSync(process.argv[3]));
  const info = eng.call("load_csv", { trim: true }, zlib.gunzipSync(fs.readFileSync(process.argv[4])));
  const specs = JSON.parse(process.argv[5]);
  const out = specs.map((s) => eng.call("pivot", { table: info.table, spec: s }));
  out.push(eng.call("crosstab", { table: info.table, spec: JSON.parse(process.argv[6]) }));
  out.push(eng.call("drill", { table: info.table, spec: specs[0], row_path: ["Masters"], col_path: ["Female"], limit: 3 }));
  process.stdout.write(JSON.stringify(out));
})();
"""


def same(a, b, path="$"):
    if isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            same(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_committed_wasm_gives_the_native_engines_answers(adult, tmp_path):
    script = tmp_path / "check.js"
    script.write_text(NODE_SCRIPT, encoding="utf-8")
    run = subprocess.run(
        [
            "node",
            str(script),
            str(WEB / "engine.js"),
            str(WEB / WASM_NAME),
            str(ADULT),
            json.dumps(SPECS),
            json.dumps(CROSSTAB),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    from_wasm = json.loads(run.stdout)
    native = [adult.pivot(s) for s in SPECS]
    native.append(adult.crosstab(CROSSTAB))
    native.append(adult.drill(SPECS[0], ["Masters"], ["Female"], 3))
    assert len(from_wasm) == len(native)
    for got, want in zip(from_wasm, native, strict=True):
        same(got, want)
