"""The HTML report: number formatting, palettes, self-containment, and escaping of hostile data."""

from __future__ import annotations

import json
import re
from itertools import pairwise

import polars as pl
import pytest
from conftest import grid_features
from sales_heatmap import geo_heatmap as gh
from sales_heatmap.geo_heatmap import OK, aggregate_to_regions, build_model
from sales_heatmap.report import (
    PALETTES,
    format_full,
    format_number,
    palette_colors,
    relative_luminance,
    render_html,
    safe_json,
)


class TestFormatting:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (
                150e9,
                "$150B",
            ),  # regression: trailing-zero stripping once turned this into $15B
            (15e9, "$15B"),
            (1_234_567, "$1.23M"),
            (999_960, "$1M"),  # rounds up across a unit boundary
            (1500, "$1.5K"),
            (100, "$100"),
            (12.34, "$12.3"),
            (0, "$0"),
            (0.5, "$0.5"),
            (-2.5e6, "-$2.5M"),
            (1e12, "$1T"),
        ],
    )
    def test_format_number(self, value, expected):
        assert format_number(value, "$") == expected

    def test_non_finite(self):
        assert (
            format_number(float("nan")) == "n/a"
            and format_number(None) == "n/a"
            and format_full(float("inf")) == "n/a"
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            (150_000_000_000, "$150,000,000,000"),
            (1234.5, "$1,234"),
            (12.5, "$12.5"),
            (0.123456, "$0.123"),
            (-7, "-$7"),
        ],
    )
    def test_format_full(self, value, expected):
        assert format_full(value, "$") == expected


class TestPalettes:
    @pytest.mark.parametrize("name", sorted(PALETTES))
    @pytest.mark.parametrize("k", range(2, 10))
    def test_sequential_and_well_formed(self, name, k):
        colors = palette_colors(name, k)
        assert len(colors) == k and all(
            re.fullmatch(r"#[0-9a-f]{6}", c) for c in colors
        )
        lum = [relative_luminance(c) for c in colors]
        # magnitude is carried by lightness, so it survives any colour-vision deficiency
        assert all(a > b for a, b in pairwise(lum))

    def test_unknown_palette(self):
        with pytest.raises(ValueError, match="unknown palette"):
            palette_colors("rainbow", 5)

    def test_luminance_extremes(self):
        assert relative_luminance("#000000") == 0 and relative_luminance(
            "#ffffff"
        ) == pytest.approx(1)


def test_safe_json_cannot_close_a_script_block():
    text = safe_json({"a": "</script><!-- &   <b>"})
    assert "<" not in text and ">" not in text and "&" not in text
    assert json.loads(text)["a"] == "</script><!-- &   <b>"


def make_model(tmp_path, name_of=None, keys=None):
    feats = grid_features(3, 3, name_of)
    path = tmp_path / "g.geojson"
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8"
    )
    b = gh.load_custom_boundaries(path, "id", "name")
    cells = keys or [
        "C00",
        "C10",
        "C20",
        "C01",
        "C11",
        "C21",
        "C02",
        "C12",
        "C22",
        "??",
    ]
    data = pl.DataFrame(
        {
            "cell": cells,
            "v": [str(i * i) for i in range(len(cells))],
            "p": ["10"] * len(cells),
        }
    )
    values, join = aggregate_to_regions(data, "cell", "v", "sum", b.index)
    pop = aggregate_to_regions(data, "cell", "p", "max", b.index)
    return build_model(
        b,
        values,
        join,
        title="My <b>title</b>",
        value_label="Sales",
        prefix="$",
        population=pop,
        permutations=19,
        lisa_permutations=19,
        source_note="Source & notes",
    )


def payload_of(text: str) -> dict:
    m = re.search(r"var D = (\{.*?\});\n  var views", text, re.DOTALL)
    assert m, "payload not found"
    return json.loads(m.group(1))


class TestRender:
    def test_is_self_contained(self, tmp_path):
        text = render_html(make_model(tmp_path))
        # no resource loads: scripts, stylesheets, images, fonts, tiles
        assert not re.search(r"<script[^>]+src=", text) and not re.search(
            r"<link[^>]+href=", text
        )
        assert "url(http" not in text and "@import" not in text
        assert "L.map(" in text and "leaflet" in text.lower()

    def test_payload_is_consistent_with_the_model(self, tmp_path):
        model = make_model(tmp_path)
        d = payload_of(render_html(model))
        keys = [v["key"] for v in d["views"]]
        assert (
            keys.count("lisa") == 1 and len(keys) == 9
        )  # 2 measures x 4 schemes + hot/cold spots
        n = len(model.boundaries.features)
        for v in d["views"]:
            assert len(v["fill"]) == n and max(v["fill"]) < len(v["palette"])
            assert (
                sum(row["count"] for row in v["legend"]) == n
            )  # legend accounts for every region
        assert d["defaultMeasure"] == "percap" and {
            m["key"] for m in d["measures"]
        } == {"raw", "percap", "lisa"}
        assert len(d["bounds"]) == 2

    def test_class_colours_follow_the_palette_and_statuses_are_grey(self, tmp_path):
        model = make_model(tmp_path)
        d = payload_of(render_html(model))
        view = next(v for v in d["views"] if v["key"] == "raw:quantile")
        n_classes = len(model.views[0].edges) - 1
        assert view["palette"][:n_classes] == palette_colors(model.palette, n_classes)
        assert all(
            c in ("#7f7f7f", "#d4d4d4", "#a9a9a9") for c in view["palette"][n_classes:]
        )

    def test_missing_region_is_not_coloured_like_the_lowest_class(self, tmp_path):
        model = make_model(
            tmp_path, keys=["C00", "C10", "C20", "C01"]
        )  # 5 cells have no row
        d = payload_of(render_html(model))
        view = next(v for v in d["views"] if v["key"] == "raw:quantile")
        assert view["palette"][view["fill"][8]] != view["palette"][view["fill"][0]]
        assert any(
            "No row" in row["label"] and row["count"] == 5 for row in view["legend"]
        )

    def test_report_sections_present(self, tmp_path):
        text = render_html(make_model(tmp_path))
        for heading in (
            "Data join",
            "Why raw totals mislead",
            "Classification schemes compared",
            "Spatial autocorrelation",
        ):
            assert heading in text
        assert "1 distinct key(s) did not match" in text and "<code>??</code>" in text

    def test_title_and_notes_are_escaped(self, tmp_path):
        text = render_html(make_model(tmp_path))
        assert (
            "<title>My &lt;b&gt;title&lt;/b&gt;</title>" in text
            and "Source &amp; notes" in text
        )
        assert "<b>title</b>" not in text

    def test_tooltips_carry_values_and_statuses(self, tmp_path):
        model = make_model(tmp_path, keys=["C00", "C10", "C20"])
        text = render_html(model)
        assert "Sales: $" in text and "No row for this region in the data" in text

    def test_hostile_region_names_and_keys_cannot_inject_markup(self, tmp_path):
        evil = "</script><img src=x onerror=alert(1)>\"'&"
        model = make_model(
            tmp_path,
            name_of=lambda k: evil if k == "C11" else f"cell {k}",
            keys=["C00", "C10", evil, "C11"],
        )
        text = render_html(model)
        assert "<img src=x" not in text and "</script><img" not in text
        assert (
            "&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt;" in text
        )  # shown, but as text
        # the script block count is exactly what folium + leaflet + our controls emit: nothing was added
        assert text.count("</script>") == text.count("<script")

    def test_hostile_csv_value_column_header_and_label_are_escaped(self, tmp_path):
        model = make_model(tmp_path)
        model.measures["raw"].label = "<img src=x onerror=1>"
        model.measures["percap"].label = "</script><svg onload=1>"
        text = render_html(model)
        assert (
            "<img src=x" not in text
            and "<svg onload" not in text
            and "</script><svg" not in text
        )

    def test_no_data_at_all_still_renders(self, tmp_path):
        model = make_model(tmp_path, keys=["??"])
        text = render_html(model)
        assert "L.map(" in text and model.measures["raw"].status.tolist().count(OK) == 0

    def test_state_level_render_has_insets_and_labels(self):
        b = gh.load_bundled_boundaries("state")
        sales, pop = gh.load_bundled_tables("state", "44-45")
        values, join = aggregate_to_regions(sales, "fips", "sales", "sum", b.index)
        pv = aggregate_to_regions(pop, "fips", "population", "sum", b.index)
        d = payload_of(
            render_html(
                build_model(
                    b,
                    values,
                    join,
                    title="t",
                    value_label="Sales",
                    population=pv,
                    permutations=19,
                    lisa_permutations=19,
                )
            )
        )
        assert {i["label"] for i in d["insets"]} == {
            "Alaska (35% scale)",
            "Hawaii",
            "Puerto Rico",
        }
        assert d["weight"] > 0.5
