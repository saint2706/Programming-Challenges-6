"""Render a :class:`~geo_heatmap.MapModel` to one self-contained HTML file (Folium + inlined Leaflet).

Nothing is fetched at view time: Leaflet's JS/CSS are inlined from ``vendor/``, the map has no
tile layer, and the boundaries are embedded.  Every data-derived string reaches the page either
through :func:`html.escape` (page text, tooltips) or through ``textContent`` (the JS controls).
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import folium
import numpy as np
from branca.element import Element, MacroElement
from classify import SCHEME_LABELS, SCHEMES
from geo_heatmap import (
    BLANK,
    NO_POPULATION,
    NO_RECORD,
    OK,
    STATUS_LABELS,
    MapModel,
    View,
)
from jinja2 import Template
from projection import group_of, inset_boxes
from spatial import CLUSTER_LABELS

VENDOR = Path(__file__).resolve().parent / "vendor"

# Anchor colours, light (low values) -> dark (high values), interpolated to the requested class count.
# viridis and cividis are perceptually uniform and designed for colour-vision deficiency;
# YlOrRd is ColorBrewer's colour-blind-safe sequential ramp.
PALETTES: dict[str, list[str]] = {
    "viridis": [
        "#fde725",
        "#b5de2b",
        "#6ece58",
        "#35b779",
        "#1f9e89",
        "#26828e",
        "#31688e",
        "#3e4989",
        "#482878",
        "#440154",
    ],
    "cividis": ["#ffea46", "#cbba69", "#958f78", "#666970", "#31446b", "#00204d"],
    "ylorrd": [
        "#ffffcc",
        "#ffeda0",
        "#fed976",
        "#feb24c",
        "#fd8d3c",
        "#fc4e2a",
        "#e31a1c",
        "#bd0026",
        "#800026",
    ],
}
STATUS_COLORS = {BLANK: "#7f7f7f", NO_RECORD: "#d4d4d4", NO_POPULATION: "#a9a9a9"}
# Okabe-Ito based, separated by hue *and* lightness; "not significant" stays out of the way.
CLUSTER_COLORS = {0: "#eeeeee", 1: "#d55e00", 2: "#0072b2", 3: "#f2b98f", 4: "#9ecae1"}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def safe_json(obj: object) -> str:
    """JSON for embedding in a ``<script>``: no ``</script>``, ``<!--`` or U+2028 can end the block early."""
    text = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


# --------------------------------------------------------------------------- #
# Colour and number helpers
# --------------------------------------------------------------------------- #


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def palette_colors(name: str, k: int) -> list[str]:
    """``k`` colours from ``name``, ordered from the *lowest* class (light) to the *highest* (dark/saturated)."""
    if name not in PALETTES:
        raise ValueError(f"unknown palette {name!r}; choose from {sorted(PALETTES)}")
    anchors = PALETTES[name]
    rgb = [_hex_to_rgb(a) for a in anchors]
    out = []
    for i in range(k):
        t = i / (k - 1) if k > 1 else 0.0
        pos = t * (len(rgb) - 1)
        lo = min(math.floor(pos), len(rgb) - 2)
        frac = pos - lo
        mix = [
            round(rgb[lo][c] + (rgb[lo + 1][c] - rgb[lo][c]) * frac) for c in range(3)
        ]
        out.append("#{:02x}{:02x}{:02x}".format(*mix))
    return out


def relative_luminance(color: str) -> float:
    """WCAG relative luminance of an sRGB hex colour."""

    def lin(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in _hex_to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _trim(x: float, digits: int) -> str:
    """Fixed decimals with a trailing fractional zero dropped (never the zeros of an integer: 150 stays 150)."""
    text = f"{x:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_number(value: float | None, prefix: str = "") -> str:
    """Compact human form: 1234 -> 1.23K, 5.2e9 -> 5.2B.  Sign is kept."""
    if value is None or not math.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    a = abs(value)
    units = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))
    for i, (div, suffix) in enumerate(units):
        if a >= div:
            x = a / div
            digits = 0 if x >= 100 else 1 if x >= 10 else 2
            if (
                round(x, digits) >= 1000 and i > 0
            ):  # 999.96K rounds to 1000K: promote to 1M
                div, suffix = units[i - 1]
                x, digits = a / div, 2
            return f"{sign}{prefix}{_trim(x, digits)}{suffix}"
    digits = 0 if a >= 100 else 1 if a >= 10 else 2
    return f"{sign}{prefix}{_trim(a, digits)}"


def format_full(value: float | None, prefix: str = "") -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    a = abs(value)
    if a >= 1000:
        body = f"{a:,.0f}"
    elif a >= 1:
        body = _trim(a, 2)
    else:
        body = f"{a:.3g}"
    return f"{'-' if value < 0 else ''}{prefix}{body}"


# --------------------------------------------------------------------------- #
# Views -> JSON payload for the in-page controls
# --------------------------------------------------------------------------- #


def _class_labels(view: View, prefix: str) -> list[str]:
    e = view.edges
    labels = []
    for i in range(len(e) - 1):
        lo, hi = (
            format_number(float(e[i]), prefix),
            format_number(float(e[i + 1]), prefix),
        )
        labels.append(f"{lo} to {hi}" if lo != hi else lo)
    return labels


def _view_payload(model: MapModel, view: View) -> dict:
    measure = model.measures[view.measure]
    colors = palette_colors(model.palette, max(len(view.edges) - 1, 1))
    n_classes = len(colors)
    legend = [
        {"color": colors[i], "label": label, "count": view.counts[i]}
        for i, label in enumerate(_class_labels(view, model.prefix))
    ]
    palette = list(colors)
    status_slot: dict[int, int] = {}
    fills = []
    for cls in view.classes.tolist():
        if cls >= 0:
            fills.append(cls)
            continue
        status = -cls
        if status not in status_slot:
            status_slot[status] = len(palette)
            palette.append(STATUS_COLORS[status])
        fills.append(status_slot[status])
    for status, slot in status_slot.items():
        legend.append(
            {
                "color": palette[slot],
                "label": STATUS_LABELS[status],
                "count": int((view.classes == -status).sum()),
            }
        )
    del n_classes
    return {
        "key": view.key,
        "measure": view.measure,
        "scheme": view.scheme,
        "title": f"{measure.label}, {SCHEME_LABELS[view.scheme].lower()} classes",
        "palette": palette,
        "fill": fills,
        "legend": legend,
        "gvf": round(view.gvf, 4),
    }


def _lisa_payload(model: MapModel) -> dict | None:
    lisa = model.lisa
    if lisa is None:
        return None
    label = model.measures[lisa.measure].label
    palette = [CLUSTER_COLORS[c] for c in range(5)] + [STATUS_COLORS[NO_RECORD]]
    fills = [5 if c < 0 else int(c) for c in lisa.cluster.tolist()]
    counts = {c: int((lisa.cluster == c).sum()) for c in range(5)}
    legend = [
        {"color": CLUSTER_COLORS[c], "label": CLUSTER_LABELS[c], "count": counts[c]}
        for c in (1, 2, 3, 4, 0)
    ]
    not_analysed = int((lisa.cluster < 0).sum())
    if not_analysed:
        legend.append(
            {
                "color": STATUS_COLORS[NO_RECORD],
                "label": "Not analysed (no value or no neighbours)",
                "count": not_analysed,
            }
        )
    return {
        "key": "lisa",
        "measure": "lisa",
        "scheme": "",
        "title": f"Hot and cold spots of {label.lower()} (local Moran's I, FDR-controlled)",
        "palette": palette,
        "fill": fills,
        "legend": legend,
        "gvf": None,
    }


def _tooltips(model: MapModel) -> list[str]:
    names = model.boundaries.index.names
    raw = model.measures["raw"]
    pc = model.measures.get("percap")
    pop = model.population
    lisa = model.lisa
    out = []
    for i, name in enumerate(names):
        rows = [f"<b>{esc(name)}</b>"]
        rows.append(_tt_line(raw.label, raw.values[i], raw.status[i], model.prefix))
        if pc is not None:
            rows.append(_tt_line(pc.label, pc.values[i], pc.status[i], model.prefix))
            if pop is not None and np.isfinite(pop[i]):
                rows.append(f"Population: {esc(format_full(float(pop[i])))}")
        if lisa is not None and lisa.cluster[i] >= 0:
            rows.append(f"Cluster: {esc(CLUSTER_LABELS[int(lisa.cluster[i])])}")
        out.append("<br>".join(rows))
    return out


def _tt_line(label: str, value: float, status: int, prefix: str) -> str:
    if status == OK:
        return f"{esc(label)}: {esc(format_full(float(value), prefix))}"
    return f"{esc(label)}: <i>{esc(STATUS_LABELS[int(status)])}</i>"


# --------------------------------------------------------------------------- #
# The Leaflet control
# --------------------------------------------------------------------------- #

_JS = r"""
(function () {
  var map = __MAP__;
  var layer = __LAYER__;
  var D = __PAYLOAD__;
  var views = {};
  D.views.forEach(function (v) { views[v.key] = v; });
  var current = null;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  var legendBox = el('div', 'gh-panel gh-legend');
  var legendControl = L.control({position: 'bottomright'});
  legendControl.onAdd = function () { return legendBox; };
  legendControl.addTo(map);

  function paint(view) {
    current = view;
    layer.eachLayer(function (l) {
      var i = l.feature.properties.i;
      l.setStyle({fillColor: view.palette[view.fill[i]], fillOpacity: 0.95, weight: D.weight, color: '#ffffff', opacity: 1});
    });
    legendBox.textContent = '';
    legendBox.appendChild(el('div', 'gh-title', view.title));
    view.legend.forEach(function (row) {
      var line = el('div', 'gh-row');
      var sw = el('span', 'gh-swatch');
      sw.style.background = row.color;
      line.appendChild(sw);
      line.appendChild(el('span', 'gh-label', row.label));
      line.appendChild(el('span', 'gh-count', String(row.count)));
      legendBox.appendChild(line);
    });
    if (view.gvf !== null) {
      legendBox.appendChild(el('div', 'gh-note', 'Goodness of variance fit: ' + view.gvf.toFixed(3)));
    }
    if (D.insetNote) legendBox.appendChild(el('div', 'gh-note', D.insetNote));
  }

  function currentKey() {
    var m = measureSel.value;
    return m === 'lisa' ? 'lisa' : m + ':' + schemeSel.value;
  }
  function refresh() {
    schemeSel.disabled = measureSel.value === 'lisa';
    paint(views[currentKey()]);
  }

  var panel = el('div', 'gh-panel gh-controls');
  function field(labelText, select) {
    var wrap = el('label', 'gh-field');
    wrap.appendChild(el('span', 'gh-fieldlabel', labelText));
    wrap.appendChild(select);
    return wrap;
  }
  var measureSel = el('select');
  D.measures.forEach(function (m) {
    var o = el('option', '', m.label); o.value = m.key; measureSel.appendChild(o);
  });
  var schemeSel = el('select');
  D.schemes.forEach(function (s) {
    var o = el('option', '', s.label); o.value = s.key; schemeSel.appendChild(o);
  });
  measureSel.value = D.defaultMeasure;
  schemeSel.value = D.defaultScheme;
  measureSel.addEventListener('change', refresh);
  schemeSel.addEventListener('change', refresh);
  panel.appendChild(field('Show', measureSel));
  panel.appendChild(field('Classes', schemeSel));
  var ctl = L.control({position: 'topright'});
  ctl.onAdd = function () {
    L.DomEvent.disableClickPropagation(panel);
    return panel;
  };
  ctl.addTo(map);

  layer.eachLayer(function (l) {
    l.bindTooltip(l.feature.properties.tt, {sticky: true, className: 'gh-tooltip'});
    l.on('mouseover', function () { l.setStyle({weight: 2.2, color: '#111111'}); l.bringToFront(); });
    l.on('mouseout', function () { l.setStyle({weight: D.weight, color: '#ffffff'}); });
  });

  D.insets.forEach(function (ins) {
    L.tooltip({permanent: true, direction: 'top', className: 'gh-inset', interactive: false})
      .setContent(ins.label).setLatLng([ins.y, ins.x]).addTo(map);
  });

  // folium clamps minZoom at 0, but a CRS.Simple map measured in km needs negative zoom levels to fit
  map.setMinZoom(-6);
  map.setMaxZoom(4);
  map.options.zoomSnap = 0.25;
  // keep the right-hand strip clear for the legend and controls on wide screens
  var gutter = map.getSize().x >= 900 ? 350 : 8;
  map.fitBounds(D.bounds, {paddingTopLeft: [8, 8], paddingBottomRight: [gutter, 8]});
  refresh();
})();
"""

_CSS = """
html, body { height: auto !important; margin: 0; }
body { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; color: #1c1f23; background: #f6f7f9; }
.folium-map { position: relative !important; height: 78vh !important; min-height: 420px; background: #e9eef3; }
.gh-panel { background: rgba(255,255,255,.96); color: #1c1f23; border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,.3);
  padding: 8px 10px; font-size: 13px; line-height: 1.35; }
.gh-controls { display: grid; gap: 6px; }
.gh-field { display: grid; gap: 2px; }
.gh-fieldlabel { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #555; }
.gh-controls select { font: inherit; padding: 3px 4px; }
.gh-legend { max-width: 330px; }
.gh-title { font-weight: 600; margin-bottom: 4px; }
.gh-row { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
.gh-swatch { width: 16px; height: 12px; border: 1px solid rgba(0,0,0,.25); flex: none; }
.gh-label { flex: 1; }
.gh-count { color: #555; font-variant-numeric: tabular-nums; }
.gh-note { margin-top: 4px; font-size: 11px; color: #555; }
.gh-inset { background: none; border: none; box-shadow: none; font-size: 11px; color: #444; font-weight: 600; }
.gh-inset::before { display: none; }
main.gh-report { max-width: 1040px; margin: 0 auto; padding: 20px 16px 48px; }
main.gh-report h1 { font-size: 1.5rem; margin: 0 0 4px; }
main.gh-report h2 { font-size: 1.15rem; margin: 28px 0 8px; }
main.gh-report p, main.gh-report li { line-height: 1.5; }
.gh-lede { color: #4a5058; margin: 0 0 8px; }
.gh-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.gh-table-wrap { overflow-x: auto; }
table.gh { border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }
table.gh th, table.gh td { border: 1px solid #d9dde2; padding: 4px 8px; text-align: left; }
table.gh td.num, table.gh th.num { text-align: right; font-variant-numeric: tabular-nums; }
table.gh th { background: #eef1f4; }
.gh-warn { background: #fff4e0; border: 1px solid #f0c987; padding: 8px 12px; border-radius: 6px; }
.gh-small { font-size: 12px; color: #555; }
@media (prefers-color-scheme: dark) {
  body { background: #14171a; color: #e6e8ea; }
  .gh-lede, .gh-small { color: #a5adb5; }
  table.gh { background: #1b1f23; } table.gh th { background: #262b30; }
  table.gh th, table.gh td { border-color: #39414a; }
  .gh-warn { background: #3a2f1a; border-color: #6b5a2e; }
}
"""


class _Controls(MacroElement):
    """Adds the view selector, legend, tooltips, insets and initial extent to the map."""

    _template = Template(
        "{% macro script(this, kwargs) %}{{ this.js|safe }}{% endmacro %}"
    )

    def __init__(self, map_name: str, layer_name: str, payload: str):
        super().__init__()
        self._name = "GeoHeatmapControls"
        # payload is already script-safe (see safe_json); the names are folium-generated identifiers
        self.js = (
            _JS.replace("__MAP__", map_name)
            .replace("__LAYER__", layer_name)
            .replace("__PAYLOAD__", payload)
        )


# --------------------------------------------------------------------------- #
# The report below the map
# --------------------------------------------------------------------------- #


def _table(
    headers: list[str], rows: list[list[str]], numeric: set[int] | None = None
) -> str:
    numeric = numeric or set()
    head = "".join(
        f'<th class="{"num" if i in numeric else ""}">{esc(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if i in numeric else ""}">{cell}</td>'
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f'<div class="gh-table-wrap"><table class="gh"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _join_section(model: MapModel) -> str:
    j = model.join
    parts = [
        "<h2>Data join</h2>",
        f"<p>{j.rows:,} data rows; {j.matched_rows:,} matched a map region and were combined into "
        f"{j.matched_regions:,} of {j.total_regions:,} regions"
        + (
            f" ({j.duplicate_rows_merged:,} extra rows merged with <code>{esc(model.aggregation)}</code>)"
            if j.duplicate_rows_merged
            else ""
        )
        + f". {j.blank_values:,} blank and {j.unparseable_values:,} non-numeric value cells were treated as <em>missing</em>, never as zero.</p>",
    ]
    if j.unmatched:
        rows = [
            [f"<code>{esc(k)}</code>", esc(reason), f"{n:,}"]
            for k, reason, n in j.unmatched[:25]
        ]
        more = f" (showing 25 of {len(j.unmatched):,})" if len(j.unmatched) > 25 else ""
        parts.append(
            f"<p class='gh-warn'>{len(j.unmatched):,} distinct key(s) did not match any region and are absent from the map{more}:</p>"
        )
        parts.append(_table(["Key in data", "Why", "Rows"], rows, {2}))
    else:
        parts.append("<p>Every key matched a region.</p>")
    status_rows = []
    for m in model.measures.values():
        counts = {
            s: int((m.status == s).sum()) for s in (OK, BLANK, NO_RECORD, NO_POPULATION)
        }
        status_rows.append(
            [
                esc(m.label),
                f"{counts[OK]:,}",
                f"{counts[BLANK]:,}",
                f"{counts[NO_RECORD]:,}",
                f"{counts[NO_POPULATION]:,}",
            ]
        )
    parts.append(
        _table(
            ["Measure", "Has value", "Blank / withheld", "No row", "No population"],
            status_rows,
            {1, 2, 3, 4},
        )
    )
    missing = j.regions_without_record
    if missing:
        sample = ", ".join(esc(n) for n in missing[:12]) + (
            " …" if len(missing) > 12 else ""
        )
        parts.append(
            f"<p class='gh-small'>Regions with no row in the data ({len(missing):,}): {sample}. They are drawn grey, not as the lowest class.</p>"
        )
    if model.population_join and model.population_join.unmatched:
        pj = model.population_join
        parts.append(
            f"<p class='gh-small'>Population file: {len(pj.unmatched):,} key(s) did not match a region.</p>"
        )
    return "".join(parts)


def _top_table(model: MapModel, key: str, n: int = 10) -> str:
    m = model.measures[key]
    names = model.boundaries.index.names
    ok = np.flatnonzero(m.status == OK)
    order = ok[np.argsort(-m.values[ok], kind="stable")][:n]
    other = model.measures.get("percap" if key == "raw" else "raw")
    rank_other = {}
    if other is not None:
        ok2 = np.flatnonzero(other.status == OK)
        for r, idx in enumerate(ok2[np.argsort(-other.values[ok2], kind="stable")], 1):
            rank_other[int(idx)] = r
    rows = []
    for r, idx in enumerate(order, 1):
        row = [
            str(r),
            esc(names[idx]),
            esc(format_number(float(m.values[idx]), model.prefix)),
        ]
        if other is not None:
            row.append(str(rank_other.get(int(idx), "n/a")))
        rows.append(row)
    headers = ["#", "Region", m.label] + (
        [f"Rank by {other.label.lower()}"] if other is not None else []
    )
    return _table(headers, rows, {0, 2, 3})


def _raw_vs_percap_section(model: MapModel) -> str:
    if "percap" not in model.measures:
        return ""
    s = model.stats
    rho, overlap = s.get("spearman_raw_population"), s.get("top_decile_overlap")
    raw_label, pc_label = model.measures["raw"].label, model.measures["percap"].label
    return (
        "<h2>Why raw totals mislead</h2>"
        f"<p>Across regions, <strong>{esc(raw_label.lower())}</strong> and <strong>population</strong> have a Spearman rank "
        f"correlation of <strong>{rho:.2f}</strong>: a map of raw totals is largely a map of where people live. "
        f"Only <strong>{overlap:.0%}</strong> of the top-decile regions by total are also in the top decile "
        f"{esc(pc_label.lower())}.</p>"
        '<div class="gh-cols"><div>'
        f"<h3>Top 10 by {esc(raw_label.lower())}</h3>{_top_table(model, 'raw')}</div><div>"
        f"<h3>Top 10 {esc(pc_label.lower())}</h3>{_top_table(model, 'percap')}</div></div>"
        "<p class='gh-small'>Per-resident figures for small populations are noisy, and retail sales are booked where the "
        "store is, not where the shopper lives: a county with a regional mall or warehouse club can exceed 100% "
        "of what its residents could plausibly spend.</p>"
    )


def _classification_section(model: MapModel) -> str:
    parts = [
        "<h2>Classification schemes compared</h2>",
        (
            "<p>The same numbers, classed four ways. <strong>Goodness of variance fit</strong> (GVF, 1 = the classes explain "
            "all the variance) says how faithfully a scheme represents the data; it does not say which map is <em>clearer</em>: "
            "quantile classes maximise visual contrast at the cost of hiding outliers.</p>"
        ),
    ]
    for key, m in model.measures.items():
        rows = []
        for v in (x for x in model.views if x.measure == key):
            breaks = (
                ", ".join(
                    esc(format_number(float(e), model.prefix)) for e in v.edges[1:-1]
                )
                or "none"
            )
            counts = " / ".join(str(c) for c in v.counts)
            rows.append(
                [
                    esc(SCHEME_LABELS[v.scheme]),
                    str(len(v.edges) - 1),
                    f"{v.gvf:.3f}",
                    breaks,
                    counts,
                ]
            )
        parts.append(f"<h3>{esc(m.label)}</h3>")
        parts.append(
            _table(
                ["Scheme", "Classes", "GVF", "Upper breaks", "Regions per class"],
                rows,
                {1, 2},
            )
        )
    return "".join(parts)


def _spatial_section(model: MapModel) -> str:
    parts = ["<h2>Spatial autocorrelation</h2>"]
    rows = []
    for key, g in model.moran.items():
        label = esc(model.measures[key].label)
        if g is None:
            rows.append([label, "n/a", "", "", "", "", ""])
            continue
        rows.append(
            [
                label,
                f"{g.i:.3f}",
                f"{g.expected:.4f}",
                f"{g.z_score:.1f}",
                f"{g.p_value:.3f}",
                f"{g.n:,}",
                f"{g.islands}",
            ]
        )
    parts.append(
        "<p>Global Moran's <em>I</em> (queen contiguity, row-standardised; "
        f"{model.permutations:,}-permutation reference distribution). +1 means neighbours resemble each other, "
        "about 0 means no spatial pattern.</p>"
    )
    parts.append(
        _table(
            [
                "Measure",
                "I",
                "E[I]",
                "z (perm.)",
                "pseudo p",
                "Regions",
                "Islands dropped",
            ],
            rows,
            {1, 2, 3, 4, 5, 6},
        )
    )
    lisa = model.lisa
    if lisa is not None:
        names = model.boundaries.index.names
        hh = [names[i] for i in np.flatnonzero(lisa.cluster == 1)]
        ll = [names[i] for i in np.flatnonzero(lisa.cluster == 2)]
        r = lisa.result
        parts.append(
            f"<p>Local Moran's <em>I</em> on {esc(model.measures[lisa.measure].label.lower())} "
            f"({r.permutations:,} conditional permutations per region). Uncorrected, <strong>{r.n_uncorrected:,}</strong> of "
            f"{r.p_value.size:,} regions test significant at 5%, but with that many tests about 5% would pass by chance "
            f"alone. Controlling the false discovery rate (Benjamini-Hochberg) leaves <strong>{r.n_significant:,}</strong>: "
            f"<strong>{len(hh):,}</strong> hot spots and <strong>{len(ll):,}</strong> cold spots, plus "
            f"{int((lisa.cluster == 3).sum()):,} high-low and {int((lisa.cluster == 4).sum()):,} low-high outliers. "
            "Toggle <em>Hot and cold spots</em> on the map.</p>"
        )
    return "".join(parts)


def _page_body(model: MapModel) -> str:
    notes = "".join(f"<p class='gh-warn'>{esc(n)}</p>" for n in model.notes)
    return (
        '<main class="gh-report">'
        f"<h1>{esc(model.title)}</h1>"
        f"<p class='gh-lede'>{esc(model.source_note)}</p>"
        "<p class='gh-small'>Equal-area Albers projection so colour-over-area is honest; Alaska is drawn at 35% scale and "
        "Hawaii and Puerto Rico are moved, as on standard U.S. thematic maps. Hover a region for its values; "
        "grey regions have no value, which is not the same as zero.</p>"
        f"{notes}{_join_section(model)}{_raw_vs_percap_section(model)}{_classification_section(model)}{_spatial_section(model)}"
        "</main>"
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _bounds(fc: dict) -> list[list[float]]:
    xs, ys = [], []
    for feat in fc["features"]:
        geom = feat["geometry"]
        polys = (
            [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        )
        for poly in polys:
            for ring in poly:
                for x, y in ring:
                    xs.append(x)
                    ys.append(y)
    return [[min(ys), min(xs)], [max(ys), max(xs)]]


def _insets(model: MapModel) -> list[dict]:
    b = model.boundaries
    if b.projection != "albers-usa":
        return []
    fips_of = (
        (lambda f: f["properties"]["fips"])
        if b.level == "state"
        else (lambda f: f["properties"]["state_fips"])
    )
    boxes = inset_boxes(model.collection, fips_of)
    labels = {"AK": "Alaska (35% scale)", "HI": "Hawaii", "PR": "Puerto Rico"}
    return [
        {"label": labels[g], "x": (x0 + x1) / 2, "y": y1}
        for g, (x0, _y0, x1, y1) in boxes.items()
        if g in labels
    ]


def render_html(model: MapModel) -> str:
    """The whole report as one HTML string with no external resource references."""
    features = []
    tips = _tooltips(model)
    for i, feat in enumerate(model.collection["features"]):
        features.append(
            {
                "type": "Feature",
                "properties": {"i": i, "tt": tips[i]},
                "geometry": feat["geometry"],
            }
        )
    collection = {"type": "FeatureCollection", "features": features}

    view_payloads = [_view_payload(model, v) for v in model.views]
    lisa_payload = _lisa_payload(model)
    if lisa_payload:
        view_payloads.append(lisa_payload)
    measures = [{"key": k, "label": m.label} for k, m in model.measures.items()]
    if lisa_payload:
        measures.append({"key": "lisa", "label": "Hot and cold spots"})
    default_measure = "percap" if "percap" in model.measures else "raw"
    plate = model.boundaries.projection != "albers-usa"
    payload = {
        "views": view_payloads,
        "measures": measures,
        "schemes": [{"key": s, "label": SCHEME_LABELS[s]} for s in SCHEMES],
        "defaultMeasure": default_measure,
        "defaultScheme": "quantile",
        "bounds": _bounds(collection),
        "insets": _insets(model),
        "insetNote": None
        if plate
        else "Alaska, Hawaii and Puerto Rico are inset; Alaska at 35% scale.",
        "weight": 0.35 if model.boundaries.level == "county" else 0.9,
    }

    m = folium.Map(
        location=[0, 0],
        zoom_start=0,
        crs="Simple",
        tiles=None,
        prefer_canvas=True,
        zoom_control=True,
        min_zoom=-6,
        max_zoom=6,
        zoom_snap=0.25,
        attributionControl=False,
    )
    m.default_js = []  # Leaflet is inlined below instead of loaded from a CDN
    m.default_css = []
    root = m.get_root()
    root.header.add_child(Element(f"<title>{esc(model.title)}</title>"), name="title")
    root.header.add_child(
        Element('<meta name="viewport" content="width=device-width, initial-scale=1">'),
        name="viewport",
    )
    root.header.add_child(
        Element(
            f"<style>{(VENDOR / 'leaflet.css').read_text(encoding='utf-8')}</style>"
        ),
        name="leaflet_css",
    )
    root.header.add_child(
        Element(
            f"<script>{(VENDOR / 'leaflet.js').read_text(encoding='utf-8')}</script>"
        ),
        name="leaflet_js",
    )
    root.header.add_child(Element(f"<style>{_CSS}</style>"), name="gh_css")

    layer = folium.GeoJson(
        collection,
        name="regions",
        style_function=None,
        control=False,
        smooth_factor=0.5,
    )
    layer.add_to(m)
    controls = _Controls(m.get_name(), layer.get_name(), safe_json(payload))
    m.add_child(controls)
    rendered = root.render()
    # Appended after rendering: folium adds the map <div> to the body only at render time,
    # so anything added to root.html earlier would land *above* the map.
    head, sep, tail = rendered.rpartition("</body>")
    return head + _page_body(model) + sep + tail


__all__ = ["PALETTES", "format_number", "group_of", "palette_colors", "render_html"]
