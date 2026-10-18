"use strict";
// The pivot page. The engine runs in this page as WebAssembly; nothing is sent anywhere.
// Every string that came from the data (field names, labels, cell text) is put into the page with
// textContent, never as HTML.

(function () {
  if (typeof document === "undefined") return;

  const SEP = String.fromCharCode(1);
  const $ = (sel, root) => (root || document).querySelector(sel);
  const config = JSON.parse($("#config").textContent);

  function h(tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (k === "value" || k === "checked" || k === "selected" || k === "disabled") node[k] = v;
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }

  const AGGS = ["sum", "count", "distinct", "mean", "median", "min", "max", "std", "var", "first", "last", "mode", "weighted_mean"];
  const SHOW_AS = [
    ["normal", "No calculation"],
    ["pct_grand", "% of grand total"],
    ["pct_row", "% of row total"],
    ["pct_col", "% of column total"],
    ["pct_parent_row", "% of parent row"],
    ["pct_parent_col", "% of parent column"],
    ["index", "Index"],
    ["diff", "Difference from previous"],
    ["pct_diff", "% difference from previous"],
    ["running", "Running total"],
    ["rank", "Rank (largest first)"],
    ["rank_asc", "Rank (smallest first)"],
  ];
  const DATE_GROUPS = ["year", "quarter", "month", "year_month", "year_quarter", "weekday", "day"];
  const FILTER_OPS = {
    str: [["in", "is one of"], ["not_in", "is not one of"], ["eq", "equals"], ["ne", "does not equal"], ["contains", "contains"], ["is_null", "is blank"], ["not_null", "is not blank"]],
    num: [["between", "between"], ["eq", "equals"], ["ne", "does not equal"], ["gt", ">"], ["ge", ">="], ["lt", "<"], ["le", "<="], ["is_null", "is blank"], ["not_null", "is not blank"]],
  };

  const state = {
    fields: [],
    rows: [],
    cols: [],
    values: [],
    filters: [],
    options: { totals_row: true, totals_col: true, subtotals: false, subtotal_position: "bottom" },
    tab: "pivot",
    layer: "",
    heat: false,
    decimals: "auto",
    result: null,
    error: null,
    drill: null,
    selected: null,
    open: null, // which chip's settings are open: "zone:index"
    millis: 0,
  };
  let engine = null;
  let tableId = null;
  const uniqueCache = new Map();

  // ------------------------------------------------------------------------------- helpers
  const kindOf = (name) => {
    const f = state.fields.find((x) => x.name === name);
    return f ? f.kind : "str";
  };
  const isNumber = (name) => ["int", "float"].includes(kindOf(name));
  const isDate = (name) => ["date", "datetime"].includes(kindOf(name));

  function measureLabel(v, index, all) {
    let base = v.label || (v.calc ? "calc " + (index + 1) : v.field ? v.agg + "(" + v.field + ")" : "count");
    let label = base;
    let n = 2;
    while (all.slice(0, index).some((m, j) => measureLabel(m, j, all) === label)) label = base + " (" + n++ + ")";
    return label;
  }
  const measureLabels = () => (state.result ? state.result.measures.map((m) => m.label) : state.values.map((v, i, a) => measureLabel(v, i, a)));

  function buildSpec() {
    const clean = (dim) => {
      const d = { field: dim.field };
      if (dim.group) d.group = dim.group;
      if (dim.sort) d.sort = dim.sort;
      if (dim.top) d.top = dim.top;
      if (dim.hide_blank) d.hide_blank = true;
      return d;
    };
    const active = (f) => !(["in", "not_in"].includes(f.op) && (!f.values || f.values.length === 0)) && !(["eq", "ne", "gt", "ge", "lt", "le", "contains"].includes(f.op) && (f.value === undefined || f.value === "")) && !(f.op === "between" && !(f.values && f.values.length === 2 && f.values[0] !== "" && f.values[1] !== ""));
    return {
      rows: state.rows.map(clean),
      cols: state.cols.map(clean),
      values: state.values.map((v) => {
        const o = { show_as: v.show_as || { kind: "normal" } };
        if (v.label) o.label = v.label;
        if (v.calc) o.calc = v.calc;
        else {
          o.agg = v.agg;
          if (v.field) o.field = v.field;
          if (v.weight) o.weight = v.weight;
        }
        return o;
      }),
      filters: state.filters.filter(active).map((f) => {
        const o = { field: f.field, op: f.op };
        if (f.value !== undefined) o.value = numeric(f.field, f.value);
        if (f.values) o.values = f.values.map((x) => numeric(f.field, x));
        return o;
      }),
      options: { ...state.options },
    };
  }
  function numeric(field, v) {
    if (isNumber(field) && v !== "" && !isNaN(Number(v))) return Number(v);
    return v;
  }

  function fmt(v, percent) {
    if (v === null || v === undefined) return "";
    if (percent) return (v * 100).toFixed(state.decimals === "auto" ? 1 : Number(state.decimals)) + "%";
    if (state.decimals !== "auto") return v.toLocaleString("en-US", { minimumFractionDigits: Number(state.decimals), maximumFractionDigits: Number(state.decimals) });
    if (Math.abs(v - Math.round(v)) < 1e-9 && Math.abs(v) < 1e15) return Math.round(v).toLocaleString("en-US");
    return Math.abs(v) >= 1 ? v.toLocaleString("en-US", { maximumFractionDigits: 2, minimumFractionDigits: 2 }) : Number(v.toPrecision(4)).toString();
  }

  // --------------------------------------------------------------------------------- running
  let pending = 0;
  function schedule() {
    clearTimeout(pending);
    pending = setTimeout(run, 20);
  }

  function run() {
    state.error = null;
    state.drill = null;
    state.selected = null;
    const t0 = performance.now();
    try {
      if (state.tab === "crosstab") {
        state.crosstab = null;
        if (state.rows.length && state.cols.length) {
          const spec = { row: crossDim(state.rows[0]), col: crossDim(state.cols[0]), filters: buildSpec().filters };
          if (state.layer) spec.layer = { field: state.layer };
          state.crosstab = engine.call("crosstab", { table: tableId, spec });
        }
      }
      state.result = engine.call("pivot", { table: tableId, spec: buildSpec() });
    } catch (e) {
      state.error = e.message;
    }
    state.millis = performance.now() - t0;
    render();
  }
  const crossDim = (d) => {
    const o = { field: d.field };
    if (d.group) o.group = d.group;
    if (d.hide_blank) o.hide_blank = true;
    return o;
  };

  // ------------------------------------------------------------------------------------ view
  function render() {
    renderFields();
    renderZones();
    renderOptions();
    renderTabs();
    renderStatus();
    const out = $("#output");
    out.textContent = "";
    if (state.error) {
      out.append(h("div", { class: "error", role: "alert" }, state.error));
      return;
    }
    if (state.tab === "crosstab") out.append(renderCrosstab());
    else {
      out.append(renderPivot(state.result));
      if (state.drill) out.append(renderDrill());
    }
  }

  function renderStatus() {
    const r = state.result;
    $("#status").textContent = r
      ? r.rows_used.toLocaleString("en-US") + " of " + r.rows_total.toLocaleString("en-US") + " rows used · " + state.millis.toFixed(1) + " ms" + (r.warnings.length ? " · " + r.warnings.join("; ") : "")
      : "";
  }

  function renderTabs() {
    for (const b of document.querySelectorAll("[data-tab]")) {
      const on = b.dataset.tab === state.tab;
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.classList.toggle("on", on);
    }
  }

  // --------------------------------------------------------------------------------- fields
  function renderFields() {
    const box = $("#fields");
    box.textContent = "";
    const q = $("#field-search").value.trim().toLowerCase();
    for (const f of state.fields) {
      if (q && !f.name.toLowerCase().includes(q)) continue;
      const row = h(
        "li",
        { class: "field", draggable: "true", "data-field": f.name },
        h("span", { class: "kind k-" + f.kind, title: f.kind }, kindGlyph(f.kind)),
        h("span", { class: "name", text: f.name }),
        h("span", { class: "add" }, ...[["rows", "R", "Rows"], ["cols", "C", "Columns"], ["values", "V", "Values"], ["filters", "F", "Filters"]].map(([zone, key, label]) =>
          h("button", { type: "button", class: "mini", "aria-label": "Add " + f.name + " to " + label, title: "Add to " + label, onclick: () => addTo(zone, f.name) }, key),
        )),
      );
      row.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("text/plain", f.name);
        e.dataTransfer.effectAllowed = "copy";
      });
      box.append(row);
    }
  }
  const kindGlyph = (k) => ({ int: "#", float: "#", str: "A", date: "D", datetime: "D" })[k] || "?";

  function addTo(zone, field) {
    if (zone === "rows" || zone === "cols") state[zone].push({ field });
    else if (zone === "values") {
      const numeric = isNumber(field) || isDate(field);
      state.values.push({ field, agg: numeric ? "sum" : "count", show_as: { kind: "normal" } });
    } else if (zone === "filters") state.filters.push(isNumber(field) ? { field, op: "between", values: ["", ""] } : isDate(field) ? { field, op: "between", values: ["", ""] } : { field, op: "in", values: [] });
    state.open = zone + ":" + (state[zone].length - 1);
    schedule();
  }

  // ---------------------------------------------------------------------------------- zones
  const ZONES = [
    ["filters", "Filters"],
    ["cols", "Columns"],
    ["rows", "Rows"],
    ["values", "Values"],
  ];

  function renderZones() {
    const box = $("#zones");
    box.textContent = "";
    for (const [zone, title] of ZONES) {
      const list = h("ul", { class: "chips" });
      state[zone].forEach((item, i) => list.append(chip(zone, item, i)));
      const section = h("section", { class: "zone", "data-zone": zone, "aria-label": title }, h("h3", { text: title }), list);
      if (!state[zone].length) list.append(h("li", { class: "hint" }, zone === "values" ? "count of rows" : "drop a field here"));
      if (zone === "values") section.append(h("button", { type: "button", class: "link", onclick: addCalc }, "+ calculated field"));
      section.addEventListener("dragover", (e) => {
        e.preventDefault();
        section.classList.add("over");
      });
      section.addEventListener("dragleave", () => section.classList.remove("over"));
      section.addEventListener("drop", (e) => {
        e.preventDefault();
        section.classList.remove("over");
        const name = e.dataTransfer.getData("text/plain");
        if (state.fields.some((f) => f.name === name)) addTo(zone, name);
      });
      box.append(section);
      const openIdx = state.open && state.open.startsWith(zone + ":") ? Number(state.open.split(":")[1]) : -1;
      if (openIdx >= 0 && state[zone][openIdx]) section.append(settings(zone, state[zone][openIdx], openIdx));
    }
  }

  function chipTitle(zone, item) {
    if (zone === "values") return item.calc ? (item.label || "calculated") + " = " + item.calc : (item.label ? item.label + " — " : "") + item.agg + (item.field ? "(" + item.field + ")" : " of rows");
    if (zone === "filters") return item.field + " " + item.op;
    return item.field + (item.group ? " · " + groupName(item.group) : "");
  }
  const groupName = (g) => (g.kind === "bin" ? "bins of " + g.width : g.kind === "quantile" ? g.n + " quantiles" : g.kind === "edges" ? "bands" : g.kind.replace("_", "-"));

  function chip(zone, item, i) {
    const key = zone + ":" + i;
    const move = (d) => {
      const arr = state[zone];
      const j = i + d;
      if (j < 0 || j >= arr.length) return;
      [arr[i], arr[j]] = [arr[j], arr[i]];
      state.open = zone + ":" + j;
      schedule();
    };
    return h(
      "li",
      { class: "chip" + (state.open === key ? " open" : "") },
      h("button", { type: "button", class: "chip-label", "aria-expanded": state.open === key ? "true" : "false", title: "Settings", onclick: () => ((state.open = state.open === key ? null : key), render()) }, chipTitle(zone, item)),
      zone !== "filters" && zone !== "values" ? null : null,
      h("button", { type: "button", class: "mini", "aria-label": "Move " + chipTitle(zone, item) + " earlier", onclick: () => move(-1) }, "◀"),
      h("button", { type: "button", class: "mini", "aria-label": "Move " + chipTitle(zone, item) + " later", onclick: () => move(1) }, "▶"),
      h("button", { type: "button", class: "mini", "aria-label": "Remove " + chipTitle(zone, item), onclick: () => (state[zone].splice(i, 1), (state.open = null), schedule()) }, "✕"),
    );
  }

  function addCalc() {
    state.values.push({ calc: "", label: "", show_as: { kind: "normal" } });
    state.open = "values:" + (state.values.length - 1);
    render();
  }

  // -------------------------------------------------------------------------------- settings
  function select(options, current, onchange, label) {
    return h("select", { "aria-label": label, onchange: (e) => onchange(e.target.value) }, options.map(([v, t]) => h("option", { value: v, selected: v === current }, t)));
  }
  const row = (label, ...kids) => h("label", { class: "set" }, h("span", { text: label }), ...kids);

  function settings(zone, item, i) {
    const body = h("div", { class: "settings" });
    if (zone === "rows" || zone === "cols") dimSettings(body, item);
    else if (zone === "values") valueSettings(body, item);
    else filterSettings(body, item);
    return body;
  }

  function dimSettings(body, d) {
    const kind = kindOf(d.field);
    const groups = [["", "No grouping"]];
    if (kind === "int" || kind === "float") groups.push(["bin", "Equal-width bins"], ["edges", "Bands at cut points"], ["quantile", "Quantiles"]);
    if (kind === "date" || kind === "datetime") DATE_GROUPS.forEach((g) => groups.push([g, g.replace("_", "-")]));
    if (kind === "datetime") groups.push(["hour", "hour"]);
    const cur = d.group ? d.group.kind : "";
    body.append(
      row("Group by", select(groups, cur, (v) => {
        if (!v) delete d.group;
        else if (v === "bin") d.group = { kind: "bin", width: 10 };
        else if (v === "edges") d.group = { kind: "edges", edges: [] };
        else if (v === "quantile") d.group = { kind: "quantile", n: 4 };
        else d.group = { kind: v };
        render();
        schedule();
      }, "Group by")),
    );
    if (cur === "bin") body.append(row("Bin width", numInput(d.group.width, (v) => ((d.group.width = v), schedule())), ), row("Start at", numInput(d.group.start === undefined ? "" : d.group.start, (v) => ((d.group.start = v === "" ? undefined : v), schedule()), true)));
    if (cur === "edges") body.append(row("Cut points", h("input", { type: "text", placeholder: "18, 30, 65", value: d.group.edges.join(", "), "aria-label": "Cut points", onchange: (e) => ((d.group.edges = e.target.value.split(",").map((x) => x.trim()).filter(Boolean).map(Number).filter((x) => !isNaN(x))), schedule()) })));
    if (cur === "quantile") body.append(row("Quantiles", numInput(d.group.n, (v) => ((d.group.n = Math.max(2, Math.round(v))), schedule()))));
    const labels = measureLabels();
    const sortValue = d.sort ? (d.sort.by === "label" ? (d.sort.desc ? "label_desc" : "label_asc") : (d.sort.desc === false ? "value_asc:" : "value_desc:") + d.sort.measure) : "label_asc";
    const sortOptions = [["label_asc", "Label A to Z"], ["label_desc", "Label Z to A"]];
    labels.forEach((m) => sortOptions.push(["value_desc:" + m, "Largest " + m + " first"], ["value_asc:" + m, "Smallest " + m + " first"]));
    body.append(
      row("Sort", select(sortOptions, sortValue, (v) => {
        if (v === "label_asc") delete d.sort;
        else if (v === "label_desc") d.sort = { by: "label", desc: true };
        else {
          const [how, ...m] = v.split(":");
          d.sort = { by: "value", measure: m.join(":"), desc: how === "value_desc" };
        }
        schedule();
      }, "Sort")),
      row("Keep top", numInput(d.top ? d.top.n : "", (v) => {
        if (v === "" || v <= 0) delete d.top;
        else d.top = { n: Math.round(v), measure: d.top ? d.top.measure : labels[0], desc: d.top ? d.top.desc : true };
        render();
        schedule();
      }, true), d.top ? select(labels.map((m) => [m, "by " + m]), d.top.measure, (v) => ((d.top.measure = v), schedule()), "Top by measure") : null),
      h("label", { class: "set check" }, h("input", { type: "checkbox", checked: !!d.hide_blank, onchange: (e) => ((d.hide_blank = e.target.checked), schedule()) }), h("span", { text: "Hide (blank)" })),
    );
  }

  function numInput(value, onchange, allowEmpty) {
    return h("input", { type: "number", step: "any", value: value === undefined ? "" : value, "aria-label": "number", onchange: (e) => {
      const t = e.target.value;
      if (t === "") return onchange(allowEmpty ? "" : value);
      const n = Number(t);
      if (!isNaN(n)) onchange(n);
    } });
  }

  function valueSettings(body, v) {
    const labels = measureLabels();
    body.append(row("Label", h("input", { type: "text", value: v.label || "", placeholder: "automatic", "aria-label": "Measure label", onchange: (e) => ((v.label = e.target.value.trim()), schedule()) })));
    if (v.calc !== undefined) {
      body.append(
        row("Formula", h("input", { type: "text", value: v.calc, placeholder: "units / orders", "aria-label": "Formula", onchange: (e) => ((v.calc = e.target.value), schedule()) })),
        h("p", { class: "hint small" }, "Uses the labels of the measures above it: " + (labels.slice(0, state.values.indexOf(v)).join(", ") || "add measures first") + ". Use [brackets] for names with spaces."),
      );
    } else {
      const numeric = isNumber(v.field) || isDate(v.field);
      const aggs = AGGS.filter((a) => numeric || a === "count" || a === "distinct");
      body.append(
        row("Summarize by", select(aggs.map((a) => [a, a.replace("_", " ")]), v.agg, (a) => ((v.agg = a), render(), schedule()), "Aggregation")),
      );
      if (v.agg === "count") body.append(h("label", { class: "set check" }, h("input", { type: "checkbox", checked: !v.field, onchange: (e) => ((v.field = e.target.checked ? undefined : (state.fields.find((f) => f.name) || {}).name), render(), schedule()) }), h("span", { text: "Count rows (not just non-blank values)" })));
      if (v.agg === "weighted_mean") body.append(row("Weights", select(state.fields.filter((f) => ["int", "float"].includes(f.kind)).map((f) => [f.name, f.name]), v.weight || "", (w) => ((v.weight = w), schedule()), "Weight field")));
      if (v.agg === "weighted_mean" && !v.weight) v.weight = (state.fields.find((f) => ["int", "float"].includes(f.kind)) || {}).name;
    }
    const sa = v.show_as || { kind: "normal" };
    const cur = sa.kind === "rank" && sa.desc === false ? "rank_asc" : sa.kind;
    body.append(row("Show values as", select(SHOW_AS, cur, (k) => {
      if (["diff", "pct_diff", "running", "rank", "rank_asc"].includes(k)) v.show_as = k === "rank_asc" ? { kind: "rank", along: sa.along || "rows", desc: false } : k === "rank" ? { kind: "rank", along: sa.along || "rows", desc: true } : { kind: k, along: sa.along || "rows" };
      else v.show_as = { kind: k };
      render();
      schedule();
    }, "Show values as")));
    if (sa.along) body.append(row("Along", select([["rows", "down the rows"], ["cols", "across the columns"]], sa.along, (a) => ((v.show_as.along = a), schedule()), "Axis")));
  }

  function filterSettings(body, f) {
    const kind = isNumber(f.field) ? "num" : isDate(f.field) ? "num" : "str";
    const ops = isDate(f.field) ? FILTER_OPS.num.filter(([o]) => o !== "ne") : FILTER_OPS[kind];
    body.append(row("Show rows where " + f.field, select(ops, f.op, (op) => {
      f.op = op;
      delete f.value;
      delete f.values;
      if (op === "in" || op === "not_in" || op === "between") f.values = op === "between" ? ["", ""] : [];
      else if (!["is_null", "not_null"].includes(op)) f.value = "";
      render();
      schedule();
    }, "Filter operator")));
    if (f.op === "in" || f.op === "not_in") body.append(valuePicker(f));
    else if (f.op === "between") body.append(row("From", h("input", { type: "text", value: f.values[0], placeholder: isDate(f.field) ? "2024-01-01" : "", "aria-label": "From", onchange: (e) => ((f.values[0] = e.target.value), schedule()) })), row("To", h("input", { type: "text", value: f.values[1], placeholder: isDate(f.field) ? "2024-12-31" : "", "aria-label": "To", onchange: (e) => ((f.values[1] = e.target.value), schedule()) })));
    else if (!["is_null", "not_null"].includes(f.op)) body.append(row("Value", h("input", { type: "text", value: f.value === undefined ? "" : f.value, "aria-label": "Value", onchange: (e) => ((f.value = e.target.value), schedule()) })));
  }

  function valuePicker(f) {
    if (!uniqueCache.has(f.field)) {
      try {
        uniqueCache.set(f.field, engine.call("values", { table: tableId, field: f.field, limit: 300 }));
      } catch (e) {
        return h("p", { class: "error" }, e.message);
      }
    }
    const info = uniqueCache.get(f.field);
    const chosen = new Set(f.values || []);
    const list = h("ul", { class: "picker" });
    const draw = (q) => {
      list.textContent = "";
      for (const v of info.values) {
        if (q && !v.toLowerCase().includes(q)) continue;
        list.append(h("li", {}, h("label", {}, h("input", { type: "checkbox", checked: chosen.has(v), onchange: (e) => {
          if (e.target.checked) chosen.add(v);
          else chosen.delete(v);
          f.values = [...chosen];
          schedule();
        } }), " ", v)));
      }
    };
    draw("");
    return h("div", { class: "picker-box" }, h("input", { type: "search", placeholder: "Find a value", "aria-label": "Find a value", oninput: (e) => draw(e.target.value.toLowerCase()) }), list, info.total > info.values.length ? h("p", { class: "hint small" }, "Showing " + info.values.length + " of " + info.total + " values.") : null);
  }

  // --------------------------------------------------------------------------------- options
  function renderOptions() {
    const box = $("#options");
    box.textContent = "";
    const check = (label, key, extra) => h("label", { class: "check" }, h("input", { type: "checkbox", checked: state.options[key], onchange: (e) => ((state.options[key] = e.target.checked), extra ? extra() : null, schedule()) }), h("span", { text: label }));
    box.append(
      check("Grand total row", "totals_row"),
      check("Grand total column", "totals_col"),
      check("Subtotals", "subtotals"),
      h("label", { class: "check" }, h("input", { type: "checkbox", checked: state.heat, onchange: (e) => ((state.heat = e.target.checked), render()) }), h("span", { text: "Heat map" })),
      h("label", { class: "check" }, h("span", { text: "Decimals" }), select([["auto", "auto"], ["0", "0"], ["1", "1"], ["2", "2"], ["3", "3"], ["4", "4"]], state.decimals, (v) => ((state.decimals = v), render()), "Decimals")),
    );
  }

  // ----------------------------------------------------------------------------- the pivot
  function rowLabelParts(hd, nFields) {
    const labels = hd.labels.map((l) => l || "");
    if (hd.kind === "total") return ["Grand Total", ...Array(Math.max(nFields - 1, 0)).fill("")];
    if (hd.kind === "subtotal") {
      const out = labels.slice(0, hd.depth);
      out[hd.depth - 1] = out[hd.depth - 1] + " Total";
      while (out.length < nFields) out.push("");
      return out;
    }
    return labels;
  }

  function colTexts(hd) {
    if (hd.kind === "total") return ["Grand Total"];
    const l = hd.labels.map((x) => x || "");
    if (hd.kind === "subtotal") {
      const out = l.slice(0, hd.depth);
      out[hd.depth - 1] += " Total";
      return out;
    }
    return l;
  }

  function renderPivot(r) {
    if (!r) return h("div", { class: "empty" }, "Loading…");
    const nr = Math.max(r.row_fields.length, 1);
    const multi = r.measures.length > 1;
    const M = multi ? r.measures.length : 1;
    const levels = Math.max(r.col_fields.length, 1);
    const hasRows = r.rows.length && r.cols.length;
    if (!hasRows) return h("div", { class: "empty" }, r.warnings.length ? r.warnings.join("; ") : "Nothing to show: add fields to Rows, Columns or Values.");

    // header matrix over expanded columns
    const ncols = r.cols.length * M;
    const total = levels + (multi ? 1 : 0);
    const grid = Array.from({ length: total }, () => Array(ncols).fill(null));
    r.cols.forEach((c, ci) => {
      const texts = colTexts(c);
      for (let m = 0; m < M; m++) {
        const col = ci * M + m;
        if (c.kind === "total") {
          grid[0][col] = { text: "Grand Total", rowspan: levels, key: "T" };
        } else {
          for (let L = 0; L < texts.length; L++) {
            const rowspan = c.kind === "subtotal" && L === texts.length - 1 ? levels - L : 1;
            if (r.col_fields.length === 0) grid[L][col] = { text: "", rowspan: 1, key: "" };
            else grid[L][col] = { text: texts[L], rowspan, key: c.kind + JSON.stringify(c.labels.slice(0, L + 1)) };
          }
        }
        if (multi) grid[total - 1][col] = { text: r.measures[m].label, rowspan: 1, key: "m" + m + c.kind + ci };
      }
    });
    const thead = h("thead");
    for (let L = 0; L < total; L++) {
      const tr = h("tr");
      if (L === 0) tr.append(h("th", { class: "corner", colspan: nr, rowspan: total, scope: "col" }, r.row_fields.join(" / ")));
      let col = 0;
      while (col < ncols) {
        const cell = grid[L][col];
        if (!cell) {
          col++;
          continue;
        }
        let span = 1;
        while (col + span < ncols && grid[L][col + span] && grid[L][col + span].key === cell.key && cell.key !== "") span++;
        tr.append(h("th", { class: "colhead" + (cell.key === "T" ? " total" : ""), colspan: span, rowspan: cell.rowspan, scope: "col" }, cell.text));
        for (let k = 0; k < span; k++) if (L + cell.rowspan <= total) for (let d = 1; d < cell.rowspan; d++) grid[L + d][col + k] = null;
        col += span;
      }
      thead.append(tr);
    }

    // heat map ranges per measure, over leaf cells
    const range = r.measures.map((_, m) => {
      const xs = [];
      r.rows.forEach((rh, ri) => {
        if (rh.kind !== "leaf") return;
        r.cols.forEach((ch, ci) => {
          const v = r.cells[ri][ci][m];
          if (ch.kind === "leaf" && v !== null) xs.push(v);
        });
      });
      return xs.length ? [Math.min(...xs), Math.max(...xs)] : [0, 0];
    });

    const tbody = h("tbody");
    let previous = Array(nr).fill("");
    r.rows.forEach((rh, ri) => {
      const parts = rowLabelParts(rh, nr);
      const shown = parts.slice();
      if (rh.kind === "leaf") {
        for (let i = 0; i < nr; i++) {
          if (parts.slice(0, i + 1).join(SEP) === previous.slice(0, i + 1).join(SEP)) shown[i] = "";
          else break;
        }
        previous = parts;
      }
      const tr = h("tr", { class: rh.kind === "leaf" ? "" : rh.kind });
      shown.slice(0, nr).forEach((t) => tr.append(h("th", { class: "rowhead", scope: "row" }, t)));
      r.cols.forEach((ch, ci) => {
        for (let m = 0; m < M; m++) {
          const v = r.cells[ri][ci][m];
          const meas = r.measures[m];
          const td = h("td", { class: "num" + (ch.kind !== "leaf" ? " " + ch.kind : ""), tabindex: v === null ? undefined : "0", "data-r": ri, "data-c": ci }, fmt(v, meas.percent));
          if (state.heat && v !== null && rh.kind === "leaf" && ch.kind === "leaf" && range[m][1] > range[m][0]) {
            const t = (v - range[m][0]) / (range[m][1] - range[m][0]);
            td.style.background = "color-mix(in srgb, var(--accent) " + Math.round(t * 55) + "%, transparent)";
          }
          if (state.selected && state.selected[0] === ri && state.selected[1] === ci) td.classList.add("selected");
          if (v !== null) {
            const open = () => drillInto(ri, ci);
            td.addEventListener("click", open);
            td.addEventListener("keydown", (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                open();
              }
            });
          }
          tr.append(td);
        }
      });
      tbody.append(tr);
    });
    const table = h("table", { class: "pivot" }, h("caption", { class: "sr" }, "Pivot table"), thead, tbody);
    return h("div", { class: "scroll" }, table, h("p", { class: "hint small" }, "Click a cell to see the rows behind it."));
  }

  function pathOf(header) {
    return header.labels.slice(0, header.depth).map((l) => l || "");
  }

  function drillInto(ri, ci) {
    const r = state.result;
    try {
      const d = engine.call("drill", { table: tableId, spec: buildSpec(), row_path: pathOf(r.rows[ri]), col_path: pathOf(r.cols[ci]), limit: 100 });
      state.drill = { ...d, title: describeCell(r, ri, ci) };
      state.selected = [ri, ci];
    } catch (e) {
      state.drill = { error: e.message, title: "" };
    }
    render();
    const panel = $("#drill");
    if (panel) panel.scrollIntoView({ block: "nearest" });
  }

  function describeCell(r, ri, ci) {
    const rows = pathOf(r.rows[ri]).map((l, i) => r.row_fields[i] + " = " + l);
    const cols = pathOf(r.cols[ci]).map((l, i) => r.col_fields[i] + " = " + l);
    return [...rows, ...cols].join(", ") || "all rows";
  }

  function renderDrill() {
    const d = state.drill;
    if (d.error) return h("div", { class: "error" }, d.error);
    const head = h("tr", {}, d.fields.map((f) => h("th", { scope: "col" }, f)));
    const body = d.rows.map((row) => h("tr", {}, row.map((c) => h("td", {}, c === null ? "" : c))));
    return h("section", { id: "drill", class: "drill" }, h("h3", {}, "Rows behind the cell: " + d.title), h("p", { class: "hint small" }, "Showing " + d.rows.length.toLocaleString("en-US") + " of " + d.total.toLocaleString("en-US") + " rows."), h("div", { class: "scroll" }, h("table", { class: "plain" }, h("thead", {}, head), h("tbody", {}, body))));
  }

  // --------------------------------------------------------------------------- the cross-tab
  function renderCrosstab() {
    const wrap = h("div", { class: "crosstab" });
    const layerOptions = [["", "No layer"]].concat(state.fields.filter((f) => f.kind === "str" || (f.distinct && f.distinct <= 30)).map((f) => [f.name, f.name]));
    wrap.append(h("div", { class: "layer-pick" }, row("Layer by (Simpson's paradox check)", select(layerOptions, state.layer, (v) => ((state.layer = v), schedule()), "Layer field"))));
    const ct = state.crosstab;
    if (!ct) {
      wrap.append(h("div", { class: "empty" }, "Put one field in Rows and one in Columns to cross-tabulate them."));
      return wrap;
    }
    wrap.append(countTable(ct));
    if (ct.stats) wrap.append(statsTable(ct.stats, ct));
    ct.warnings.forEach((w) => wrap.append(h("p", { class: "warn" }, "Warning: " + w)));
    if (ct.simpson) wrap.append(simpsonBox(ct));
    return wrap;
  }

  function countTable(ct) {
    const head = h("tr", {}, h("th", { class: "corner" }, ct.row_field + " ↓ / " + ct.col_field + " →"), ct.col_labels.map((l) => h("th", { scope: "col" }, l)), h("th", { scope: "col" }, "Total"));
    const resid = ct.stats ? residualMap(ct) : null;
    const rows = ct.row_labels.map((l, i) =>
      h("tr", {}, h("th", { scope: "row" }, l), ct.counts[i].map((c, j) => {
        const td = h("td", { class: "num" }, c.toLocaleString("en-US") + (ct.row_totals[i] ? "  (" + ((c / ct.row_totals[i]) * 100).toFixed(1) + "%)" : ""));
        const z = resid ? resid[i + ":" + j] : undefined;
        if (z !== undefined && Math.abs(z) >= 2) {
          td.classList.add(z > 0 ? "over" : "under");
          td.title = "standardized residual " + z.toFixed(2);
        }
        return td;
      }), h("td", { class: "num total" }, ct.row_totals[i].toLocaleString("en-US"))),
    );
    const foot = h("tr", { class: "total" }, h("th", { scope: "row" }, "Total"), ct.col_totals.map((t) => h("td", { class: "num" }, t.toLocaleString("en-US"))), h("td", { class: "num" }, ct.n.toLocaleString("en-US")));
    return h("div", { class: "scroll" }, h("table", { class: "pivot" }, h("caption", { class: "sr" }, "Counts"), h("thead", {}, head), h("tbody", {}, rows, foot)), h("p", { class: "hint small" }, "Cells are counts (row percentage). Shaded cells have a standardized residual of 2 or more in size: more (blue) or fewer (orange) than independence predicts."));
  }

  // The engine drops empty rows and columns before computing statistics; map residuals back.
  function residualMap(ct) {
    const keepRows = ct.row_totals.map((t, i) => (t > 0 ? i : -1)).filter((i) => i >= 0);
    const keepCols = ct.col_totals.map((t, j) => (t > 0 ? j : -1)).filter((j) => j >= 0);
    const out = {};
    ct.stats.residuals.forEach((line, a) => line.forEach((z, b) => (out[keepRows[a] + ":" + keepCols[b]] = z)));
    return out;
  }

  const num3 = (x) => (x === null || x === undefined ? "n/a" : Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : x.toFixed(3));
  const pfmt = (p) => (p < 1e-300 ? "< 1e-300" : p < 0.001 ? p.toExponential(2) : p.toFixed(4));

  function statsTable(s, ct) {
    const rows = [
      ["Chi-square (dof)", s.chi2.toFixed(2) + " (" + s.dof + ")"],
      ["p-value", pfmt(s.p_value)],
    ];
    if (s.chi2_yates !== null) rows.push(["Chi-square with Yates correction", s.chi2_yates.toFixed(2) + ", p = " + pfmt(s.p_yates)]);
    rows.push(["G (likelihood ratio)", s.g.toFixed(2) + ", p = " + pfmt(s.p_g)], ["Cramér's V", num3(s.cramers_v) + " (bias-corrected " + num3(s.cramers_v_corrected) + ")"], ["Contingency coefficient", num3(s.contingency_coefficient)]);
    if (s.phi !== null) rows.push(["Phi", num3(s.phi)]);
    if (s.odds_ratio !== null) rows.push(["Odds ratio (" + ct.row_labels[0] + " vs " + ct.row_labels[1] + ", odds of " + ct.col_labels[0] + ")", num3(s.odds_ratio) + (s.odds_ratio_ci ? " (95% CI " + num3(s.odds_ratio_ci[0]) + " to " + num3(s.odds_ratio_ci[1]) + ")" : "")]);
    if (s.fisher_p !== null) rows.push(["Fisher's exact p (two-sided)", pfmt(s.fisher_p)]);
    rows.push(["Smallest expected count", s.min_expected.toFixed(2)]);
    return h("table", { class: "plain stats" }, h("caption", {}, "Is " + ct.row_field + " associated with " + ct.col_field + "?"), h("tbody", {}, rows.map(([k, v]) => h("tr", {}, h("th", { scope: "row" }, k), h("td", {}, v)))));
  }

  function simpsonBox(ct) {
    const sp = ct.simpson;
    const pct = (x) => (x === null || x === undefined ? "" : (x * 100).toFixed(1) + "%");
    const rows = ct.layers.map((l) => h("tr", {}, h("th", { scope: "row" }, l.label), h("td", { class: "num" }, l.n.toLocaleString("en-US")), h("td", { class: "num" }, l.rates ? pct(l.rates[0]) : ""), h("td", { class: "num" }, l.rates ? pct(l.rates[1]) : ""), h("td", { class: "num" }, l.odds_ratio === null ? "" : l.odds_ratio.toFixed(2))));
    rows.push(h("tr", { class: "total" }, h("th", { scope: "row" }, "Pooled"), h("td", { class: "num" }, ct.n.toLocaleString("en-US")), h("td", { class: "num" }, pct(sp.pooled_rates[0])), h("td", { class: "num" }, pct(sp.pooled_rates[1])), h("td", { class: "num" }, sp.pooled_odds_ratio === null ? "" : sp.pooled_odds_ratio.toFixed(2))));
    return h("section", { class: "simpson" },
      h("p", { class: sp.reversal ? "verdict reversal" : "verdict" }, (sp.reversal ? "⚠ " : "✓ ") + sp.verdict),
      h("table", { class: "plain" }, h("caption", {}, "Within each " + ct.layer_field), h("thead", {}, h("tr", {}, h("th", {}, ct.layer_field), h("th", {}, "n"), h("th", {}, ct.row_labels[0] + ": share " + ct.col_labels[0]), h("th", {}, ct.row_labels[1] + ": share " + ct.col_labels[0]), h("th", {}, "odds ratio"))), h("tbody", {}, rows)),
      h("p", { class: "hint small" }, "Mantel-Haenszel common odds ratio " + num3(sp.mantel_haenszel_or) + (sp.mantel_haenszel_ci ? " (95% CI " + num3(sp.mantel_haenszel_ci[0]) + " to " + num3(sp.mantel_haenszel_ci[1]) + ")" : "") + "."),
    );
  }

  // ------------------------------------------------------------------------------- exporting
  function exportCsv() {
    const r = state.result;
    if (!r) return;
    const q = (s) => (/[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s);
    const multi = r.measures.length > 1;
    const nr = Math.max(r.row_fields.length, 1);
    const header = [];
    const levels = Math.max(r.col_fields.length, 1);
    for (let L = 0; L < levels + (multi ? 1 : 0); L++) {
      const line = Array(nr).fill("");
      if (L === 0) line[0] = r.row_fields.join(" / ");
      r.cols.forEach((c) => {
        for (let m = 0; m < (multi ? r.measures.length : 1); m++) {
          const t = colTexts(c);
          line.push(multi && L === levels ? r.measures[m].label : t[L] || "");
        }
      });
      header.push(line);
    }
    const lines = header.map((l) => l.map(q).join(","));
    r.rows.forEach((rh, ri) => {
      const parts = rowLabelParts(rh, nr);
      const cells = [];
      r.cols.forEach((_, ci) => r.cells[ri][ci].forEach((v, m) => cells.push(v === null ? "" : String(v))));
      lines.push([...parts, ...cells].map(q).join(","));
    });
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
    const a = h("a", { href: URL.createObjectURL(blob), download: "pivot.csv" });
    document.body.append(a);
    a.click();
    a.remove();
  }

  // ----------------------------------------------------------------------------------- start
  function applyPreset(p) {
    state.rows = JSON.parse(JSON.stringify(p.spec.rows || []));
    state.cols = JSON.parse(JSON.stringify(p.spec.cols || []));
    state.values = JSON.parse(JSON.stringify(p.spec.values || []));
    state.filters = JSON.parse(JSON.stringify(p.spec.filters || []));
    state.options = { totals_row: true, totals_col: true, subtotals: false, subtotal_position: "bottom", ...(p.spec.options || {}) };
    state.tab = p.tab || "pivot";
    state.layer = p.layer || "";
    state.open = null;
    state.result = null;
    $("#preset-note").textContent = p.description || "";
    schedule();
  }

  async function start() {
    try {
      const wasm = await PivotEngine.gunzipBase64($("#wasm-b64").textContent.trim());
      const data = await PivotEngine.gunzipBase64($("#data-b64").textContent.trim());
      engine = await PivotEngine.Engine.load(wasm);
      const info = engine.call("load_csv", { trim: true }, data);
      tableId = info.table;
      state.fields = info.fields;
      $("#title").textContent = config.title;
      $("#subtitle").textContent = info.rows.toLocaleString("en-US") + " rows, " + info.fields.length + " fields · the engine runs in this page";
    } catch (e) {
      $("#output").textContent = "Could not start the engine: " + e.message;
      return;
    }
    const select = $("#preset");
    select.textContent = "";
    select.append(h("option", { value: "-1" }, "Start from…"));
    config.presets.forEach((p, i) => select.append(h("option", { value: String(i) }, p.name)));
    select.addEventListener("change", () => {
      const i = Number(select.value);
      if (i >= 0) applyPreset(config.presets[i]);
    });
    $("#field-search").addEventListener("input", renderFields);
    $("#export").addEventListener("click", exportCsv);
    $("#reset").addEventListener("click", () => {
      select.value = "-1";
      applyPreset({ spec: {}, description: "" });
    });
    for (const b of document.querySelectorAll("[data-tab]")) b.addEventListener("click", () => ((state.tab = b.dataset.tab), schedule()));
    if (config.presets.length) {
      select.value = "0";
      applyPreset(config.presets[0]);
    } else {
      applyPreset({ spec: config.initial || {}, description: "" });
    }
  }

  window.pivotApp = { state, schedule, applyPreset, buildSpec };
  start();
})();
