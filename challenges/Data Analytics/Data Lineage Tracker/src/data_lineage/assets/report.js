"use strict";
// The lineage viewer. Two halves: pure graph logic (also run by the tests under Node, and compared
// with the Python implementation node by node) and DOM code that draws it. Nothing from the data is
// ever put into the page as HTML: every string goes through textContent / setAttribute.

// ---------------------------------------------------------------------------------- graph logic
function key(table, column) {
  return table + "\u0000" + column;
}
function label(table, column) {
  return table + "." + column;
}

function buildIndex(data) {
  const idx = {
    tables: new Map(), // name -> {name, rank, layer, columns:[{name,...}], isSource, warnings}
    directUp: new Map(), // dst key -> [{src:[t,c], role, steps}]
    directDown: new Map(), // src key -> [[t,c]]
    indirectUp: new Map(), // model -> [{src:[t,c], reasons}]
    indirectDown: new Map(), // src key -> Set(model)
    columns: new Map(), // key -> {table, column, info}
  };
  for (const s of data.sources) {
    idx.tables.set(s.name, { ...s, isSource: true, layer: "source", warnings: [] });
    for (const c of s.columns) idx.columns.set(key(s.name, c.name), { table: s.name, column: c.name, info: c });
  }
  for (const m of data.models) {
    idx.tables.set(m.name, { ...m, isSource: false });
    for (const c of m.columns) idx.columns.set(key(m.name, c.name), { table: m.name, column: c.name, info: c });
  }
  for (const e of data.edges) {
    const dst = key(e.dst[0], e.dst[1]);
    const src = key(e.src[0], e.src[1]);
    if (!idx.directUp.has(dst)) idx.directUp.set(dst, []);
    idx.directUp.get(dst).push({ src: e.src, role: e.role, steps: e.steps });
    if (!idx.directDown.has(src)) idx.directDown.set(src, []);
    idx.directDown.get(src).push(e.dst);
  }
  for (const e of data.indirect) {
    if (!idx.indirectUp.has(e.model)) idx.indirectUp.set(e.model, []);
    idx.indirectUp.get(e.model).push({ src: e.src, reasons: e.reasons });
    const src = key(e.src[0], e.src[1]);
    if (!idx.indirectDown.has(src)) idx.indirectDown.set(src, new Set());
    idx.indirectDown.get(src).add(e.model);
  }
  return idx;
}

function upstream(idx, table, column, indirect) {
  const seen = new Set();
  const queue = [[table, column]];
  while (queue.length) {
    const [t, c] = queue.shift();
    let parents = (idx.directUp.get(key(t, c)) || []).map((p) => p.src);
    if (indirect) parents = parents.concat((idx.indirectUp.get(t) || []).map((p) => p.src));
    for (const p of parents) {
      const k = key(p[0], p[1]);
      if (!seen.has(k)) {
        seen.add(k);
        queue.push(p);
      }
    }
  }
  return seen;
}

function downstream(idx, table, column, indirect) {
  const seen = new Set();
  const queue = [[table, column]];
  while (queue.length) {
    const [t, c] = queue.shift();
    let children = (idx.directDown.get(key(t, c)) || []).slice();
    if (indirect) {
      for (const model of idx.indirectDown.get(key(t, c)) || []) {
        for (const col of idx.tables.get(model).columns) children.push([model, col.name]);
      }
    }
    for (const ch of children) {
      const k = key(ch[0], ch[1]);
      if (!seen.has(k)) {
        seen.add(k);
        queue.push(ch);
      }
    }
  }
  return seen;
}

function split(k) {
  const i = k.indexOf("\u0000");
  return [k.slice(0, i), k.slice(i + 1)];
}

function verdictsFor(verification, table, column) {
  // source column key -> {status, kind, strategies} for one target column
  const out = new Map();
  if (!verification) return out;
  for (const [status, list] of [
    ["confirmed", verification.confirmed],
    ["unexercised", verification.unexercised],
    ["missed", verification.missed],
  ]) {
    for (const p of list) {
      if (p.target[0] === table && p.target[1] === column) {
        out.set(key(p.source[0], p.source[1]), { status, kind: p.kind, strategies: p.strategies });
      }
    }
  }
  return out;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { buildIndex, upstream, downstream, verdictsFor, key, label, split };
}

// ---------------------------------------------------------------------------------- the page
if (typeof document !== "undefined") {
  (function () {
    const data = JSON.parse(document.getElementById("lineage-data").textContent);
    const verification = data.verification || null;
    const idx = buildIndex(data);
    document.getElementById("heading").textContent = data.title || "Data lineage";
    const SVGNS = "http://www.w3.org/2000/svg";
    const BOX_W = 204;
    const GAP_X = 64;
    const HEAD_H = 30;
    const ROW_H = 22;
    const PAD = 16;
    const GAP_Y = 22;

    let selected = null; // [table, column]
    let showIndirect = false;

    function el(tag, attrs, ...kids) {
      const n = document.createElement(tag);
      for (const [k, v] of Object.entries(attrs || {})) {
        if (k === "class") n.className = v;
        else n.setAttribute(k, v);
      }
      for (const kid of kids) n.append(kid);
      return n;
    }
    function svg(tag, attrs) {
      const n = document.createElementNS(SVGNS, tag);
      for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
      return n;
    }

    // layout: one column of boxes per rank, boxes stacked in model order
    const headings = [];
    const byRank = new Map();
    for (const t of idx.tables.values()) {
      if (!byRank.has(t.rank)) byRank.set(t.rank, []);
      byRank.get(t.rank).push(t);
    }
    const pos = new Map(); // table -> {x,y,h}
    let maxY = 0;
    for (const [rank, tables] of [...byRank.entries()].sort((a, b) => a[0] - b[0])) {
      let y = PAD + 28;
      for (const t of tables) {
        const h = HEAD_H + t.columns.length * ROW_H + 8;
        pos.set(t.name, { x: PAD + rank * (BOX_W + GAP_X), y, h });
        y += h + GAP_Y;
      }
      maxY = Math.max(maxY, y);
    }
    const maxRank = Math.max(...byRank.keys());
    for (const [rank, tables] of byRank) {
      const names = [...new Set(tables.map((t) => (t.isSource ? "sources" : t.layer)))];
      const head = svg("text", { x: PAD + rank * (BOX_W + GAP_X) + 2, y: PAD + 10, class: "rank" });
      head.textContent = names.join(" / ");
      headings.push(head);
    }
    const width = PAD * 2 + (maxRank + 1) * BOX_W + maxRank * GAP_X;
    const stage = document.getElementById("graph");
    const root = svg("svg", { width, height: maxY + PAD, viewBox: `0 0 ${width} ${maxY + PAD}`, role: "img", "aria-label": "Column lineage graph" });
    const edgeLayer = svg("g", { class: "edges" });
    const nodeLayer = svg("g", { class: "nodes" });
    root.append(edgeLayer, nodeLayer, ...headings);
    stage.append(root);

    const rowEls = new Map(); // key -> rect element
    for (const t of idx.tables.values()) {
      const p = pos.get(t.name);
      const g = svg("g", { class: "table" + (t.isSource ? " source" : "") });
      g.append(svg("rect", { x: p.x, y: p.y, width: BOX_W, height: p.h, rx: 6, class: "box" }));
      const title = svg("text", { x: p.x + 10, y: p.y + 20, class: "title" });
      title.textContent = t.name;
      g.append(title);
      if (t.warnings && t.warnings.length) {
        const w = svg("text", { x: p.x + BOX_W - 10, y: p.y + 20, class: "warn", "text-anchor": "end" });
        w.textContent = "⚠";
        const tip = svg("title");
        tip.textContent = t.warnings.join("\n");
        w.append(tip);
        g.append(w);
      }
      t.columns.forEach((c, i) => {
        const y = p.y + HEAD_H + i * ROW_H;
        const row = svg("g", { class: "col", tabindex: 0, role: "button", "data-key": key(t.name, c.name), "aria-label": label(t.name, c.name) });
        const rect = svg("rect", { x: p.x + 4, y, width: BOX_W - 8, height: ROW_H - 2, rx: 3, class: "row" });
        const name = svg("text", { x: p.x + 12, y: y + 15, class: "cname" });
        name.textContent = c.name;
        const meta = svg("text", { x: p.x + BOX_W - 12, y: y + 15, class: "ctype", "text-anchor": "end" });
        meta.textContent = t.isSource ? c.type.toLowerCase() : c.opaque ? "opaque" : c.role;
        row.append(rect, name, meta);
        row.addEventListener("click", () => select([t.name, c.name]));
        row.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            select([t.name, c.name]);
          }
        });
        g.append(row);
        rowEls.set(key(t.name, c.name), row);
      });
      nodeLayer.append(g);
    }

    function anchor(table, column, side) {
      const t = idx.tables.get(table);
      const p = pos.get(table);
      const i = t.columns.findIndex((c) => c.name === column);
      return [side === "out" ? p.x + BOX_W : p.x, p.y + HEAD_H + i * ROW_H + (ROW_H - 2) / 2];
    }
    function curve(a, b) {
      const dx = Math.max(40, (b[0] - a[0]) / 2);
      return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`;
    }
    function edgePath(src, dst, cls) {
      const p = svg("path", { d: curve(anchor(src[0], src[1], "out"), anchor(dst[0], dst[1], "in")), class: "edge " + cls });
      edgeLayer.append(p);
    }

    function setBanner() {
      const b = document.getElementById("verdict-banner");
      b.textContent = "";
      if (!verification) {
        b.append(el("span", { class: "muted" }, "No empirical check in this report. Run `lineage build --verify` to add one."));
        return;
      }
      const n = (l) => l.length;
      b.append(
        el("strong", {}, `${verification.runs} pipeline runs: `),
        el("span", { class: "chip ok" }, `✓ ${n(verification.confirmed)} confirmed`),
        el("span", { class: "chip maybe" }, `? ${n(verification.unexercised)} unexercised`),
        el("span", { class: "chip bad" }, `! ${n(verification.missed)} missed`),
      );
    }

    function render() {
      edgeLayer.textContent = "";
      for (const [k, row] of rowEls) row.setAttribute("class", "col");
      for (const g of nodeLayer.querySelectorAll(".table")) g.classList.remove("dim");
      const panel = document.getElementById("panel");
      panel.textContent = "";
      if (!selected) {
        panel.append(
          el("h2", {}, "Pick a column"),
          el("p", {}, "Click any column to light up where its value comes from (blue) and everything that changes if it changes (orange)."),
          el("p", { class: "muted" }, "Solid lines are values flowing into an expression. Dashed lines (toggle above) are columns that only decide which rows exist: filters, join keys, group keys."),
        );
        for (const [k, row] of rowEls) row.classList.remove("dim");
        return;
      }
      const [t, c] = selected;
      const up = upstream(idx, t, c, showIndirect);
      const down = downstream(idx, t, c, showIndirect);
      const selKey = key(t, c);
      const touched = new Set([selKey, ...up, ...down]);
      for (const [k, row] of rowEls) {
        const cls = ["col"];
        if (k === selKey) cls.push("sel");
        else if (up.has(k)) cls.push("up");
        else if (down.has(k)) cls.push("down");
        else cls.push("dim");
        row.setAttribute("class", cls.join(" "));
      }
      // edges inside the highlighted closure
      const closureUp = new Set([selKey, ...up]);
      const closureDown = new Set([selKey, ...down]);
      for (const [dstKey, parents] of idx.directUp) {
        const dst = split(dstKey);
        for (const p of parents) {
          const sk = key(p.src[0], p.src[1]);
          if (closureUp.has(dstKey) && closureUp.has(sk)) edgePath(p.src, dst, "up");
          else if (closureDown.has(dstKey) && closureDown.has(sk)) edgePath(p.src, dst, "down");
        }
      }
      if (showIndirect) {
        for (const [model, list] of idx.indirectUp) {
          for (const ind of list) {
            const sk = key(ind.src[0], ind.src[1]);
            for (const col of idx.tables.get(model).columns) {
              const dk = key(model, col.name);
              if (closureUp.has(dk) && closureUp.has(sk) && dk !== sk) edgePath(ind.src, [model, col.name], "up dashed");
              else if (closureDown.has(dk) && closureDown.has(sk) && dk !== sk) edgePath(ind.src, [model, col.name], "down dashed");
            }
          }
        }
      }
      fillPanel(panel, t, c, up, down);
    }

    function fillPanel(panel, t, c, up, down) {
      const info = idx.columns.get(key(t, c)).info;
      const table = idx.tables.get(t);
      panel.append(el("h2", {}, label(t, c)));
      if (table.isSource) {
        panel.append(el("p", {}, `Source column, ${info.type.toLowerCase()}.`));
      } else {
        panel.append(el("p", { class: "muted" }, `${info.role}${info.opaque ? ", opaque to static analysis" : ""}`), el("pre", {}, info.expr));
        if (table.warnings.length) panel.append(el("div", { class: "note warn" }, ...table.warnings.map((w) => el("div", {}, "⚠ " + w))));
      }

      // where it comes from, hop by hop
      const hops = idx.directUp.get(key(t, c)) || [];
      if (hops.length) {
        panel.append(el("h3", {}, "Computed from"));
        const list = el("ul", { class: "paths" });
        const seen = new Set();
        const walk = (tab, col, depth, ul) => {
          for (const h of idx.directUp.get(key(tab, col)) || []) {
            const k = key(tab, col) + ">" + key(h.src[0], h.src[1]);
            if (seen.has(k) || depth > 8) continue;
            seen.add(k);
            const li = el("li", {}, el("code", {}, label(h.src[0], h.src[1])), el("span", { class: "role " + h.role }, h.role));
            if (h.steps[0] && h.role !== "copy") li.append(el("div", { class: "step" }, h.steps[0]));
            ul.append(li);
            const sub = el("ul", { class: "paths" });
            walk(h.src[0], h.src[1], depth + 1, sub);
            if (sub.childNodes.length) li.append(sub);
          }
        };
        walk(t, c, 0, list);
        panel.append(list);
      }
      if (showIndirect && !table.isSource) {
        const ind = idx.indirectUp.get(t) || [];
        if (ind.length) {
          panel.append(el("h3", {}, "Shaped by (rows, not values)"));
          panel.append(el("ul", { class: "plain" }, ...ind.map((i) => el("li", {}, el("code", {}, label(i.src[0], i.src[1])), el("span", { class: "role" }, i.reasons.join(", "))))));
        }
      }

      // the empirical verdict per source column
      const srcKeys = [...up].filter((k) => idx.tables.get(split(k)[0]).isSource);
      if (verification && !table.isSource) {
        const verdicts = verdictsFor(verification, t, c);
        const rows = [];
        for (const [k, v] of verdicts) rows.push([k, v]);
        if (rows.length) {
          panel.append(el("h3", {}, "Checked against the data"));
          const ul = el("ul", { class: "plain" });
          const icon = { confirmed: "✓", unexercised: "?", missed: "!" };
          const text = { confirmed: "confirmed", unexercised: "not exercised by the data", missed: "MISSED by the static analysis" };
          for (const [k, v] of rows.sort((a, b) => a[0].localeCompare(b[0]))) {
            const [st, sc] = split(k);
            const via = v.status === "missed" ? "" : v.kind === "indirect" ? " (rows only)" : "";
            const why = v.strategies.length ? ` moved by ${v.strategies.join(", ")}` : "";
            ul.append(el("li", { class: v.status }, el("span", { class: "chip " + (v.status === "confirmed" ? "ok" : v.status === "missed" ? "bad" : "maybe") }, `${icon[v.status]} ${text[v.status]}`), el("code", {}, label(st, sc)), el("span", { class: "muted" }, via + why)));
          }
          panel.append(ul);
        }
      } else if (srcKeys.length) {
        panel.append(el("h3", {}, "Source columns"));
        panel.append(el("ul", { class: "plain" }, ...srcKeys.sort().map((k) => el("li", {}, el("code", {}, label(...split(k)))))));
      }

      // what it changes
      const models = new Map();
      for (const k of down) {
        const [tt, cc] = split(k);
        if (!models.has(tt)) models.set(tt, []);
        models.get(tt).push(cc);
      }
      panel.append(el("h3", {}, `Downstream (${down.size} columns in ${models.size} models)`));
      if (!down.size) panel.append(el("p", { class: "muted" }, "Nothing reads this column."));
      for (const [tt, cols] of models) panel.append(el("div", { class: "down-model" }, el("code", {}, tt), el("span", { class: "muted" }, " " + cols.join(", "))));
    }

    function select(col) {
      selected = selected && selected[0] === col[0] && selected[1] === col[1] ? null : col;
      render();
      if (selected) rowEls.get(key(selected[0], selected[1])).scrollIntoView({ block: "nearest", inline: "nearest" });
    }

    document.getElementById("indirect").addEventListener("change", (ev) => {
      showIndirect = ev.target.checked;
      render();
    });
    const search = document.getElementById("search");
    search.addEventListener("input", () => {
      const q = search.value.trim().toLowerCase();
      for (const [k, row] of rowEls) {
        const hit = q && k.replace("\u0000", ".").toLowerCase().includes(q);
        row.classList.toggle("hit", Boolean(hit));
      }
    });
    search.addEventListener("keydown", (ev) => {
      if (ev.key !== "Enter") return;
      const q = search.value.trim().toLowerCase();
      for (const [k] of rowEls) {
        if (q && k.replace("\u0000", ".").toLowerCase().includes(q)) {
          const [t, c] = split(k);
          selected = null;
          select([t, c]);
          rowEls.get(k).scrollIntoView({ block: "center", inline: "center" });
          break;
        }
      }
    });
    window.lineage = { select: (t, c) => select([t, c]), state: () => ({ selected, showIndirect }) };
    setBanner();
    render();
  })();
}
