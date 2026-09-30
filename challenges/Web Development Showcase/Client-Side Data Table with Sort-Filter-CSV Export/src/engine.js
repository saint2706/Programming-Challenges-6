import { comparatorFor, isMissing } from './comparators.js';
import { isNumeric } from './columns.js';
import { matchesGlobal, parseRangeFilter, searchHaystack } from './filters.js';

/**
 * Filter + sort engine over an index permutation of the raw row array.
 *
 * Why not table-core's row models? Measured on 100k rows, `getCoreRowModel`
 * alone took ~650 ms and a single sort ~1.5 s, with ~2 GB of Row/Cell closures
 * (see README). So table-core owns the *state* (sorting, filters, toggle
 * semantics, multi-sort) and this engine computes the result: sort keys are
 * flat Float64Arrays built lazily per column, filters are index loops, and the
 * output is an Int32Array of row indices. Rows are never copied or wrapped.
 */
export function createEngine(rows, columns) {
  const byId = new Map(columns.map((c) => [c.id, c]));
  const sortKeys = new Map();
  const lowered = new Map();
  let haystacks = null;
  let filteredCache = { signature: null, order: null };

  /** Float64Array of sortable keys; NaN marks a missing value (always sorts last). */
  function sortKey(col) {
    let key = sortKeys.get(col.id);
    if (key) return key;
    const n = rows.length;
    key = new Float64Array(n);
    if (isNumeric(col.type)) {
      for (let i = 0; i < n; i++) {
        const v = rows[i][col.id];
        key[i] = isMissing(v) ? NaN : v;
      }
    } else {
      // Rank the DISTINCT values once with the real comparator, then sort on ints.
      // 100k emails cost one collator sort; 288 distinct names cost nothing.
      const compare = comparatorFor(col.type);
      const distinct = new Set();
      for (let i = 0; i < n; i++) {
        const v = rows[i][col.id];
        if (!isMissing(v)) distinct.add(v);
      }
      const ranks = new Map();
      [...distinct].sort(compare).forEach((v, r) => ranks.set(v, r));
      for (let i = 0; i < n; i++) {
        const v = rows[i][col.id];
        key[i] = isMissing(v) ? NaN : ranks.get(v);
      }
    }
    sortKeys.set(col.id, key);
    return key;
  }

  function lowerColumn(col) {
    let arr = lowered.get(col.id);
    if (!arr) {
      arr = rows.map((r) => (isMissing(r[col.id]) ? '' : String(r[col.id]).toLowerCase()));
      lowered.set(col.id, arr);
    }
    return arr;
  }

  function compileColumnFilter(id, value) {
    const col = byId.get(id);
    if (!col || value === undefined || value === '') return null;
    if (isNumeric(col.type) || col.type === 'date') {
      const parsed = parseRangeFilter(col.type === 'date' ? 'date' : 'number', String(value));
      return parsed.valid ? (i) => parsed.test(rows[i][id]) : null;
    }
    if (col.type === 'enum') return (i) => rows[i][id] === value;
    if (col.type === 'boolean') return (i) => String(rows[i][id]) === value;
    const needle = String(value).toLowerCase();
    const arr = lowerColumn(col);
    return (i) => arr[i].includes(needle);
  }

  function compileGlobal(query) {
    const tokens = String(query ?? '').toLowerCase().split(/\s+/).filter(Boolean);
    if (tokens.length === 0) return null;
    haystacks ??= rows.map(searchHaystack);
    return (i) => {
      const h = haystacks[i];
      for (let t = 0; t < tokens.length; t++) if (!h.includes(tokens[t])) return false;
      return true;
    };
  }

  function filter({ columnFilters = [], globalFilter }) {
    const signature = JSON.stringify([columnFilters, globalFilter ?? '']);
    if (filteredCache.signature === signature) return filteredCache.order;
    const preds = columnFilters.map((f) => compileColumnFilter(f.id, f.value)).filter(Boolean);
    const global = compileGlobal(globalFilter);
    if (global) preds.push(global);
    const n = rows.length;
    let order;
    if (preds.length === 0) {
      order = new Int32Array(n);
      for (let i = 0; i < n; i++) order[i] = i;
    } else {
      const buf = new Int32Array(n);
      let k = 0;
      outer: for (let i = 0; i < n; i++) {
        for (let p = 0; p < preds.length; p++) if (!preds[p](i)) continue outer;
        buf[k++] = i;
      }
      order = buf.slice(0, k);
    }
    filteredCache = { signature, order };
    return order;
  }

  /** Stable multi-column sort; missing values last in both directions. */
  function sort(order, sorting) {
    const specs = sorting
      .map((s) => ({ key: byId.has(s.id) ? sortKey(byId.get(s.id)) : null, desc: !!s.desc }))
      .filter((s) => s.key);
    if (specs.length === 0) return order;
    const out = order.slice();
    out.sort((a, b) => {
      for (let s = 0; s < specs.length; s++) {
        const { key, desc } = specs[s];
        const va = key[a];
        const vb = key[b];
        if (va !== va) {
          if (vb === vb) return 1;
        } else if (vb !== vb) {
          return -1;
        } else if (va !== vb) {
          return desc ? vb - va : va - vb;
        }
      }
      return a - b; // stability: original order breaks all ties
    });
    return out;
  }

  return {
    /** @returns {Int32Array} indices into `rows` for the current view. */
    query(state) {
      return sort(filter(state), state.sorting ?? []);
    },
    /** Reference implementation used by tests to cross-check the fast path. */
    slowMatch: (row, query) => matchesGlobal(row, query),
  };
}
