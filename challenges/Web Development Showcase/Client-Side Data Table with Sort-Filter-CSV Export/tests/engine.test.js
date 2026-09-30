import { describe, expect, it } from 'vitest';
import { COLUMNS, isNumeric } from '../src/columns.js';
import { comparatorFor, isMissing } from '../src/comparators.js';
import { generateRows } from '../src/data.js';
import { createEngine } from '../src/engine.js';
import { matchesGlobal, parseRangeFilter } from '../src/filters.js';

const byId = new Map(COLUMNS.map((c) => [c.id, c]));

/** Slow, obviously-correct reference: object rows, Array#filter, stable Array#sort. */
function reference(rows, { columnFilters = [], globalFilter = '', sorting = [] }) {
  let idx = rows.map((_, i) => i);
  for (const { id, value } of columnFilters) {
    const col = byId.get(id);
    if (value === '' || value === undefined) continue;
    idx = idx.filter((i) => {
      const v = rows[i][id];
      if (isNumeric(col.type)) return parseRangeFilter('number', value).test(v);
      if (col.type === 'date') return parseRangeFilter('date', value).test(v);
      if (col.type === 'enum') return v === value;
      if (col.type === 'boolean') return String(v) === value;
      return (isMissing(v) ? '' : String(v)).toLowerCase().includes(String(value).toLowerCase());
    });
  }
  if (globalFilter) idx = idx.filter((i) => matchesGlobal(rows[i], globalFilter));
  idx.sort((a, b) => {
    for (const { id, desc } of sorting) {
      const va = rows[a][id];
      const vb = rows[b][id];
      const ma = isMissing(va);
      const mb = isMissing(vb);
      if (ma || mb) {
        if (ma && mb) continue;
        return ma ? 1 : -1; // missing last, whichever the direction
      }
      const c = comparatorFor(byId.get(id).type)(va, vb);
      if (c !== 0) return desc ? -c : c;
    }
    return 0; // Array#sort is stable, so ties keep original order
  });
  return idx;
}

const rows = generateRows(3000, 7);
const engine = createEngine(rows, COLUMNS);
const run = (state) => Array.from(engine.query(state));

describe('engine.query', () => {
  it('returns every row in original order with no state', () => {
    const out = engine.query({});
    expect(out).toBeInstanceOf(Int32Array);
    expect(out.length).toBe(rows.length);
    expect(out[0]).toBe(0);
    expect(out[rows.length - 1]).toBe(rows.length - 1);
  });

  it('matches the reference for single-column sorts of every column, both directions', () => {
    for (const col of COLUMNS) {
      for (const desc of [false, true]) {
        const state = { sorting: [{ id: col.id, desc }] };
        expect(run(state), `${col.id} ${desc ? 'desc' : 'asc'}`).toEqual(reference(rows, state));
      }
    }
  });

  it('keeps missing values last in BOTH directions', () => {
    const missingSalary = rows.filter((r) => r.salary === null).length;
    expect(missingSalary).toBeGreaterThan(0); // the seed must actually contain nulls
    for (const desc of [false, true]) {
      const out = run({ sorting: [{ id: 'salary', desc }] });
      const tail = out.slice(out.length - missingSalary).map((i) => rows[i].salary);
      const head = out.slice(0, out.length - missingSalary).map((i) => rows[i].salary);
      expect(tail.every((v) => v === null)).toBe(true);
      expect(head.every((v) => v !== null)).toBe(true);
    }
  });

  it('is stable: ties keep original row order (ascending and descending)', () => {
    for (const desc of [false, true]) {
      const out = run({ sorting: [{ id: 'department', desc }] });
      for (let i = 1; i < out.length; i++) {
        if (rows[out[i]].department === rows[out[i - 1]].department) expect(out[i]).toBeGreaterThan(out[i - 1]);
      }
    }
  });

  it('multi-column sort uses later keys only to break earlier ties', () => {
    const state = { sorting: [{ id: 'department', desc: false }, { id: 'salary', desc: true }, { id: 'name', desc: false }] };
    expect(run(state)).toEqual(reference(rows, state));
  });

  it('matches the reference across combined filters and sorts (randomised)', () => {
    const cases = [
      { columnFilters: [{ id: 'department', value: 'Data' }], sorting: [{ id: 'joined', desc: true }] },
      { columnFilters: [{ id: 'salary', value: '50000..90000' }, { id: 'active', value: 'true' }], sorting: [{ id: 'name', desc: false }] },
      { columnFilters: [{ id: 'city', value: 'ber' }, { id: 'joined', value: '>=2020' }], globalFilter: 'engineering' },
      { columnFilters: [{ id: 'id', value: '<=100' }], sorting: [{ id: 'salary', desc: true }] },
      { globalFilter: 'billing invoicing', sorting: [{ id: 'email', desc: false }] },
      { columnFilters: [{ id: 'notes', value: '=' }] },
      { columnFilters: [{ id: 'name', value: 'zoë' }, { id: 'department', value: 'Sales' }], globalFilter: 'yes' },
    ];
    for (const state of cases) expect(run(state), JSON.stringify(state)).toEqual(reference(rows, state));
  });

  it('an invalid range expression filters nothing (typo tolerant)', () => {
    expect(run({ columnFilters: [{ id: 'salary', value: 'abc' }] })).toHaveLength(rows.length);
  });

  it('ignores unknown columns and empty filter values', () => {
    expect(run({ columnFilters: [{ id: 'nope', value: 'x' }, { id: 'city', value: '' }], sorting: [{ id: 'nope' }] })).toHaveLength(rows.length);
  });

  it('a filter that matches nothing yields an empty view', () => {
    expect(engine.query({ globalFilter: 'zzzz-no-such-text' })).toHaveLength(0);
  });

  it('text search ignores case but not accents', () => {
    const accented = run({ globalFilter: 'zoë' });
    expect(run({ globalFilter: 'ZOË' })).toEqual(accented);
    expect(accented.length).toBeGreaterThan(0);
    // Emails are ASCII slugs ("zoe.patel1@..."), so the unaccented query is a superset...
    const plain = new Set(run({ globalFilter: 'zoe' }));
    expect(accented.every((i) => plain.has(i))).toBe(true);
    // ...and every extra hit matched only through the email, never the accented name.
    expect(plain.size).toBeGreaterThanOrEqual(accented.length);
    for (const i of plain) if (!accented.includes(i)) expect(rows[i].name.toLowerCase()).not.toContain('zoë');
  });

  it('does not mutate the cached filtered order when sorting', () => {
    const before = Array.from(engine.query({}));
    engine.query({ sorting: [{ id: 'salary', desc: true }] });
    expect(Array.from(engine.query({}))).toEqual(before);
  });

  it('sorts text by collation (natural, accent-aware), via distinct-value ranks', () => {
    const small = [{ n: 'item10' }, { n: 'Item2' }, { n: null }, { n: 'élan' }, { n: 'Eagle' }];
    const cols = [{ id: 'n', header: 'n', type: 'text', width: 1 }];
    const e = createEngine(small, cols);
    const order = Array.from(e.query({ sorting: [{ id: 'n', desc: false }] })).map((i) => small[i].n);
    expect(order).toEqual(['Eagle', 'élan', 'Item2', 'item10', null]);
    const desc = Array.from(e.query({ sorting: [{ id: 'n', desc: true }] })).map((i) => small[i].n);
    expect(desc).toEqual(['item10', 'Item2', 'élan', 'Eagle', null]);
  });
});
