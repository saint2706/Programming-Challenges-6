import { describe, expect, it } from 'vitest';
import { COLUMNS } from '../src/columns.js';
import { toCsv } from '../src/csv.js';
import { generateRows } from '../src/data.js';
import { createTableModel } from '../src/tableModel.js';

// Sanity ceilings, not benchmarks: ~10-20x looser than measured (see README) so a slow CI
// box passes, yet tight enough to catch an accidental O(n^2) or a per-row object explosion.
const N = 100_000;
const CEILING_MS = 3000;
const rows = generateRows(N);

function timed(fn) {
  const t0 = performance.now();
  const value = fn();
  return { ms: performance.now() - t0, value };
}

describe(`${N.toLocaleString()} rows`, () => {
  const model = createTableModel({ rows });

  it('sorts a numeric column, cold, under the ceiling', () => {
    model.toggleSort('salary');
    const { ms } = timed(() => model.order);
    expect(ms).toBeLessThan(CEILING_MS);
    expect(model.count).toBe(N);
    model.resetAll();
  });

  it('sorts a 100k-distinct text column, cold, under the ceiling', () => {
    model.toggleSort('email');
    const { ms } = timed(() => model.order);
    expect(ms).toBeLessThan(CEILING_MS);
    const first = model.getRow(0).email;
    const last = model.getRow(N - 1).email;
    expect(first.localeCompare(last, 'en', { numeric: true })).toBeLessThan(0);
    model.resetAll();
  });

  it('multi-column sort + filter + global search compose under the ceiling', () => {
    model.setColumnFilter('salary', '>=100000');
    model.setGlobalFilter('smith');
    model.toggleSort('department');
    model.toggleSort('salary', true);
    const { ms } = timed(() => model.order);
    expect(ms).toBeLessThan(CEILING_MS);
    expect(model.count).toBeGreaterThan(0);
    expect(model.count).toBeLessThan(N);
    model.resetAll();
  });

  it('serializes the whole table to CSV under the ceiling', () => {
    const { ms, value } = timed(() => toCsv(COLUMNS, model.visibleRows()));
    expect(ms).toBeLessThan(CEILING_MS);
    expect(value.endsWith('\r\n')).toBe(true);
    expect(value.length).toBeGreaterThan(N * 50);
  });

  it('never materialises per-row wrappers: the view is a flat Int32Array of indices', () => {
    expect(model.order).toBeInstanceOf(Int32Array);
    expect(model.order.byteLength).toBe(N * 4);
  });
});
