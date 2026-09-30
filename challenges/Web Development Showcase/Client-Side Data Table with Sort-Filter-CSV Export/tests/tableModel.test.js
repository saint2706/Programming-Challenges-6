import { describe, expect, it, vi } from 'vitest';
import { generateRows } from '../src/data.js';
import { mulberry32 } from '../src/prng.js';
import { createTableModel } from '../src/tableModel.js';

const rows = generateRows(500, 11);
const make = (onChange) => createTableModel({ rows, onChange });

describe('createTableModel', () => {
  it('starts unsorted and unfiltered', () => {
    const m = make();
    expect(m.total).toBe(500);
    expect(m.count).toBe(500);
    expect(m.sorting).toEqual([]);
    expect(m.getRow(0)).toBe(rows[0]);
  });

  it('cycles a column ascending -> descending -> off', () => {
    const m = make();
    m.toggleSort('salary');
    expect(m.sorting).toEqual([{ id: 'salary', desc: false }]);
    m.toggleSort('salary');
    expect(m.sorting).toEqual([{ id: 'salary', desc: true }]);
    m.toggleSort('salary');
    expect(m.sorting).toEqual([]);
    expect(m.getRow(0)).toBe(rows[0]);
  });

  it('every type starts ascending on the first click (text, number, date, boolean)', () => {
    for (const id of ['name', 'salary', 'joined', 'active', 'department']) {
      const m = make();
      m.toggleSort(id);
      expect(m.sorting).toEqual([{ id, desc: false }]);
    }
  });

  it('a plain click replaces the sort; a shift click appends a secondary key', () => {
    const m = make();
    m.toggleSort('department');
    m.toggleSort('salary'); // plain click on another column replaces
    expect(m.sorting.map((s) => s.id)).toEqual(['salary']);
    m.toggleSort('department', true);
    expect(m.sorting.map((s) => s.id)).toEqual(['salary', 'department']);
  });

  it('multi-sort orders by the primary key, then the secondary', () => {
    const m = make();
    m.toggleSort('department');
    m.toggleSort('salary', true);
    m.toggleSort('salary', true); // secondary descending
    const view = m.visibleRows();
    for (let i = 1; i < view.length; i++) {
      const a = view[i - 1];
      const b = view[i];
      expect(a.department <= b.department).toBe(true);
      if (a.department === b.department && a.salary !== null && b.salary !== null) expect(a.salary).toBeGreaterThanOrEqual(b.salary);
    }
  });

  it('sort is stable: switching direction on equal keys never reorders ties arbitrarily', () => {
    const m = make();
    m.toggleSort('active');
    const ids = m.visibleRows().map((r) => r.id);
    const trues = ids.filter((id) => rows[id - 1].active);
    expect(trues).toEqual([...trues].sort((a, b) => a - b)); // original (id) order inside the group
  });

  it('column filters compose (AND) and clear when set to empty', () => {
    const m = make();
    m.setColumnFilter('department', 'Data');
    const dataCount = m.count;
    expect(dataCount).toBeGreaterThan(0);
    expect(m.visibleRows().every((r) => r.department === 'Data')).toBe(true);
    m.setColumnFilter('salary', '>100000');
    expect(m.count).toBeLessThan(dataCount);
    expect(m.visibleRows().every((r) => r.department === 'Data' && r.salary > 100000)).toBe(true);
    m.setColumnFilter('salary', '');
    m.setColumnFilter('department', '');
    expect(m.count).toBe(500);
  });

  it('global filter narrows the view and clears with an empty string', () => {
    const m = make();
    m.setGlobalFilter('berlin');
    expect(m.count).toBeGreaterThan(0);
    expect(m.count).toBeLessThan(500);
    expect(m.visibleRows().every((r) => JSON.stringify(r).toLowerCase().includes('berlin'))).toBe(true);
    m.setGlobalFilter('');
    expect(m.count).toBe(500);
  });

  it('visibleRows reflects filter THEN sort (what the CSV export uses)', () => {
    const m = make();
    m.setColumnFilter('active', 'true');
    m.toggleSort('id');
    m.toggleSort('id'); // descending
    const view = m.visibleRows();
    expect(view.every((r) => r.active)).toBe(true);
    expect(view.map((r) => r.id)).toEqual(view.map((r) => r.id).sort((a, b) => b - a));
    expect(view).toHaveLength(m.count);
  });

  it('notifies onChange once per state change with the model, and recomputes lazily', () => {
    const onChange = vi.fn();
    const m = make(onChange);
    m.toggleSort('name');
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0]).toBe(m);
    m.setGlobalFilter('x');
    expect(onChange).toHaveBeenCalledTimes(2);
  });

  it('resetAll clears sorting, column filters and the global filter', () => {
    const m = make();
    m.toggleSort('salary');
    m.setColumnFilter('city', 'ber');
    m.setGlobalFilter('data');
    m.resetAll();
    expect(m.sorting).toEqual([]);
    expect(m.count).toBe(500);
  });

  it('agrees with a plain sort over random states (property check)', () => {
    const rnd = mulberry32(99);
    const ids = ['id', 'name', 'city', 'salary', 'joined', 'active', 'department'];
    for (let t = 0; t < 25; t++) {
      const m = make();
      const id = ids[Math.floor(rnd() * ids.length)];
      m.toggleSort(id);
      if (rnd() < 0.5) m.toggleSort(id);
      const view = m.visibleRows();
      const desc = m.sorting[0].desc;
      const present = view.map((r) => r[id]).filter((v) => v !== null && v !== '');
      const cmp = (a, b) => (typeof a === 'string' ? a.localeCompare(b, 'en', { numeric: true, sensitivity: 'base' }) : a < b ? -1 : a > b ? 1 : 0);
      for (let i = 1; i < present.length; i++) expect(desc ? cmp(present[i - 1], present[i]) >= 0 : cmp(present[i - 1], present[i]) <= 0).toBe(true);
    }
  });
});
