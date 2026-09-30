import { describe, expect, it } from 'vitest';
import { COLUMNS, DEPARTMENTS, formatCell } from '../src/columns.js';
import { AWKWARD_NOTES, generateRows } from '../src/data.js';
import { mulberry32 } from '../src/prng.js';

describe('mulberry32 / generateRows', () => {
  it('is deterministic per seed and differs across seeds', () => {
    const a = mulberry32(1);
    const b = mulberry32(1);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
    expect(mulberry32(1)()).not.toBe(mulberry32(2)());
  });

  it('yields values in [0, 1)', () => {
    const r = mulberry32(5);
    for (let i = 0; i < 1000; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('generates the same rows for the same seed', () => {
    expect(generateRows(200, 3)).toEqual(generateRows(200, 3));
    expect(generateRows(200, 3)).not.toEqual(generateRows(200, 4));
  });

  it('produces well-formed rows with unique sequential ids and unique emails', () => {
    const rows = generateRows(1000, 1);
    expect(rows.map((r) => r.id)).toEqual(Array.from({ length: 1000 }, (_, i) => i + 1));
    expect(new Set(rows.map((r) => r.email)).size).toBe(1000);
    for (const r of rows) {
      expect(DEPARTMENTS).toContain(r.department);
      expect(r.email).toMatch(/^[a-z]+\.[a-z]+\d+@example\.com$/);
      if (r.joined !== null) expect(r.joined).toMatch(/^20(1[5-9]|2[0-5])-(0[1-9]|1[0-2])-(0[1-9]|1\d|2[0-8])$/);
      if (r.salary !== null) expect(r.salary).toBeGreaterThanOrEqual(30000);
    }
  });

  it('includes missing values and every awkward CSV note so exports get exercised', () => {
    const rows = generateRows(5000, 1);
    expect(rows.some((r) => r.notes === null)).toBe(true);
    expect(rows.some((r) => r.notes === '')).toBe(true);
    expect(rows.some((r) => r.salary === null)).toBe(true);
    expect(rows.some((r) => r.joined === null)).toBe(true);
    expect(rows.some((r) => r.city === null)).toBe(true);
    for (const note of AWKWARD_NOTES) expect(rows.some((r) => r.notes === note)).toBe(true);
  });
});

describe('formatCell', () => {
  const col = (id) => COLUMNS.find((c) => c.id === id);
  it('formats display values without touching raw data', () => {
    expect(formatCell(col('salary'), 1234.5)).toBe('$1,234.50');
    expect(formatCell(col('joined'), '2021-03-04')).toBe('04 Mar 2021');
    expect(formatCell(col('active'), true)).toBe('Yes');
    expect(formatCell(col('active'), false)).toBe('No');
    expect(formatCell(col('name'), 'Ada')).toBe('Ada');
  });

  it('renders missing values as an empty cell', () => {
    for (const c of COLUMNS) {
      expect(formatCell(c, null)).toBe('');
      expect(formatCell(c, undefined)).toBe('');
    }
  });
});
