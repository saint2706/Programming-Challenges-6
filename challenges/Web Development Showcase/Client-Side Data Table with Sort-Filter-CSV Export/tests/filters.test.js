import { afterEach, describe, expect, it, vi } from 'vitest';
import { debounce, matchesGlobal, parseRangeFilter, searchHaystack } from '../src/filters.js';

const num = (expr) => parseRangeFilter('number', expr);
const date = (expr) => parseRangeFilter('date', expr);

describe('numeric range filter', () => {
  it('empty expression matches everything, including missing values', () => {
    const f = num('  ');
    expect(f.valid).toBe(true);
    expect(f.test(null)).toBe(true);
    expect(f.test(5)).toBe(true);
  });

  it.each([
    ['>50', [49, 50, 51], [false, false, true]],
    ['>=50', [49, 50, 51], [false, true, true]],
    ['<50', [49, 50, 51], [true, false, false]],
    ['<=50', [49, 50, 51], [true, true, false]],
    ['=50', [49, 50, 51], [false, true, false]],
    ['10..20', [9, 10, 15, 20, 21], [false, true, true, true, false]],
    ['> 50', [50, 51], [false, true]],
    ['>-5', [-6, -5, -4], [false, false, true]],
    ['>=1.5', [1.4, 1.5], [false, true]],
  ])('%s', (expr, values, expected) => {
    const f = num(expr);
    expect(f.valid).toBe(true);
    expect(values.map((v) => f.test(v))).toEqual(expected);
  });

  it('never matches missing values once a real filter is active', () => {
    for (const expr of ['>0', '<=999', '=0', '0..1', '1']) {
      const f = num(expr);
      expect(f.test(null)).toBe(false);
      expect(f.test(undefined)).toBe(false);
      expect(f.test('')).toBe(false);
    }
  });

  it('a bare number is a prefix match on the raw value', () => {
    const f = num('12');
    expect([12, 120, 1234, 21, 112].map((v) => f.test(v))).toEqual([true, true, true, false, false]);
  });

  it('flags nonsense as invalid but matching everything, so a typo does not blank the table', () => {
    for (const expr of ['abc', '>abc', '5..x', '..5', '>', '1 2']) {
      const f = num(expr);
      expect(f.valid, expr).toBe(false);
      expect(f.test(3)).toBe(true);
    }
  });
});

describe('date range filter', () => {
  it('partial ISO dates cover the whole period', () => {
    expect(date('>=2021').test('2021-01-01')).toBe(true);
    expect(date('>=2021').test('2020-12-31')).toBe(false);
    expect(date('>2021').test('2021-12-31')).toBe(false); // after the WHOLE of 2021
    expect(date('>2021').test('2022-01-01')).toBe(true);
    expect(date('<2021').test('2020-12-31')).toBe(true);
    expect(date('<2021').test('2021-01-01')).toBe(false);
    expect(date('<=2021-03').test('2021-03-31')).toBe(true);
    expect(date('<=2021-03').test('2021-04-01')).toBe(false);
  });

  it('ranges and equality', () => {
    const r = date('2020..2021');
    expect(['2019-12-31', '2020-01-01', '2021-12-31', '2022-01-01'].map((d) => r.test(d))).toEqual([false, true, true, false]);
    expect(date('=2021-03').test('2021-03-15')).toBe(true);
    expect(date('=2021-03').test('2021-04-01')).toBe(false);
  });

  it('a bare partial date is a prefix match', () => {
    const f = date('2021-0');
    expect(f.valid).toBe(true);
    expect(f.test('2021-09-30')).toBe(true);
    expect(f.test('2021-10-01')).toBe(false);
  });

  it('rejects non-dates and never matches missing values', () => {
    expect(date('yesterday').valid).toBe(false);
    expect(date('>2021-13-45x').valid).toBe(false);
    expect(date('>=2021').test(null)).toBe(false);
  });
});

describe('global search', () => {
  const row = { id: 7, name: 'Zoë Müller', city: null, active: true, notes: 'Owns billing, invoicing' };

  it('AND-matches whitespace separated tokens across any column, case-insensitively', () => {
    expect(matchesGlobal(row, 'zoë billing')).toBe(true);
    expect(matchesGlobal(row, '  MÜLLER   INVOICING ')).toBe(true);
    expect(matchesGlobal(row, 'zoë nonsense')).toBe(false);
  });

  it('an empty query matches; null cells contribute nothing', () => {
    expect(matchesGlobal(row, '')).toBe(true);
    expect(matchesGlobal(row, '   ')).toBe(true);
    expect(searchHaystack(row)).not.toContain('null');
  });

  it('booleans are searchable as yes/no/true/false', () => {
    expect(matchesGlobal(row, 'yes')).toBe(true);
    expect(matchesGlobal({ active: false }, 'no')).toBe(true);
  });

  it('caches the haystack per row object', () => {
    expect(searchHaystack(row)).toBe(searchHaystack(row));
  });
});

describe('debounce', () => {
  afterEach(() => vi.useRealTimers());

  it('collapses a burst into one trailing call with the last arguments', () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const d = debounce(fn, 100);
    d('a');
    vi.advanceTimersByTime(50);
    d('b');
    vi.advanceTimersByTime(99);
    expect(fn).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(fn).toHaveBeenCalledOnce();
    expect(fn).toHaveBeenCalledWith('b');
  });

  it('cancel() drops the pending call', () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const d = debounce(fn, 10);
    d();
    d.cancel();
    vi.runAllTimers();
    expect(fn).not.toHaveBeenCalled();
  });
});
