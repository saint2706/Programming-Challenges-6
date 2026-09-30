import { describe, expect, it } from 'vitest';
import { compareBoolean, compareDate, compareNumber, compareText, comparatorFor, isMissing } from '../src/comparators.js';

const sorted = (list, cmp) => [...list].sort(cmp);

describe('compareText', () => {
  it('orders digit runs numerically, not lexically', () => {
    expect(sorted(['item10', 'item2', 'item1'], compareText)).toEqual(['item1', 'item2', 'item10']);
  });

  it('ignores case and accents (base sensitivity)', () => {
    expect(compareText('a', 'A')).toBe(0);
    expect(compareText('e', 'é')).toBe(0);
    expect(compareText('Zoë', 'zoe')).toBe(0);
  });

  it('places accented letters next to their base letter instead of after "z"', () => {
    expect(sorted(['Zoë', 'Åsa', 'Anna', 'Zed'], compareText)).toEqual(['Anna', 'Åsa', 'Zed', 'Zoë']);
  });

  it('handles CJK, emoji and empty strings without throwing', () => {
    expect(() => sorted(['李', 'Li', '☕', ''], compareText)).not.toThrow();
  });
});

describe('compareNumber', () => {
  it('sorts numerically, negatives and decimals included', () => {
    expect(sorted([10, -2, 3.5, 0, -10], compareNumber)).toEqual([-10, -2, 0, 3.5, 10]);
  });

  it('is 0 for equal values', () => {
    expect(compareNumber(2, 2)).toBe(0);
  });
});

describe('compareDate / compareBoolean', () => {
  it('sorts ISO dates chronologically as plain strings', () => {
    expect(sorted(['2021-03-04', '2019-12-31', '2021-01-15'], compareDate)).toEqual(['2019-12-31', '2021-01-15', '2021-03-04']);
  });

  it('sorts false before true', () => {
    expect(sorted([true, false, true, false], compareBoolean)).toEqual([false, false, true, true]);
  });
});

describe('comparatorFor / isMissing', () => {
  it('maps column types to comparators', () => {
    expect(comparatorFor('number')).toBe(compareNumber);
    expect(comparatorFor('currency')).toBe(compareNumber);
    expect(comparatorFor('date')).toBe(compareDate);
    expect(comparatorFor('boolean')).toBe(compareBoolean);
    expect(comparatorFor('text')).toBe(compareText);
    expect(comparatorFor('enum')).toBe(compareText);
    expect(comparatorFor('unknown')).toBe(compareText);
  });

  it('treats null, undefined and empty string as missing, but not 0/false/whitespace', () => {
    for (const v of [null, undefined, '']) expect(isMissing(v)).toBe(true);
    for (const v of [0, false, ' ', 'x', NaN]) expect(isMissing(v)).toBe(false);
  });
});
