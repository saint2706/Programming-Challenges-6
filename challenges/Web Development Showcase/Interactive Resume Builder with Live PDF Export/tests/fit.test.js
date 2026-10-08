import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { fitToPages } from '../src/lib/fit.js';

describe('fitToPages', () => {
  it('returns the largest scale that still fits', () => {
    // pretend the resume needs ceil(scale * 2.2) pages
    const pageCountAt = (scale) => Math.ceil(scale * 2.2 - 1e-9);
    const result = fitToPages(2, pageCountAt);
    expect(result.fits).toBe(true);
    expect(result.pageCount).toBeLessThanOrEqual(2);
    expect(result.scale).toBeCloseTo(0.9, 5);
    expect(pageCountAt(result.scale + 0.01)).toBeGreaterThan(2);
  });

  it('uses the maximum scale when everything fits', () => {
    const result = fitToPages(3, () => 1);
    expect(result).toEqual({ scale: 1.15, pageCount: 1, fits: true });
  });

  it('reports the minimum scale and its page count when nothing fits', () => {
    const result = fitToPages(1, () => 3);
    expect(result).toEqual({ scale: 0.82, pageCount: 3, fits: false });
  });

  it('takes few measurements', () => {
    let calls = 0;
    fitToPages(2, (scale) => {
      calls++;
      return Math.ceil(scale * 2.2 - 1e-9);
    });
    expect(calls).toBeLessThanOrEqual(9);
  });

  it('honours a custom range and step', () => {
    const result = fitToPages(1, (s) => (s <= 0.5 ? 1 : 2), { min: 0.4, max: 0.7, step: 0.05 });
    expect(result.scale).toBeCloseTo(0.5, 5);
  });

  it('matches a brute-force scan for any monotone page-count function', () => {
    fc.assert(
      fc.property(fc.double({ min: 0.5, max: 3, noNaN: true }), fc.integer({ min: 1, max: 4 }), (rate, target) => {
        const pageCountAt = (scale) => Math.max(1, Math.ceil(scale * rate - 1e-9));
        const result = fitToPages(target, pageCountAt);
        const grid = Array.from({ length: 34 }, (_, i) => Number((0.82 + i * 0.01).toFixed(2)));
        const best = grid.filter((s) => pageCountAt(s) <= target).at(-1);
        if (best === undefined) {
          expect(result.fits).toBe(false);
          expect(result.scale).toBe(0.82);
        } else {
          expect(result.fits).toBe(true);
          expect(result.scale).toBeCloseTo(best, 5);
        }
      }),
      { numRuns: 200 },
    );
  });

  it('rejects a bad target', () => {
    expect(() => fitToPages(0, () => 1)).toThrow(RangeError);
    expect(() => fitToPages(1.5, () => 1)).toThrow(RangeError);
  });
});
