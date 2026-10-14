import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { TARGETS, apca, apcaLevel, nearestPassing, wcagLevel, wcagRatio } from '../src/contrast.js';
import { contrastRows, pairPasses, pairs } from '../src/derive.js';
import { INPUTS, defaults } from '../src/schema.js';

const rgb = fc.tuple(fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }));
const BLACK = [0, 0, 0];
const WHITE = [255, 255, 255];

describe('WCAG 2 contrast ratio (published reference values)', () => {
  it('black on white is 21:1 and a colour on itself is 1:1', () => {
    expect(wcagRatio(BLACK, WHITE)).toBeCloseTo(21, 10);
    expect(wcagRatio([12, 99, 200], [12, 99, 200])).toBe(1);
  });

  it('#767676 is the lightest grey that passes AA on white; #777777 just misses', () => {
    expect(wcagRatio([0x76, 0x76, 0x76], WHITE)).toBeCloseTo(4.54, 2);
    expect(wcagRatio([0x77, 0x77, 0x77], WHITE)).toBeCloseTo(4.48, 2);
  });

  it('labels the bands', () => {
    expect([21, 7, 6.99, 4.5, 4.49, 3, 2.99].map(wcagLevel)).toEqual(['AAA', 'AAA', 'AA', 'AA', 'AA large', 'AA large', 'Fail']);
  });

  it('is symmetric and between 1 and 21', () => {
    fc.assert(
      fc.property(rgb, rgb, (a, b) => {
        const r = wcagRatio(a, b);
        expect(r).toBeCloseTo(wcagRatio(b, a), 12);
        expect(r).toBeGreaterThanOrEqual(1);
        expect(r).toBeLessThanOrEqual(21 + 1e-9);
      }),
    );
  });
});

describe('APCA-W3 0.0.98G-4g (published reference values)', () => {
  it('black on white is Lc 106.04 and white on black is -107.88', () => {
    expect(apca(BLACK, WHITE)).toBeCloseTo(106.04, 2);
    expect(apca(WHITE, BLACK)).toBeCloseTo(-107.88, 2);
  });

  it('is 0 for equal colours and keeps the polarity: positive for dark text, negative for light', () => {
    expect(apca([90, 90, 90], [90, 90, 90])).toBe(0);
    fc.assert(
      fc.property(rgb, rgb, (a, b) => {
        const lc = apca(a, b);
        const reverse = apca(b, a);
        expect(Math.sign(lc) * Math.sign(reverse)).toBeLessThanOrEqual(0);
        expect(Math.abs(lc)).toBeLessThan(115);
      }),
    );
  });

  it('is not symmetric: the same pair reads differently with the roles swapped', () => {
    expect(Math.abs(apca([0x77, 0x77, 0x77], WHITE))).not.toBeCloseTo(Math.abs(apca(WHITE, [0x77, 0x77, 0x77])), 0);
  });

  it('labels the bands', () => {
    expect([106, -75, 60, -45, 30, 29].map(apcaLevel)).toEqual(['Body text', 'Body text', 'Content', 'Large text', 'Spot text', 'Fail']);
  });
});

describe('the default theme', () => {
  it('passes WCAG AA on every pair in both modes', () => {
    for (const row of contrastRows(defaults())) expect(row.wcag, `${row.scheme} ${row.label}`).toBeGreaterThanOrEqual(4.5);
  });
});

describe('nearestPassing', () => {
  const brandL = INPUTS.find((i) => i.key === 'brandL');
  const pair = pairs.find((p) => p.id === 'brand-surface');
  const aa = TARGETS[0];

  it('returns null when the pair already passes', () => {
    expect(nearestPassing(defaults(), brandL, pairPasses(pair, 'light', aa))).toBeNull();
  });

  it('finds the closest lightness that fixes a failing brand link, and it really passes', () => {
    const theme = { ...defaults(), brandL: 0.75 };
    const passes = pairPasses(pair, 'light', aa);
    expect(passes(theme)).toBe(false);
    const fix = nearestPassing(theme, brandL, passes);
    expect(passes({ ...theme, brandL: fix })).toBe(true);
    // and it is the nearest: the next step toward the original value fails
    expect(passes({ ...theme, brandL: +(fix + brandL.step).toFixed(4) })).toBe(false);
    expect(fix).toBeLessThan(0.75);
  });

  it('returns null when nothing in range can pass', () => {
    expect(nearestPassing(defaults(), brandL, () => false)).toBeNull();
  });
});
