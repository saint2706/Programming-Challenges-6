import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { hexToOklch, isInGamut, linearToOklch, mixOklch, oklchToLinear, toHex, toRgb8 } from '../src/color.js';

const near = (actual, expected, tol) => expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol);

describe('OKLCH reference values (CSS Color 4 / Ottosson)', () => {
  it.each([
    ['red', '#ff0000', { l: 0.62796, c: 0.25768, h: 29.234 }],
    ['green', '#00ff00', { l: 0.86644, c: 0.29483, h: 142.495 }],
    ['blue', '#0000ff', { l: 0.45201, c: 0.31321, h: 264.052 }],
    ['white', '#ffffff', { l: 1, c: 0, h: 0 }],
    ['black', '#000000', { l: 0, c: 0, h: 0 }],
  ])('%s converts both ways', (_, hex, ref) => {
    const c = hexToOklch(hex);
    near(c.l, ref.l, 1e-4);
    near(c.c, ref.c, 1e-4);
    if (ref.c > 0) near(c.h, ref.h, 0.01);
    expect(toHex(ref)).toBe(hex);
  });

  it('puts mid grey #777777 at the lightness CSS gives it', () => {
    near(hexToOklch('#777777').l, 0.5695, 5e-4);
  });
});

describe('hexToOklch', () => {
  it('accepts #rgb, #rrggbb and no hash, rejects the rest', () => {
    expect(toHex(hexToOklch('#f00'))).toBe('#ff0000');
    expect(toHex(hexToOklch('3366cc'))).toBe('#3366cc');
    for (const bad of ['', '#12', '#12345', 'red', '#gggggg', 'rgb(1,2,3)', '#ff0000; x']) expect(hexToOklch(bad)).toBeNull();
  });
});

describe('gamut', () => {
  it('clips an out-of-gamut colour per channel, as Chrome paints it (probed: 0,214,0 and 8,76,255)', () => {
    expect(toRgb8({ l: 0.7, c: 0.4, h: 150 })).toEqual([0, 214, 0]);
    expect(toRgb8({ l: 0.55, c: 0.3, h: 264 })).toEqual([8, 76, 255]);
    expect(isInGamut({ l: 0.7, c: 0.4, h: 150 })).toBe(false);
    expect(isInGamut({ l: 0.55, c: 0.19, h: 264 })).toBe(true);
  });
});

describe('properties', () => {
  it('any 8-bit sRGB colour survives sRGB -> OKLCH -> sRGB exactly', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), fc.integer({ min: 0, max: 255 }), (r, g, b) => {
        const hex = `#${[r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('')}`;
        expect(toHex(hexToOklch(hex))).toBe(hex);
      }),
      { numRuns: 500 },
    );
  });

  it('OKLCH -> linear -> OKLCH round-trips in float', () => {
    fc.assert(
      fc.property(fc.double({ min: 0.05, max: 0.95, noNaN: true }), fc.double({ min: 0.01, max: 0.2, noNaN: true }), fc.double({ min: 0, max: 359.9, noNaN: true }), (l, c, h) => {
        const back = linearToOklch(oklchToLinear({ l, c, h }));
        near(back.l, l, 1e-6);
        near(back.c, c, 1e-6);
        near(((back.h - h + 540) % 360) - 180, 0, 1e-3);
      }),
    );
  });
});

describe('mixOklch (color-mix in oklch)', () => {
  it('mixes with black by lowering lightness and chroma, keeping the hue', () => {
    const m = mixOklch({ l: 0.5, c: 0.2, h: 100 }, { l: 0, c: 0, h: 0 }, 14);
    near(m.l, 0.43, 1e-9);
    near(m.c, 0.172, 1e-9);
    expect(m.h).toBe(100);
  });

  it('takes the shorter way round the hue circle', () => {
    expect(mixOklch({ l: 0.5, c: 0.1, h: 350 }, { l: 0.5, c: 0.1, h: 10 }, 50).h).toBeCloseTo(0, 9);
  });
});
