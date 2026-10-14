import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { isInGamut, toRgb8 } from '../src/color.js';
import { COLOR_TOKENS, RAMP, contrastRows, deriveColors, fallbackStyle, radiusScale, ramp, spaceScale, typeScale } from '../src/derive.js';
import { PRESETS } from '../src/presets.js';
import { defaults } from '../src/schema.js';
import { themeArb } from './helpers.js';

describe('deriveColors', () => {
  it('uses the dark-mode brand lightness only in dark mode', () => {
    const t = { ...defaults(), brandL: 0.4, brandLDark: 0.8 };
    expect(deriveColors(t, 'light').brand.l).toBe(0.4);
    expect(deriveColors(t, 'dark').brand.l).toBe(0.8);
  });

  it('picks a white label on dark brands and a near-black one on light brands', () => {
    expect(toRgb8(deriveColors({ ...defaults(), brandL: 0.5 }, 'light').onBrand)).toEqual([255, 255, 255]);
    expect(toRgb8(deriveColors({ ...defaults(), brandL: 0.7 }, 'light').onBrand)).not.toEqual([255, 255, 255]);
  });

  it('shifts the accent hue with wrap-around', () => {
    expect(deriveColors({ ...defaults(), brandH: 300, accentShift: 90 }, 'light').accent.h).toBe(30);
    expect(deriveColors({ ...defaults(), brandH: 20, accentShift: -90 }, 'light').accent.h).toBe(290);
  });

  it('makes dark surfaces darker than light ones, and text the other way round', () => {
    fc.assert(
      fc.property(themeArb, (t) => {
        const [lt, dk] = [deriveColors(t, 'light'), deriveColors(t, 'dark')];
        expect(dk.bg.l).toBeLessThan(lt.bg.l);
        expect(dk.text.l).toBeGreaterThan(lt.text.l);
      }),
    );
  });
});

describe('ramp, scales', () => {
  it('runs from light to dark in ten steps and follows the brand hue', () => {
    const r = ramp({ ...defaults(), brandH: 120 });
    expect(r.map((s) => s.step)).toEqual(RAMP.map((s) => s[0]));
    for (let i = 1; i < r.length; i++) expect(r[i].color.l).toBeLessThan(r[i - 1].color.l);
    expect(r.every((s) => s.color.h === 120)).toBe(true);
  });

  it('computes the type, space and radius scales the CSS computes', () => {
    const t = { ...defaults(), typeBase: 16, typeRatio: 1.25, space: 4, radius: 10 };
    expect(typeScale(t).map((s) => s.px)).toEqual([12.8, 16, 20, 25, 31.25, 39.06, 48.83]);
    expect(spaceScale(t).map((s) => s.px)).toEqual([4, 8, 12, 16, 24, 32]);
    expect(radiusScale(t).map((s) => s.px)).toEqual([5, 10, 15]);
  });
});

describe('fallbackStyle (what a browser without the CSS features gets)', () => {
  it('only contains hex colours and one shadow built from numbers', () => {
    fc.assert(
      fc.property(themeArb, fc.constantFrom('light', 'dark'), (t, scheme) => {
        const style = fallbackStyle(t, scheme);
        for (const [name, value] of Object.entries(style)) {
          expect(name).toMatch(/^--[a-z0-9-]+$/);
          if (name === '--shadow') expect(value).toMatch(/^0 [\d.]+px [\d.]+px rgb\(0 0 0 \/ [\d.]+\)$/);
          else expect(value).toMatch(/^#[0-9a-f]{6}$/);
        }
        expect(Object.keys(style)).toHaveLength(Object.keys(COLOR_TOKENS).length + RAMP.length + 1);
      }),
    );
  });
});

describe('presets', () => {
  it('pass WCAG AA on every pair in both modes, and sit inside sRGB', () => {
    for (const { name, theme } of PRESETS) {
      for (const row of contrastRows(theme)) expect(row.wcag, `${name} ${row.scheme} ${row.label}`).toBeGreaterThanOrEqual(4.5);
      for (const scheme of ['light', 'dark']) expect(isInGamut(deriveColors(theme, scheme).brand), `${name} ${scheme} brand`).toBe(true);
    }
  });
});
