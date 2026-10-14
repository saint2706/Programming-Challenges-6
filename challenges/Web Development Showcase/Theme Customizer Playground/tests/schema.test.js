import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { FONTS, INPUTS, cssValue, defaults, validate } from '../src/schema.js';
import { themeArb } from './helpers.js';

describe('schema', () => {
  it('has unique keys and CSS names, and every default is inside its own range', () => {
    expect(new Set(INPUTS.map((i) => i.key)).size).toBe(INPUTS.length);
    expect(new Set(INPUTS.map((i) => i.css)).size).toBe(INPUTS.length);
    for (const i of INPUTS.filter((x) => x.kind === 'number')) {
      expect(i.default, i.key).toBeGreaterThanOrEqual(i.min);
      expect(i.default, i.key).toBeLessThanOrEqual(i.max);
    }
  });

  it('accepts the defaults and any in-range theme, unchanged', () => {
    expect(validate(defaults())).toMatchObject({ ok: true, ignored: 0 });
    fc.assert(
      fc.property(themeArb, (theme) => {
        expect(validate(theme)).toMatchObject({ ok: true, theme });
      }),
    );
  });

  it('keeps the base for missing keys, and counts unknown ones as ignored', () => {
    const r = validate({ radius: 12, shoe: 1, __proto__x: 2 });
    expect(r.ok).toBe(true);
    expect(r.theme).toEqual({ ...defaults(), radius: 12 });
    expect(r.ignored).toBe(2);
  });

  it('rejects every kind of bad value, listing each', () => {
    const r = validate({ radius: 99, brandH: '264', brandC: NaN, space: Infinity, bodyFont: 'comic', headingFont: 7, typeRatio: null });
    expect(r.ok).toBe(false);
    expect(r.errors).toHaveLength(7);
  });

  it('rejects non-objects', () => {
    for (const bad of [null, undefined, 5, 'x', [], [1]]) expect(validate(bad).ok).toBe(false);
  });

  it('is not fooled by inherited keys', () => {
    const raw = Object.create({ radius: 12 });
    expect(validate(raw).theme).toEqual(defaults());
  });

  it('turns values into CSS only from numbers and the font allow-list', () => {
    const radius = INPUTS.find((i) => i.key === 'radius');
    const font = INPUTS.find((i) => i.key === 'bodyFont');
    expect(cssValue(radius, 10)).toBe('10px');
    expect(cssValue(radius, 0.30000000000000004)).toBe('0.3px');
    expect(cssValue(font, 'serif')).toBe(FONTS.serif.join(', '));
  });

  it('no font stack contains anything that needs quoting or could break out of a declaration', () => {
    for (const names of Object.values(FONTS)) for (const n of names) expect(n).toMatch(/^[a-z-]+$/i);
  });
});
