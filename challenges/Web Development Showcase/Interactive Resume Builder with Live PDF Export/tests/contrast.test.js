import { describe, expect, it } from 'vitest';
import { contrastRatio, meetsAA, parseHex } from '../src/lib/contrast.js';

describe('parseHex', () => {
  it('parses 6 and 3 digit hex, any case', () => {
    expect(parseHex('#ff0000')).toEqual([255, 0, 0]);
    expect(parseHex('#0F0')).toEqual([0, 255, 0]);
  });

  it.each(['red', '#12', '#gggggg', '', 'ff0000', null])('throws on %j', (value) => {
    expect(() => parseHex(value)).toThrow(/hex color/);
  });
});

describe('contrastRatio', () => {
  it('matches the WCAG reference values', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 5);
    expect(contrastRatio('#ffffff', '#ffffff')).toBeCloseTo(1, 5);
    expect(contrastRatio('#767676', '#ffffff')).toBeCloseTo(4.54, 2);
    expect(contrastRatio('#777777', '#ffffff')).toBeCloseTo(4.48, 2);
  });

  it('is symmetric', () => {
    expect(contrastRatio('#1d4ed8', '#ffffff')).toBeCloseTo(contrastRatio('#ffffff', '#1d4ed8'), 10);
  });
});

describe('meetsAA', () => {
  it('uses 4.5:1 for normal text', () => {
    expect(meetsAA('#767676')).toBe(true);
    expect(meetsAA('#777777')).toBe(false);
    expect(meetsAA('#1d4ed8')).toBe(true);
    expect(meetsAA('#f5c518')).toBe(false);
  });
});
