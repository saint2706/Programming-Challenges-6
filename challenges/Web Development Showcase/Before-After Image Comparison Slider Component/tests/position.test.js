import { describe, expect, it } from 'vitest';
import { clamp, isSafeImageUrl, keyTarget, parsePosition, parseRatio, positionFromPoint, valueText } from '../src/position.js';

const rect = { left: 100, top: 50, width: 400, height: 200 };

describe('parsePosition', () => {
  it.each([
    [40, 40],
    ['40', 40],
    ['40%', 40],
    ['  12.5 ', 12.5],
    [-5, 0],
    [250, 100],
    ['Infinity', 50],
  ])('%j -> %j', (input, expected) => expect(parsePosition(input)).toBe(expected));

  it.each([null, undefined, '', 'abc', NaN, {}])('%j falls back', (input) => {
    expect(parsePosition(input)).toBe(50);
    expect(parsePosition(input, 7)).toBe(7);
  });
});

describe('positionFromPoint', () => {
  it('maps x across the width, clamped at both edges', () => {
    expect(positionFromPoint(rect, 300, 0)).toBe(50);
    expect(positionFromPoint(rect, 100, 0)).toBe(0);
    expect(positionFromPoint(rect, -999, 0)).toBe(0);
    expect(positionFromPoint(rect, 9999, 0)).toBe(100);
  });

  it('measures from the right in RTL', () => {
    expect(positionFromPoint(rect, 400, 0, { rtl: true })).toBe(25);
    expect(positionFromPoint(rect, 500, 0, { rtl: true })).toBe(0);
  });

  it('maps y across the height when vertical, and RTL does not apply', () => {
    expect(positionFromPoint(rect, 0, 100, { vertical: true })).toBe(25);
    expect(positionFromPoint(rect, 0, 100, { vertical: true, rtl: true })).toBe(25);
  });

  it('returns null for a box with no size', () => {
    expect(positionFromPoint({ ...rect, width: 0 }, 100, 0)).toBeNull();
    expect(positionFromPoint({ ...rect, height: 0 }, 100, 0, { vertical: true })).toBeNull();
  });
});

describe('keyTarget', () => {
  const k = (key, current = 50, mode = {}) => keyTarget(key, current, mode);

  it('moves horizontally in the arrow direction, and flips in RTL', () => {
    expect(k('ArrowRight')).toBe(51);
    expect(k('ArrowLeft')).toBe(49);
    expect(k('ArrowRight', 50, { rtl: true })).toBe(49);
    expect(k('ArrowLeft', 50, { rtl: true })).toBe(51);
  });

  it('treats Up as more and Down as less on a horizontal slider, in RTL too', () => {
    expect(k('ArrowUp')).toBe(51);
    expect(k('ArrowDown')).toBe(49);
    expect(k('ArrowUp', 50, { rtl: true })).toBe(51);
  });

  it('moves vertically in the arrow direction (down = more), with Right/Left as more/less', () => {
    const v = { vertical: true };
    expect(k('ArrowDown', 50, v)).toBe(51);
    expect(k('ArrowUp', 50, v)).toBe(49);
    expect(k('ArrowRight', 50, v)).toBe(51);
    expect(k('ArrowLeft', 50, v)).toBe(49);
    expect(k('ArrowRight', 50, { ...v, rtl: true })).toBe(51);
  });

  it('uses the big step with Shift and for PageUp/PageDown', () => {
    expect(k('ArrowRight', 50, { shift: true })).toBe(60);
    expect(k('ArrowRight', 50, { shift: true, bigStep: 25 })).toBe(75);
    expect(k('PageUp')).toBe(60);
    expect(k('PageDown', 50, { bigStep: 20 })).toBe(30);
    expect(k('ArrowRight', 50, { step: 5 })).toBe(55);
  });

  it('jumps to the ends with Home and End', () => {
    expect(k('Home')).toBe(0);
    expect(k('End')).toBe(100);
  });

  it('clamps and does not drift on floats', () => {
    expect(k('ArrowRight', 99.5, { step: 5 })).toBe(100);
    expect(k('ArrowLeft', 0)).toBe(0);
    expect(k('ArrowRight', 0.1, { step: 0.2 })).toBe(0.3);
  });

  it('returns null for keys it does not own, so Tab and typing still work', () => {
    for (const key of ['Tab', 'Enter', ' ', 'a', 'Escape', 'ArrowRightx']) expect(k(key)).toBeNull();
  });
});

describe('valueText / parseRatio / clamp', () => {
  it('writes both halves, summing to 100', () => {
    expect(valueText(40, 'Before', 'After')).toBe('40% Before, 60% After');
    expect(valueText(33.4, 'Gray', 'Color')).toBe('33% Gray, 67% Color');
  });

  it('parses aspect ratios and rejects anything else', () => {
    expect(parseRatio('16/9')).toBe('16 / 9');
    expect(parseRatio(' 4 / 3 ')).toBe('4 / 3');
    expect(parseRatio('1.5')).toBe('1.5 / 1');
    for (const bad of ['', null, 'auto', '16/0', '0/5', '1;color:red', 'url(x)', '-3/2']) expect(parseRatio(bad)).toBeNull();
  });

  it('clamps', () => {
    expect([clamp(-1), clamp(101), clamp(33)]).toEqual([0, 100, 33]);
  });
});

describe('isSafeImageUrl', () => {
  it.each([
    'photo.webp',
    '/img/a.png',
    '../a.png',
    'https://example.com/a.jpg',
    'http://example.com/a.jpg',
    'blob:https://example.com/1234',
    'data:image/png;base64,AAAA',
    ' data:image/svg+xml,%3Csvg%3E%3C/svg%3E',
  ])('allows %s', (url) => expect(isSafeImageUrl(url)).toBe(true));

  it.each([
    'javascript:alert(1)',
    ' JaVaScRiPt:alert(1)',
    'vbscript:msgbox(1)',
    'data:text/html,<script>alert(1)</script>',
    'ftp://example.com/a.png',
    'http://',
  ])('rejects %s', (url) => expect(isSafeImageUrl(url)).toBe(false));
});
