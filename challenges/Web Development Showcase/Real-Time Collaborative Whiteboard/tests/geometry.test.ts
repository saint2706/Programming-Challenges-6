import fc from 'fast-check';
import { describe, expect, test } from 'vitest';
import type { ShapeItem, StrokeItem, TextItem } from '../shared/types.ts';
import { distanceTo, itemBounds, layoutText, svgPath, touches, wrapText } from '../src/geometry.ts';

/** Every glyph is 10 units wide, so widths are exact and the tests do not depend on a font. */
const measure = (text: string, fs: number) => text.length * fs * 0.5;

const stroke = (pts: number[], size = 4): StrokeItem => ({ t: 'stroke', id: 's', author: 'x', color: '#000000', size, pts });
const shape = (k: ShapeItem['k'], x1: number, y1: number, x2: number, y2: number, size = 4): ShapeItem => ({ t: 'shape', id: 'h', author: 'x', color: '#000000', size, k, x1, y1, x2, y2 });
const text = (t: string, sticky = false, size = 20): TextItem => ({ t: 'text', id: 't', author: 'x', color: '#000000', size, x: 100, y: 50, text: t, sticky });

describe('wrapText', () => {
    test('breaks at spaces and keeps explicit newlines', () => {
        expect(wrapText('aa bb cc dd', 60, 20, measure)).toEqual(['aa bb', 'cc dd']); // 10 units per char, 6 chars max
        expect(wrapText('one\ntwo', 1000, 20, measure)).toEqual(['one', 'two']);
        expect(wrapText('', 100, 20, measure)).toEqual(['']);
    });

    test('splits a word that is wider than the box', () => {
        const lines = wrapText('abcdefghijklmnop', 50, 20, measure);
        expect(lines.join('')).toBe('abcdefghijklmnop');
        expect(lines.every((l) => measure(l, 20) <= 50)).toBe(true);
    });

    test('no line is ever wider than the box, and no text is lost', () => {
        fc.assert(
            fc.property(fc.string({ maxLength: 120 }), fc.integer({ min: 20, max: 400 }), (s, max) => {
                const lines = wrapText(s, max, 20, measure);
                for (const l of lines) expect(measure(l.trimEnd(), 20)).toBeLessThanOrEqual(Math.max(max, 10)); // a single glyph may overhang
                const strip = (x: string) => x.replace(/\s/g, '');
                expect(strip(lines.join(''))).toBe(strip(s));
            }),
        );
    });
});

describe('text cost (a sticky note must never be able to freeze the page)', () => {
    test('a 2,000-letter word wraps with a few thousand measurements, not hundreds of thousands', () => {
        let calls = 0;
        const counting = (t: string, fs: number) => (calls++, measure(t, fs));
        const lines = wrapText('A'.repeat(2000), 200, 20, counting);
        expect(lines.join('')).toBe('A'.repeat(2000));
        expect(lines.every((l) => measure(l, 20) <= 200)).toBe(true);
        expect(calls).toBeLessThan(5000);
    });

    test('a layout is computed once per item and measure function', () => {
        let calls = 0;
        const counting = (t: string, fs: number) => (calls++, measure(t, fs));
        const note = text('word '.repeat(100), true);
        layoutText(note, counting);
        const after = calls;
        expect(after).toBeGreaterThan(0);
        layoutText(note, counting);
        expect(calls).toBe(after);
        layoutText({ ...note }, counting); // a new object (an edit) is laid out afresh
        expect(calls).toBeGreaterThan(after);
    });
});

describe('layoutText', () => {
    test('plain text is sized by its widest line', () => {
        const l = layoutText(text('hello\nhi'), measure);
        expect(l.w).toBe(50);
        expect(l.lines).toEqual(['hello', 'hi']);
        expect(l.h).toBeCloseTo(2 * 20 * 1.3);
    });

    test('a sticky note has a fixed width, wraps, grows with its text and is at least nearly square', () => {
        const short = layoutText(text('hi', true), measure);
        const long = layoutText(text('word '.repeat(60), true), measure);
        expect(short.w).toBe(180);
        expect(long.w).toBe(short.w);
        expect(long.h).toBeGreaterThan(short.h);
        expect(short.h).toBeGreaterThanOrEqual(short.w * 0.8);
    });
});

describe('bounds', () => {
    test('a stroke is bounded by its points and its width', () => {
        const b = itemBounds(stroke([10, 20, 30, 5], 10), measure);
        expect(b).toEqual({ minX: 4, minY: -1, maxX: 36, maxY: 26 });
    });

    test('every point of any stroke lies inside its bounds', () => {
        fc.assert(
            fc.property(fc.array(fc.double({ min: -1e4, max: 1e4, noNaN: true }), { minLength: 2, maxLength: 60 }).filter((a) => a.length % 2 === 0), (pts) => {
                const b = itemBounds(stroke(pts), measure);
                for (let i = 0; i < pts.length; i += 2) {
                    expect(pts[i]!).toBeGreaterThanOrEqual(b.minX);
                    expect(pts[i]!).toBeLessThanOrEqual(b.maxX);
                    expect(pts[i + 1]!).toBeGreaterThanOrEqual(b.minY);
                    expect(pts[i + 1]!).toBeLessThanOrEqual(b.maxY);
                }
            }),
        );
    });

    test('an arrow leaves room for its head', () => {
        const line = itemBounds(shape('line', 0, 0, 100, 0), measure);
        const arrow = itemBounds(shape('arrow', 0, 0, 100, 0), measure);
        expect(arrow.maxX - arrow.minX).toBeGreaterThan(line.maxX - line.minX);
    });
});

describe('distanceTo (what the eraser and the select tool hit)', () => {
    test('a stroke is hit near its line, not near its far end of the page', () => {
        const s = stroke([0, 0, 100, 0], 6);
        expect(distanceTo(s, 50, 2, measure)).toBeLessThan(0); // inside the line's width
        expect(distanceTo(s, 50, 20, measure)).toBeCloseTo(17);
        expect(distanceTo(s, 150, 0, measure)).toBeCloseTo(47);
    });

    test('a single-point stroke is a dot', () => {
        expect(distanceTo(stroke([10, 10], 8), 10, 16, measure)).toBeCloseTo(2);
    });

    test('an unfilled rectangle is only its outline, so its middle is empty', () => {
        const r = shape('rect', 0, 0, 100, 60);
        expect(distanceTo(r, 50, 30, measure)).toBeGreaterThan(20);
        expect(distanceTo(r, 50, 0, measure)).toBeLessThan(0);
        expect(distanceTo(r, 100, 30, measure)).toBeLessThan(0);
    });

    test('an ellipse is hit on its rim and not at its centre', () => {
        const e = shape('ellipse', 0, 0, 200, 100);
        expect(distanceTo(e, 200, 50, measure)).toBeLessThan(1);
        expect(distanceTo(e, 100, 0, measure)).toBeLessThan(1);
        expect(distanceTo(e, 100, 50, measure)).toBeGreaterThan(40);
    });

    test('text is hit anywhere inside its box', () => {
        const t = text('hello');
        const b = layoutText(t, measure);
        expect(distanceTo(t, b.x + b.w / 2, b.y + b.h / 2, measure)).toBe(0);
        expect(distanceTo(t, b.x - 30, b.y, measure)).toBeCloseTo(30);
    });

    test('the eraser path touches things it sweeps across, even between samples', () => {
        const s = stroke([0, 100, 200, 100], 4);
        expect(touches(s, 100, 0, 100, 200, 8, measure)).toBe(true); // a fast swipe straight across
        expect(touches(s, 100, 0, 100, 50, 8, measure)).toBe(false); // stops short
    });
});

describe('svgPath', () => {
    test('too few points make no path', () => {
        expect(svgPath([[0, 0], [1, 1], [2, 2]])).toBe('');
    });

    test('a polygon becomes a closed quadratic path', () => {
        const d = svgPath([[0, 0], [10, 0], [10, 10], [0, 10]]);
        expect(d.startsWith('M0.00,0.00 Q10.00,0.00')).toBe(true);
        expect(d.endsWith('Z')).toBe(true);
    });
});
