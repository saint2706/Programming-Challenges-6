// @vitest-environment jsdom
import { describe, expect, test } from 'vitest';
import type { Item } from '../shared/types.ts';
import { esc, sceneBounds, toSvg } from '../src/export.ts';

const measure = (text: string, fs: number) => text.length * fs * 0.5;
const at = (item: Item, dx = 0, dy = 0) => ({ item, dx, dy });

const stroke: Item = { t: 'stroke', id: 's', author: 'x', color: '#d1242f', size: 6, pts: [0, 0, 40, 10, 80, 0, 120, 20] };
const rect: Item = { t: 'shape', id: 'r', author: 'x', color: '#0969da', size: 4, k: 'rect', x1: 10, y1: 50, x2: 90, y2: 120 };

function parse(svg: string): Document {
    const doc = new DOMParser().parseFromString(svg, 'image/svg+xml');
    expect(doc.querySelector('parsererror'), svg).toBeNull();
    return doc;
}

describe('toSvg', () => {
    test('an empty board exports nothing', () => {
        expect(toSvg([], measure)).toBeNull();
    });

    test('is well-formed XML and has one element per item inside a padded viewBox', () => {
        const svg = toSvg([at(stroke), at(rect)], measure)!;
        const doc = parse(svg);
        expect(doc.documentElement.namespaceURI).toBe('http://www.w3.org/2000/svg');
        expect(doc.querySelectorAll('path')).toHaveLength(1);
        expect(doc.querySelectorAll('rect')).toHaveLength(2); // the background and the shape
        const [x, y, w, h] = doc.documentElement.getAttribute('viewBox')!.split(' ').map(Number) as [number, number, number, number];
        const b = sceneBounds([at(stroke), at(rect)], measure)!;
        expect(x).toBeCloseTo(b.minX - 24);
        expect(y).toBeCloseTo(b.minY - 24);
        expect(w).toBeCloseTo(b.maxX - b.minX + 48);
        expect(h).toBeCloseTo(b.maxY - b.minY + 48);
    });

    test('every shape kind exports', () => {
        const kinds = (['line', 'arrow', 'rect', 'ellipse'] as const).map((k): Item => ({ t: 'shape', id: k, author: 'x', color: '#000000', size: 3, k, x1: 0, y1: 0, x2: 50, y2: 30 }));
        const doc = parse(toSvg(kinds.map((k) => at(k)), measure)!);
        expect(doc.querySelectorAll('line')).toHaveLength(1);
        expect(doc.querySelectorAll('ellipse')).toHaveLength(1);
        expect(doc.querySelectorAll('path')).toHaveLength(1); // the arrow
    });

    test('a moved item carries its offset and moves the bounds', () => {
        const svg = toSvg([at(rect, 100, 200)], measure)!;
        expect(svg).toContain('transform="translate(100 200)"');
        const b = sceneBounds([at(rect, 100, 200)], measure)!;
        expect(b.minX).toBeGreaterThan(100);
        expect(b.minY).toBeGreaterThan(200);
    });

    test('a highlighter is translucent', () => {
        const hl: Item = { ...stroke, id: 'h', hl: true } as Item;
        expect(toSvg([at(hl)], measure)).toContain('fill-opacity="0.35"');
    });

    test('text is escaped, so a note can never add markup to the file', () => {
        const evil: Item = { t: 'text', id: 't', author: 'x', color: '#000000', size: 20, x: 0, y: 0, sticky: false, text: '</text><script>alert(1)</script> & "quoted" \u0001\u0000' };
        const svg = toSvg([at(evil)], measure)!;
        const doc = parse(svg);
        expect(doc.querySelectorAll('script')).toHaveLength(0);
        expect(doc.querySelector('tspan')!.textContent).toBe('</text><script>alert(1)</script> & "quoted" ');
        expect(svg).not.toContain('<script');
        expect(svg).not.toMatch(/[\u0000-\u0008]/);
    });

    test('a sticky note has paper under its text and wraps long lines', () => {
        const note: Item = { t: 'text', id: 'n', author: 'x', color: '#fff3a3', size: 20, x: 0, y: 0, sticky: true, text: 'word '.repeat(40) };
        const doc = parse(toSvg([at(note)], measure)!);
        expect(doc.querySelectorAll('tspan').length).toBeGreaterThan(3);
        expect([...doc.querySelectorAll('rect')].some((r) => r.getAttribute('fill') === '#fff3a3')).toBe(true);
    });

    test('a colour that is not #rrggbb is replaced rather than copied into the file', () => {
        const bad = { ...rect, color: 'red" onload="alert(1)' } as Item;
        const svg = toSvg([at(bad)], measure)!;
        expect(svg).not.toContain('onload');
        parse(svg);
    });
});

test('esc handles the five XML specials and drops characters XML cannot carry', () => {
    expect(esc(`<a href="x">'&'</a>`)).toBe('&lt;a href=&quot;x&quot;&gt;&apos;&amp;&apos;&lt;/a&gt;');
    expect(esc('a\u0000b\u0008c퟿d￾e')).toBe('abc퟿de');
    expect(esc('tab\tnewline\ncr\r')).toBe('tab\tnewline\ncr\r');
});
